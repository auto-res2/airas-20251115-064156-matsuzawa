import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import (
    HyperLoreController,
    SketchNACController,
    build_model_and_tokenizer,
)
from .preprocess import build_datasets, get_data_collator

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _maybe_profile_start(device: torch.device):
    """Return a callable that, when invoked, returns step-time in milliseconds."""
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        def _stop() -> float:
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)  # ms

        return _stop
    # CPU fallback
    wall_start = time.perf_counter_ns()

    def _stop() -> float:
        return (time.perf_counter_ns() - wall_start) / 1e6

    return _stop


def _configure_wandb(cfg: DictConfig):
    """Initialise Weights & Biases unless disabled."""
    if cfg.wandb.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run_id,
        mode=cfg.wandb.mode,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"[wandb]  URL: {run.url}")
    return run


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

def _init_controller(cfg: DictConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if cfg.training.scheduler == "hyperlore":
        hyp_cfg = OmegaConf.merge(cfg.training.hyperlore, {})
        if "rank" not in hyp_cfg or hyp_cfg.rank is None:
            hyp_cfg.rank = cfg.model.lora.rank
        return HyperLoreController(hyp_cfg, model, optimizer)
    if cfg.training.scheduler == "sketch_nac":
        return SketchNACController(cfg.training.sketch_nac, model, optimizer)
    return None

# ---------------------------------------------------------------------------
# Training / evaluation routines
# ---------------------------------------------------------------------------

def _train_one_epoch(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    controller,
    dataloader: DataLoader,
    epoch: int,
    run: Optional[wandb.sdk.wandb_run.Run],
) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.train()
    losses: List[float] = []
    step_times: List[float] = []
    global_step_base = (epoch - 1) * len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for batch_idx, batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        stop_timer = _maybe_profile_start(device)
        outputs = model(**batch)
        loss = outputs.loss / cfg.training.accumulation_steps
        loss.backward()

        if controller is not None:
            controller.after_backward(global_step_base + batch_idx)

        if (batch_idx + 1) % cfg.training.accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        elapsed_ms = stop_timer()
        losses.append(loss.item() * cfg.training.accumulation_steps)
        step_times.append(elapsed_ms)

        if run is not None:
            wandb.log(
                {
                    "train_loss": losses[-1],
                    "step_time_ms": elapsed_ms,
                    "epoch": epoch,
                },
                step=global_step_base + batch_idx,
            )
        pbar.set_description(
            f"Epoch {epoch} | loss={np.mean(losses):.4f} | {elapsed_ms:.1f} ms"
        )

    return float(np.mean(losses)), float(np.mean(step_times))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataloader: DataLoader,
    split: str,
) -> Dict[str, object]:
    from sklearn.metrics import confusion_matrix  # local import to avoid overhead when unused

    device = next(model.parameters()).device
    model.eval()
    losses, step_times = [], []
    all_preds: List[int] = []
    all_labels: List[int] = []

    pbar = tqdm(dataloader, leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            stop_timer = _maybe_profile_start(device)
            outputs = model(**batch)
            loss = outputs.loss
            elapsed = stop_timer()
        losses.append(loss.item())
        step_times.append(elapsed)

        preds = torch.argmax(outputs.logits, dim=-1)
        mask = batch["labels"] != -100
        preds = preds.masked_select(mask).cpu().tolist()
        labels = batch["labels"].masked_select(mask).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
        pbar.set_description(f"{split.capitalize()} loss={np.mean(losses):.4f}")

    accuracy = float(sum(p == l for p, l in zip(all_preds, all_labels))) / max(
        len(all_labels), 1
    )

    # Confusion matrix over most common tokens (top-k + "other") -----------
    top_k = getattr(cfg.dataset, "confusion_top_k", 20)
    most_common = [token for token, _ in Counter(all_labels).most_common(top_k)]
    label_set = most_common + [-1]  # -1 stands for "other"

    def _map(t: int):
        return t if t in most_common else -1

    mapped_preds = [_map(t) for t in all_preds]
    mapped_labels = [_map(t) for t in all_labels]
    cm = confusion_matrix(mapped_labels, mapped_preds, labels=label_set)
    class_labels = [str(t) for t in most_common] + ["other"]

    return {
        f"{split}_loss": float(np.mean(losses)),
        f"{split}_acc": accuracy,
        f"{split}_step_time_ms": float(np.mean(step_times)),
        f"{split}_confusion": {"labels": class_labels, "matrix": cm.tolist()},
    }


# ---------------------------------------------------------------------------
# Optuna objective (search without WandB I/O)
# ---------------------------------------------------------------------------

def _objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True, deepcopy=True))

    # Sample hyper-parameters -------------------------------------------------
    for name, space in cfg.optuna.search_space.items():
        if space.type == "categorical":
            val = trial.suggest_categorical(name, space.choices)
        elif space.type == "loguniform":
            val = trial.suggest_float(name, space.low, space.high, log=True)
        elif space.type == "uniform":
            val = trial.suggest_float(name, space.low, space.high)
        else:
            raise ValueError(f"Unsupported space type {space.type}")
        if cfg.training.scheduler == "hyperlore":
            cfg.training.hyperlore[name] = val
        else:
            cfg.training.sketch_nac[name] = val

    tokenizer, model = build_model_and_tokenizer(cfg)
    train_ds, val_ds, _ = build_datasets(cfg, tokenizer)
    collator = get_data_collator(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.base_learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    controller = _init_controller(cfg, model, optimizer)

    best_acc = 0.0
    for epoch in range(1, cfg.training.epochs + 1):
        _train_one_epoch(cfg, model, optimizer, controller, train_loader, epoch, run=None)
        metrics = _evaluate(cfg, model, val_loader, "val")
        acc = metrics["val_acc"]
        best_acc = max(best_acc, acc)
        trial.report(acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_acc


# ---------------------------------------------------------------------------
# Main training workflow (with WandB)
# ---------------------------------------------------------------------------

def _only_scalar(d: Dict[str, object]) -> Dict[str, float]:
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


def _run_training(cfg: DictConfig):
    run = _configure_wandb(cfg)

    tokenizer, model = build_model_and_tokenizer(cfg)
    train_ds, val_ds, test_ds = build_datasets(cfg, tokenizer)
    collator = get_data_collator(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.base_learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    controller = _init_controller(cfg, model, optimizer)

    best_val_acc, best_epoch = 0.0, 0
    train_step_times: List[float] = []

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_step_time = _train_one_epoch(
            cfg, model, optimizer, controller, train_loader, epoch, run
        )
        train_step_times.append(train_step_time)

        val_metrics = _evaluate(cfg, model, val_loader, "val")
        test_metrics = _evaluate(cfg, model, test_loader, "test")

        if run is not None:
            wandb.log(
                {
                    **_only_scalar(val_metrics),
                    **_only_scalar(test_metrics),
                    "epoch": epoch,
                    "epoch_train_loss": train_loss,
                },
                step=epoch * len(train_loader),
            )

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc, best_epoch = val_metrics["val_acc"], epoch
            ckpt_dir = Path(cfg.results_dir) / cfg.run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            tokenizer.save_pretrained(ckpt_dir)

    mean_step_time_ms = float(np.mean(train_step_times)) if train_step_times else np.nan

    if run is not None:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["final_test_acc"] = test_metrics.get("test_acc", np.nan)
        wandb.summary["mean_step_time_ms"] = mean_step_time_ms
        # Store confusion matrix in summary for downstream evaluation
        wandb.summary["test_confusion"] = test_metrics.get("test_confusion")
        run.finish()


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def train_entry(cfg: DictConfig):
    # Merge run-specific configuration --------------------------------------
    run_cfg_path = (
        Path(__file__).resolve().parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    )
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Run-config not found: {run_cfg_path}")
    run_cfg = OmegaConf.load(run_cfg_path)
    # Disable struct mode to allow new keys from run_cfg
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, run_cfg)

    cfg.run_id = cfg.get("run_id", run_cfg.get("run_id"))

    # Mode-specific overrides ----------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.dataset.batch_size = max(1, min(cfg.dataset.batch_size, 2))
        cfg.dataset.limit = cfg.dataset.get("limit", 50)
        print("[Mode] TRIAL – quick check (1 epoch, limited data, wandb disabled)")
    else:
        cfg.wandb.mode = "online"

    # Optuna search ---------------------------------------------------------
    if cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(lambda t: _objective(t, cfg), n_trials=cfg.optuna.n_trials)
        print(
            f"[Optuna] Best value={study.best_value:.4f} | Params={study.best_params}"
        )
        for k, v in study.best_params.items():
            if cfg.training.scheduler == "hyperlore":
                cfg.training.hyperlore[k] = v
            else:
                cfg.training.sketch_nac[k] = v

    # Save effective configuration -----------------------------------------
    eff_dir = Path(cfg.results_dir) / cfg.run_id
    eff_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, eff_dir / "effective_config.yaml")

    # Launch training -------------------------------------------------------
    _run_training(cfg)


if __name__ == "__main__":
    train_entry()
