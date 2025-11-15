import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


def _save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _learning_curve(history: pd.DataFrame, run_id: str, save_dir: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    if "train_loss" in history:
        sns.lineplot(x=history.index, y=history["train_loss"], label="train_loss", ax=ax1)
    if "val_loss" in history:
        sns.lineplot(x=history.index, y=history["val_loss"], label="val_loss", ax=ax1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    if "val_acc" in history:
        sns.lineplot(
            x=history.index, y=history["val_acc"], label="val_acc", ax=ax2, color="green"
        )
        ax2.set_ylabel("Val Accuracy")
    fig.tight_layout()
    out_path = save_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _confusion_heatmap(conf_matrix: List[List[int]], labels: List[str], run_id: str, save_dir: Path) -> Path:
    cm = np.array(conf_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    out_path = save_dir / f"{run_id}_confusion_matrix.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _box_plot(metric_map: Dict[str, float], ylabel: str, out_dir: Path) -> Path:
    df = pd.DataFrame(list(metric_map.items()), columns=["run_id", "value"])
    df["group"] = np.where(df["run_id"].str.contains("proposed"), "proposed", "baseline")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="group", y="value", data=df, ax=ax)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path = out_dir / f"comparison_{ylabel.replace(' ', '_')}_box.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _bar_chart(metric_map: Dict[str, float], ylabel: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, len(metric_map) * 0.8), 4))
    sns.barplot(x=list(metric_map.keys()), y=list(metric_map.values()), ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    for idx, (rid, val) in enumerate(metric_map.items()):
        ax.text(idx, val, f"{val:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    path = out_dir / f"comparison_{ylabel.replace(' ', '_')}_bar.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _ttest(metric_map: Dict[str, float]) -> Dict[str, Any]:
    groups = {"proposed": [], "baseline": []}
    for rid, val in metric_map.items():
        groups["proposed" if "proposed" in rid else "baseline"].append(val)
    if len(groups["proposed"]) < 2 or len(groups["baseline"]) < 2:
        return {"p_value": None, "t_stat": None}
    t_stat, p_val = stats.ttest_ind(groups["proposed"], groups["baseline"], equal_var=False)
    return {"p_value": float(p_val), "t_stat": float(t_stat)}


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Comprehensive evaluation & visualisation via WandB")
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON list of wandb run_ids")
    args = parser.parse_args()

    run_ids: List[str] = json.loads(args.run_ids)
    out_root = Path(args.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load global config to fetch WandB credentials & primary metric name ----
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    entity, project = base_cfg["wandb"]["entity"], base_cfg["wandb"]["project"]
    primary_metric_name = base_cfg.get("primary_metric", "final_test_acc")

    api = wandb.Api()

    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    primary_metric_values: Dict[str, float] = {}
    generated_paths: List[str] = []

    # ------------------------------------------------------------------
    # Per-run processing
    # ------------------------------------------------------------------
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        history = run.history()  # pandas DataFrame with all time-series metrics
        summary = run.summary._json_dict
        config = dict(run.config)

        run_dir = out_root / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # Export metrics JSON ------------------------------------------------
        metrics_path = run_dir / "metrics.json"
        _save_json(
            {
                "summary": summary,
                "config": config,
                "history": history.to_dict(orient="list"),
            },
            metrics_path,
        )
        generated_paths.append(str(metrics_path))

        # Learning curve ----------------------------------------------------
        lc_path = _learning_curve(history, rid, run_dir)
        generated_paths.append(str(lc_path))

        # Confusion matrix --------------------------------------------------
        conf = summary.get("test_confusion") or summary.get("val_confusion")
        if conf and "matrix" in conf:
            cm_path = _confusion_heatmap(conf["matrix"], conf["labels"], rid, run_dir)
            generated_paths.append(str(cm_path))

        # Aggregate scalar metrics -----------------------------------------
        for k, v in summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                aggregated_metrics.setdefault(k, {})[rid] = v
        primary_metric_values[rid] = summary.get(primary_metric_name, np.nan)

    # ------------------------------------------------------------------
    # Aggregated analysis
    # ------------------------------------------------------------------
    comp_dir = out_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    best_prop = max(
        ((r, v) for r, v in primary_metric_values.items() if "proposed" in r),
        key=lambda x: x[1],
        default=(None, np.nan),
    )
    best_base = max(
        (
            (r, v)
            for r, v in primary_metric_values.items()
            if "baseline" in r or "comparative" in r or "sketch" in r
        ),
        key=lambda x: x[1],
        default=(None, np.nan),
    )

    minimise_keywords = ["loss", "error", "perplexity", "time"]
    higher_is_better = not any(k in primary_metric_name.lower() for k in minimise_keywords)
    gap = np.nan
    if not np.isnan(best_prop[1]) and not np.isnan(best_base[1]):
        improvement = (best_prop[1] - best_base[1]) / abs(best_base[1]) * 100
        gap = improvement if higher_is_better else -improvement

    aggregated_dict = {
        "primary_metric": primary_metric_name,
        "metrics": aggregated_metrics,
        "best_proposed": {"run_id": best_prop[0], "value": best_prop[1]},
        "best_baseline": {"run_id": best_base[0], "value": best_base[1]},
        "gap": gap,
        "stat_test": _ttest(primary_metric_values),
    }

    agg_json_path = comp_dir / "aggregated_metrics.json"
    _save_json(aggregated_dict, agg_json_path)
    generated_paths.append(str(agg_json_path))

    # Comparison figures ---------------------------------------------------
    bar_path = _bar_chart(primary_metric_values, primary_metric_name, comp_dir)
    box_path = _box_plot(primary_metric_values, primary_metric_name, comp_dir)
    generated_paths.extend([str(bar_path), str(box_path)])

    # Print all generated paths -------------------------------------------
    for p in generated_paths:
        print(p)


if __name__ == "__main__":
    main()
