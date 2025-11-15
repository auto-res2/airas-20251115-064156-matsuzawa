import subprocess
import sys
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main orchestrator: merges run config & spawns training subprocess."""
    run_cfg_path = Path(__file__).resolve().parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(run_cfg_path)
    run_cfg = OmegaConf.load(run_cfg_path)

    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, run_cfg)
    cfg.run_id = cfg.get("run_id", run_cfg.get("run_id"))

    # Mode handling ---------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.dataset.batch_size = max(1, min(cfg.dataset.batch_size, 2))
        cfg.dataset.limit = cfg.dataset.get("limit", 50)
        print("[Main] TRIAL mode activated (fast checks)")
    else:
        cfg.wandb.mode = "online"

    results_dir = Path(cfg.results_dir) / cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, results_dir / "effective_config.yaml")

    # Add project root to PYTHONPATH for subprocess
    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("[Main] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
