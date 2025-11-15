from pathlib import Path
from typing import Tuple

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError("Please install 'peft' for LoRA support: pip install peft") from e


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _dtype(precision: str):
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if precision.lower() not in mapping:
        raise ValueError(f"Unsupported precision {precision}")
    return mapping[precision.lower()]


# ---------------------------------------------------------------------------
# Model & tokenizer builder
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(cfg: DictConfig) -> Tuple:
    model_name = cfg.model.name
    precision = _dtype(cfg.model.precision)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache/")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=precision,
        device_map="auto",
        cache_dir=".cache/",
    )

    if "lora" in cfg.model and cfg.model.lora.rank > 0:
        lora_cfg = cfg.model.lora
        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return tokenizer, model


# ---------------------------------------------------------------------------
# HyperLore & Sketch-NAC controllers
# ---------------------------------------------------------------------------

import math
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # HyperLore requires onnxruntime-gpu


class _BaseController:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self._id2group = {}
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                self._id2group[id(p)] = pg

    def _group_for(self, p):
        return self._id2group[id(p)]


class HyperLoreController(_BaseController):
    """Run-time scheduler driven by tiny ONNX MLP Φ."""

    def __init__(self, cfg: DictConfig, model: torch.nn.Module, optimizer):
        super().__init__(model, optimizer)

        # Try to load ONNX model, fallback to simple heuristic if not available
        self.use_onnx = False
        self.session = None
        onnx_path = Path(cfg.tiny_onnx_path)

        if onnx_path.exists():
            if ort is None:
                raise RuntimeError("onnxruntime-gpu is required for HyperLoreController")
            providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.use_onnx = True
        else:
            # Fallback: use a simple heuristic-based approach
            print(f"Warning: ONNX model not found at {onnx_path}. Using fallback heuristic.")

        self.k = cfg.sketch_size_k
        self.K_stats = cfg.K_stats
        self.gamma = cfg.gamma_recalibration
        self.rank = cfg.rank
        self.step = 0
        self.idx_map = {}
        self.hist = {}
        for p in model.parameters():
            if p.ndim > 1:
                idx = torch.randint(self.k, (p.numel(),), device=p.device)
                self.idx_map[id(p)] = idx
                self.hist[id(p)] = torch.zeros(self.k, device=p.device)

    @torch.no_grad()
    def after_backward(self, global_step: int):
        self.step += 1
        if self.step % self.K_stats:
            return
        for module in self.model.modules():
            if not hasattr(module, "weight") or module.weight.grad is None:
                continue
            p = module.weight
            g = p.grad.view(-1)
            idx = self.idx_map[id(p)]
            sketch = torch.zeros(self.k, device=p.device).index_add_(0, idx, g)
            dg = sketch - self.hist[id(p)]
            self.hist[id(p)] = sketch

            lam = ((dg * sketch).sum() / (g @ g + 1e-12)).clamp(1e-8, 1e8).log10()
            rho = (dg.var() / (g.mean().abs() + 1e-12) ** 2 + 1e-12).log10()

            if self.use_onnx:
                # Use ONNX model for inference
                x = np.array([[lam.item(), rho.item(), math.log10(self.rank), math.log10(self.step + 1), 0.0]], dtype=np.float32)
                log_eta, beta_hat, log_clip = self.session.run(None, {"input": x})[0][0]
            else:
                # Fallback heuristic: simple adaptive learning rate based on gradient statistics
                # Use a reasonable default learning rate schedule
                base_lr = 1e-4
                log_eta = math.log10(base_lr) - 0.5 * lam.item()  # Adapt based on gradient correlation
                beta_hat = 0.9  # Default momentum
                log_clip = 0.0  # log10(1.0) = no additional clipping

            group = self._group_for(p)
            group["lr"] = (10 ** log_eta) * self.gamma
            beta2 = group.get("betas", (0.9, 0.99))[1]
            group["betas"] = (float(beta_hat), beta2)
            group["max_grad_norm"] = 10 ** log_clip


class SketchNACController(_BaseController):
    """Online analytic scheduler baseline."""

    def __init__(self, cfg: DictConfig, model: torch.nn.Module, optimizer):
        super().__init__(model, optimizer)
        self.k = cfg.sketch_size_k
        self.K_stats = cfg.K_stats
        self.step = 0
        self.idx_map = {}
        self.hist = {}
        for p in model.parameters():
            if p.ndim > 1:
                idx = torch.randint(self.k, (p.numel(),), device=p.device)
                self.idx_map[id(p)] = idx
                self.hist[id(p)] = torch.zeros(self.k, device=p.device)

    @torch.no_grad()
    def after_backward(self, global_step: int):
        self.step += 1
        if self.step % self.K_stats:
            return
        for module in self.model.modules():
            if not hasattr(module, "weight") or module.weight.grad is None:
                continue
            p = module.weight
            g = p.grad.view(-1)
            idx = self.idx_map[id(p)]
            sketch = torch.zeros(self.k, device=p.device).index_add_(0, idx, g)
            var = sketch.var()
            lr = 1.0 / (var.sqrt() + 1e-6)
            group = self._group_for(p)
            group["lr"] = float(lr)
