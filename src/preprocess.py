from typing import Tuple

from datasets import load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, default_data_collator

SPECIAL_TOKENS = {"question": "Question:", "answer": "Answer:"}


def _format_example(ex):
    q, a = ex["question"].strip(), ex["answer"].strip()
    prompt = f"{SPECIAL_TOKENS['question']} {q}\n{SPECIAL_TOKENS['answer']}"
    return {
        "prompt": prompt,
        "full_text": f"{prompt} {a}",
        "answer": a,
    }


def build_datasets(cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> Tuple:
    """Load & preprocess GSM8K-like dataset for causal-LM fine-tuning."""
    ds = load_dataset(cfg.dataset.name, cfg.dataset.get("config", None), cache_dir=".cache/")

    if "validation" not in ds:
        val_frac = cfg.dataset.split.validation_fraction
        ds_split = ds[cfg.dataset.split.train].train_test_split(
            test_size=val_frac, seed=cfg.training.seed
        )
        ds = {
            "train": ds_split["train"],
            "validation": ds_split["test"],
            "test": ds_split["test"],
        }
    else:
        ds = {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds.get("test", ds["validation"]),
        }

    ds = {k: v.map(_format_example, remove_columns=v.column_names) for k, v in ds.items()}

    def _tok(batch):
        tok_full = tokenizer(
            batch["full_text"],
            truncation=True,
            max_length=cfg.dataset.max_length,
            padding="max_length",
        )
        tok_prompt = tokenizer(batch["prompt"], truncation=True, max_length=cfg.dataset.max_length)[
            "input_ids"
        ]
        # Create a proper deep copy of input_ids for labels
        import copy
        labels = copy.deepcopy(tok_full["input_ids"])
        for i, p_ids in enumerate(tok_prompt):
            n = len(p_ids)
            for j in range(n):
                labels[i][j] = -100  # mask prompt tokens
        tok_full["labels"] = labels
        return tok_full

    tokenised = {k: v.map(_tok, batched=True, remove_columns=v.column_names) for k, v in ds.items()}

    # Trial-mode subset
    if getattr(cfg.dataset, "limit", None):
        lim = int(cfg.dataset.limit)
        tokenised = {k: d.select(range(min(lim, len(d)))) for k, d in tokenised.items()}

    return tokenised["train"], tokenised["validation"], tokenised["test"]


def get_data_collator(tokenizer):
    return default_data_collator
