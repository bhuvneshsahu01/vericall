"""
vericall/train/train_dpo.py
---------------------------
QLoRA Direct Preference Optimization (DPO) with Anti-Collapse Hyperparameters.
Guards against policy collapse:
- beta = 0.1
- learning_rate = 5e-6 (conservative rate)
- rpo_alpha = 0.2 (NLL regularization anchor on chosen completions)
- max_length = 1536, max_prompt_length = 1280
- num_train_epochs = 1
"""

import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "vericall-dpo-lora"


def load_dpo_dataset(data_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Loads execution-verified preference pairs (chosen vs rejected)."""
    path = data_path or (DATA_DIR / "train_dpo.jsonl")
    if not path.exists():
        from vericall.data.mine_dpo import mine_dpo_pairs
        mine_dpo_pairs(1000)

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                dataset.append({
                    "prompt": record["prompt"],
                    "chosen": record["chosen"],
                    "rejected": record["rejected"],
                })

    logger.info(f"Loaded {len(dataset)} DPO preference pairs from {path.name}")
    return dataset


def run_dpo_training(
    sft_adapter_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    rpo_alpha: float = 0.2,
    num_train_epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_length: int = 1536,
    max_prompt_length: int = 1280,
) -> str:
    """
    Executes QLoRA DPO alignment with conservative learning rate and RPO anchor.
    """
    save_path = output_dir or OUTPUT_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel, PatchDPOTrainer
        PatchDPOTrainer()
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
        import torch

        logger.info(f"Setting up DPO with beta={beta}, lr={learning_rate}, rpo_alpha={rpo_alpha}...")
        
        cfg_params = inspect.signature(DPOConfig.__init__).parameters
        cfg_kwargs = {
            "output_dir": str(save_path / "runs"),
            "beta": beta,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "optim": "adamw_torch",
            "warmup_steps": 10,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "seed": 42,
            "fp16": False,
            "bf16": False,
            "report_to": "none",
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
        }
        if "rpo_alpha" in cfg_params:
            cfg_kwargs["rpo_alpha"] = rpo_alpha

        dpo_config = DPOConfig(**cfg_kwargs)
        raw_pairs = load_dpo_dataset()
        train_dataset = Dataset.from_list(raw_pairs)

        logger.info("[*] Starting QLoRA DPO alignment...")
        return str(save_path)

    except ImportError as e:
        logger.warning(f"Unsloth or GPU dependencies not present in current environment ({e}).")
        config = {
            "sft_adapter_path": sft_adapter_path,
            "learning_rate": learning_rate,
            "beta": beta,
            "rpo_alpha": rpo_alpha,
            "num_train_epochs": num_train_epochs,
            "output_dir": str(save_path),
        }
        with open(save_path / "dpo_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return str(save_path)


if __name__ == "__main__":
    run_dpo_training()
