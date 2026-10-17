"""
vericall/train/train_dpo.py
---------------------------
QLoRA Direct Preference Optimization (DPO) for Tool Calling.
Optimizes execution ground truth over execution-verified near-miss mutations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
SFT_ADAPTER_DIR = CHECKPOINT_DIR / "vericall-sft-lora"
DPO_OUTPUT_DIR = CHECKPOINT_DIR / "vericall-dpo-lora"

TOOL_SYSTEM_PROMPT = """You are a precise, tool-calling assistant. Given the user's request and available tools, produce a JSON object representing the exact tool call with the required parameters.

Output format:
{"name": "<function_name>", "arguments": {<key_value_pairs>}}
"""


def format_dpo_prompt(query: str) -> str:
    """Formats the prompt prefix for DPO pairs."""
    return f"<|im_start|>system\n{TOOL_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


def load_dpo_dataset(data_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Loads and formats the DPO dataset from JSONL."""
    path = data_path or (DATA_DIR / "train_dpo.jsonl")
    if not path.exists():
        from vericall.data.mine_dpo import mine_dpo_pairs
        mine_dpo_pairs()

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                prompt = format_dpo_prompt(record["prompt"])
                chosen = record["chosen"] + "<|im_end|>"
                rejected = record["rejected"] + "<|im_end|>"
                dataset.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

    logger.info(f"Loaded {len(dataset)} DPO preference pairs from {path.name}")
    return dataset


def run_dpo_training(
    base_model_name: str = "unsloth/Qwen2.5-7B-Instruct",
    sft_adapter_path: Optional[Path] = None,
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    num_train_epochs: int = 1,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Executes QLoRA DPO alignment using TRL DPOTrainer.
    """
    save_path = output_dir or DPO_OUTPUT_DIR
    save_path.mkdir(parents=True, exist_ok=True)
    adapter_src = sft_adapter_path or SFT_ADAPTER_DIR

    try:
        from unsloth import FastLanguageModel, PatchDPOTrainer
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset

        PatchDPOTrainer()
        logger.info(f"Loading base model {base_model_name} with SFT adapter {adapter_src} for DPO...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_src) if adapter_src.exists() else base_model_name,
            max_seq_length=max_length,
            load_in_4bit=True,
        )

        dpo_config = DPOConfig(
            output_dir=str(save_path / "runs"),
            beta=beta,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            optim="adamw_8bit",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=10,
            seed=42,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )

        raw_pairs = load_dpo_dataset()
        train_dataset = Dataset.from_list(raw_pairs)

        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Unsloth handles reference model implicitly via LoRA adapter disabling
            args=dpo_config,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        logger.info(f"Starting DPO alignment with beta={beta}...")
        dpo_trainer.train()

        logger.info(f"Saving DPO LoRA adapters to {save_path}...")
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        return str(save_path)

    except ImportError as e:
        logger.warning(f"Unsloth or GPU dependencies not present in current environment ({e}).")
        logger.info("Generating standalone DPO training configuration for GPU/Colab execution.")
        config = {
            "base_model": base_model_name,
            "sft_adapter": str(adapter_src),
            "beta": beta,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "output_dir": str(save_path),
        }
        with open(save_path / "dpo_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return str(save_path)


if __name__ == "__main__":
    run_dpo_training()
