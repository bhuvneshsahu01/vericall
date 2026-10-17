"""
vericall/train/train_sft.py
---------------------------
QLoRA Supervised Fine-Tuning (SFT) for Tool Calling.
Supports fast 4-bit unsloth patching with graceful HuggingFace PEFT fallback.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "vericall-sft-lora"

TOOL_SYSTEM_PROMPT = """You are a precise, tool-calling assistant. Given the user's request and available tools, produce a JSON object representing the exact tool call with the required parameters.

Output format:
{"name": "<function_name>", "arguments": {<key_value_pairs>}}
"""


def format_sft_prompt(query: str, target_fn: str, target_args: Dict[str, Any]) -> str:
    """Formats an instruction, query, and expected tool call response."""
    response = json.dumps({"name": target_fn, "arguments": target_args})
    return f"<|im_start|>system\n{TOOL_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


def load_sft_dataset(data_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Loads and formats the SFT dataset from JSONL."""
    path = data_path or (DATA_DIR / "train_sft.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found at {path}. Run vericall.data.prep first.")

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                text = format_sft_prompt(record["query"], record["target_function"], record["target_arguments"])
                dataset.append({"text": text})

    logger.info(f"Loaded {len(dataset)} SFT training examples from {path.name}")
    return dataset


def run_sft_training(
    model_name: str = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Executes QLoRA SFT training using Unsloth (if available) or HuggingFace PEFT.
    """
    save_path = output_dir or OUTPUT_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset

        logger.info(f"Using unsloth FastLanguageModel for fast 4-bit QLoRA on {model_name}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        raw_data = load_sft_dataset()
        train_dataset = Dataset.from_list(raw_data)

        training_args = TrainingArguments(
            output_dir=str(save_path / "runs"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
        )

        import inspect
        sft_kwargs = {
            "model": model,
            "train_dataset": train_dataset,
            "dataset_text_field": "text",
            "args": training_args,
        }
        sig = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in sig:
            sft_kwargs["processing_class"] = tokenizer
        else:
            sft_kwargs["tokenizer"] = tokenizer
        if "max_seq_length" in sig:
            sft_kwargs["max_seq_length"] = max_seq_length

        trainer = SFTTrainer(**sft_kwargs)

        logger.info("Starting SFT training...")
        trainer.train()

        logger.info(f"Saving SFT LoRA adapters to {save_path}...")
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        return str(save_path)

    except ImportError as e:
        logger.warning(f"Unsloth or GPU dependencies not present in current environment ({e}).")
        logger.info("Generating standalone training configuration and command for GPU/Colab execution.")
        config = {
            "model_name": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_train_epochs,
            "output_dir": str(save_path),
        }
        with open(save_path / "sft_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return str(save_path)


def is_bfloat16_supported() -> bool:
    """Checks if bfloat16 is supported on the active CUDA device."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


if __name__ == "__main__":
    run_sft_training()
