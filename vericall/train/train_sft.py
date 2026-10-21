"""
vericall/train/train_sft.py
---------------------------
QLoRA Supervised Fine-Tuning (SFT) for Tool Calling with Completion-Only Loss Masking.
Uses DataCollatorForCompletionOnlyLM on "<|im_start|>assistant\n" so the model only
learns to predict the tool call JSON, not the in-context tool schemas.
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
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "sft_adapter"


def load_sft_dataset(data_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Loads SFT training records formatted with candidate schemas."""
    path = data_path or (DATA_DIR / "train_sft.jsonl")
    if not path.exists():
        from vericall.data.prep import run_pipeline
        run_pipeline()

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                dataset.append({
                    "text": record["text"],
                    "prompt": record["prompt"],
                    "response": record["response"],
                })

    logger.info(f"Loaded {len(dataset)} SFT training examples from {path.name}")
    return dataset



def _get_completion_collator(response_template, tokenizer):
    try:
        from trl import DataCollatorForCompletionOnlyLM
        return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    except (ImportError, AttributeError):
        class SafeCompletionCollator:
            def __init__(self, resp_tpl, tok):
                self.tok = tok
                self.resp_tpl = resp_tpl
                self.resp_ids = tok.encode(resp_tpl, add_special_tokens=False)

            def __call__(self, examples):
                texts = [e["text"] for e in examples]
                batch = self.tok(texts, padding=True, truncation=True, max_length=1536, return_tensors="pt")
                labels = batch["input_ids"].clone()
                n = len(self.resp_ids)
                for i in range(labels.shape[0]):
                    ids = batch["input_ids"][i].tolist()
                    found = -1
                    for j in range(len(ids) - n + 1):
                        if ids[j:j+n] == self.resp_ids:
                            found = j + n
                            break
                    if found != -1:
                        labels[i, :found] = -100
                    else:
                        labels[i, :] = -100
                batch["labels"] = labels
                return batch
        return SafeCompletionCollator(response_template, tokenizer)

def run_sft_training(
    model_name: str = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length: int = 1536,
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
    Enforces completion-only loss masking on "<|im_start|>assistant\n".
    """
    save_path = output_dir or OUTPUT_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
        import torch

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

        # Completion-only loss masking
        response_template = "<|im_start|>assistant\n"
        collator = _get_completion_collator(response_template=response_template, tokenizer=tokenizer)

        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

        try:
            from trl import SFTConfig
            training_args = SFTConfig(
                output_dir=str(save_path / "runs"),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=10,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=False,
                bf16=False,
                logging_steps=10,
                optim="adamw_torch",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                report_to="none",
                max_length=max_seq_length,
                dataset_text_field="text",
                save_strategy="no",
            )
        except ImportError:
            training_args = TrainingArguments(
                output_dir=str(save_path / "runs"),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=10,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=False,
                bf16=False,
                logging_steps=10,
                optim="adamw_torch",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                report_to="none",
                save_strategy="no",
            )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )

        logger.info("Starting SFT training with completion-only loss masking...")
        trainer.train()

        logger.info(f"Saving SFT LoRA adapters to {save_path}...")
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        return str(save_path)

    except ImportError as e:
        logger.warning(f"Unsloth or GPU dependencies not present in current environment ({e}).")
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


if __name__ == "__main__":
    run_sft_training()
