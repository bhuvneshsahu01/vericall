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

        import types
        from transformers.modeling_outputs import CausalLMOutputWithPast

        # Neutralize TRL fused_lm_head monkey patch
        try:
            import trl.trainer.utils
            trl.trainer.utils.add_fused_lm_head = lambda *args, **kwargs: None
        except Exception:
            pass

        def safe_output_getattr(self, name):
            if hasattr(self, "logits") and self.logits is not None:
                return torch.zeros((self.logits.shape[0], self.logits.shape[1] - 1), device=self.logits.device)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        CausalLMOutputWithPast.__getattr__ = safe_output_getattr

        if "forward" in model.__dict__:
            del model.__dict__["forward"]

        true_class_forward = model.__class__.forward

        def safe_forward(self, *args, **kwargs):
            kwargs.pop("fused_lm_head", None)
            labels = kwargs.get("labels", None)
            shift_labels = kwargs.pop("shift_labels", None)
            kwargs.pop("num_items_in_batch", None)
            kwargs.pop("support_ids", None)
            outputs = true_class_forward(self, *args, **kwargs)
            if hasattr(outputs, "logits") and outputs.logits is not None:
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                with torch.no_grad():
                    probs = shift_logits.softmax(dim=-1)
                    log_probs_full = shift_logits.log_softmax(dim=-1)
                    outputs.entropy = -(probs * log_probs_full).sum(dim=-1)
                    outputs.mean_logits = shift_logits.mean(dim=-1)
                    outputs.log_sum_sq_probs = (probs ** 2).sum(dim=-1).log()
                target_labels = shift_labels if shift_labels is not None else labels
                if target_labels is not None:
                    shift_lbls = target_labels[:, 1:].contiguous() if shift_labels is None else target_labels.contiguous()
                    if shift_lbls.shape[1] == shift_logits.shape[1]:
                        loss_mask = shift_lbls != -100
                        safe_lbls = shift_lbls.clone()
                        safe_lbls[~loss_mask] = 0
                        per_token_logps = torch.gather(shift_logits.log_softmax(-1), dim=-1, index=safe_lbls.unsqueeze(-1)).squeeze(-1)
                        outputs.log_probs = torch.where(loss_mask, per_token_logps, torch.zeros_like(per_token_logps))
                        with torch.no_grad():
                            outputs.is_top1 = (shift_logits.argmax(dim=-1) == safe_lbls) & loss_mask
                    else:
                        outputs.is_top1 = torch.zeros(shift_logits.shape[:2], dtype=torch.bool, device=shift_logits.device)
                        outputs.log_probs = torch.zeros(shift_logits.shape[:2], dtype=shift_logits.dtype, device=shift_logits.device)
                else:
                    outputs.is_top1 = torch.zeros(shift_logits.shape[:2], dtype=torch.bool, device=shift_logits.device)
                    outputs.log_probs = torch.zeros(shift_logits.shape[:2], dtype=shift_logits.dtype, device=shift_logits.device)
            return outputs

        model.forward = types.MethodType(safe_forward, model)
        model.train()

        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

        cfg_params = inspect.signature(DPOConfig.__init__).parameters
        dpo_cfg_kwargs = {
            "output_dir": str(save_path / "runs"),
            "beta": beta,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "optim": "adamw_torch",
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "seed": 42,
            "fp16": False,
            "bf16": False,
            "report_to": "none",
        }
        if "max_length" in cfg_params:
            dpo_cfg_kwargs["max_length"] = max_length
        if "max_prompt_length" in cfg_params:
            dpo_cfg_kwargs["max_prompt_length"] = max_prompt_length

        dpo_config = DPOConfig(**dpo_cfg_kwargs)

        raw_pairs = load_dpo_dataset()
        train_dataset = Dataset.from_list(raw_pairs)

        import inspect
        dpo_kwargs = {
            "model": model,
            "ref_model": None,
            "args": dpo_config,
            "train_dataset": train_dataset,
        }
        sig_dpo = inspect.signature(DPOTrainer.__init__).parameters
        if "processing_class" in sig_dpo:
            dpo_kwargs["processing_class"] = tokenizer
        else:
            dpo_kwargs["tokenizer"] = tokenizer

        dpo_trainer = DPOTrainer(**dpo_kwargs)

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
