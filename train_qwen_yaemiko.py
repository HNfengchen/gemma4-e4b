"""
QLoRA Fine-tuning Script for Qwen3.5-2B Yaemiko Persona
Standard HuggingFace + PEFT training (no Unsloth/Triton dependency)
Optimized for RTX 4070 Laptop (8GB VRAM)
"""
import os
import sys
import json
import time
import torch
import psutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model, TaskType

DATASET_PATH = Path(r"e:\projects\agent\gemma4-e4b\dataset\shenzi\yaemiko_lines.jsonl")
OUTPUT_DIR = Path(__file__).parent / "lora_adapter_qwen"
LOG_DIR = Path(__file__).parent / "training_logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "E:/projects/agent/gemma4-e4b/models/unsloth/Qwen3___5-2B"

MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 5
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.1
SEED = 42

TRAINING_LOG = []


def log_event(msg: str):
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(entry)
    TRAINING_LOG.append(entry)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_environment():
    log_event("=== Environment Check ===")
    log_event(f"PyTorch: {torch.__version__}")
    log_event(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_event(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log_event(f"VRAM: {vram:.1f} GB")
        log_event(f"CUDA: {torch.version.cuda}")
        log_event(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    cpu_p = psutil.cpu_count(logical=False)
    cpu_l = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory().total / (1024**3)
    log_event(f"CPU: {cpu_p}P/{cpu_l}L")
    log_event(f"RAM: {ram:.1f} GB")


def load_dataset():
    log_event(f"Loading dataset from {DATASET_PATH}")
    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    log_event(f"Loaded {len(data)} samples")

    val_count = max(1, int(len(data) * VAL_SPLIT))
    indices = list(range(len(data)))
    random.shuffle(indices)
    val_indices = set(indices[:val_count])
    train_data = [data[i] for i in range(len(data)) if i not in val_indices]
    val_data = [data[i] for i in range(len(data)) if i in val_indices]
    log_event(f"Train: {len(train_data)} | Val: {len(val_data)}")
    return train_data, val_data


def format_chat_qwen(messages):
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return text


class YaemikoTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def main():
    start_time = time.time()
    set_seed(SEED)
    check_environment()

    train_data, val_data = load_dataset()

    log_event("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    log_event("Adding LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(examples):
        messages = examples["messages"]
        text = format_chat_qwen(messages)
        result = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        result["labels"] = [
            (tid if tid != tokenizer.pad_token_id else -100)
            for tid in result["input_ids"]
        ]
        return result

    log_event("Tokenizing datasets...")
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    train_tokenized = train_ds.map(
        tokenize_fn,
        batched=False,
        remove_columns=["messages"],
    )
    val_tokenized = val_ds.map(
        tokenize_fn,
        batched=False,
        remove_columns=["messages"],
    )
    log_event(f"Tokenized: train={len(train_tokenized)}, val={len(val_tokenized)}")

    bf16 = torch.cuda.is_bf16_supported()
    fp16 = not bf16

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = YaemikoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    total_steps = (len(train_tokenized) * EPOCHS) // effective_batch
    log_event("=== Training Configuration ===")
    log_event(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, GradAccum: {GRADIENT_ACCUMULATION_STEPS}")
    log_event(f"Effective batch: {effective_batch}, Total steps: ~{total_steps}")
    log_event(f"LR: {LEARNING_RATE}, LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    log_event(f"FP16={fp16}, BF16={bf16}")

    log_event("Starting training...")
    train_result = trainer.train()

    log_event("=== Training Results ===")
    log_event(f"Train loss: {train_result.training_loss:.4f}")
    log_event(f"Train runtime: {train_result.metrics['train_runtime']:.1f}s")

    eval_result = trainer.evaluate()
    log_event(f"Eval loss: {eval_result['eval_loss']:.4f}")
    ppl = torch.exp(torch.tensor(eval_result['eval_loss']))
    log_event(f"Perplexity: {ppl:.2f}")

    log_event("Saving LoRA adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    log_event(f"Saved to: {OUTPUT_DIR}")

    elapsed = time.time() - start_time
    log_event(f"Total time: {elapsed/60:.1f} min")

    mem_alloc = 0
    mem_reserved = 0
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        log_event(f"Peak GPU memory allocated: {mem_alloc:.2f} GB")
        log_event(f"Peak GPU memory reserved: {mem_reserved:.2f} GB")

    report = {
        "model": "Qwen3.5-2B",
        "dataset": str(DATASET_PATH),
        "dataset_size": len(train_data) + len(val_data),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "hyperparameters": {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": effective_batch,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "max_seq_length": MAX_SEQ_LENGTH,
            "fp16": fp16,
            "bf16": bf16,
            "seed": SEED,
        },
        "results": {
            "train_loss": round(train_result.training_loss, 4),
            "eval_loss": round(eval_result["eval_loss"], 4),
            "perplexity": round(float(ppl), 2),
            "train_runtime_seconds": round(train_result.metrics["train_runtime"], 1),
            "total_time_seconds": round(elapsed, 1),
            "total_time_minutes": round(elapsed / 60, 1),
        },
        "resources": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if torch.cuda.is_available() else 0,
            "peak_gpu_allocated_gb": round(mem_alloc, 2),
            "peak_gpu_reserved_gb": round(mem_reserved, 2),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
        },
        "log": TRAINING_LOG,
    }

    report_path = LOG_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log_event(f"Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Train Loss: {train_result.training_loss:.4f}")
    print(f"Eval Loss:  {eval_result['eval_loss']:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Time:       {elapsed/60:.1f} min")
    print(f"Adapter:    {OUTPUT_DIR}")
    print(f"Report:     {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
