#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train2.py
Скрипт для параметрического запуска обучения (SFT + LoRA/QLoRA).
Принимает гиперпараметры через аргументы CLI для проведения экспериментов.
Использует датасет prompts2.jsonl.
"""

import os
import sys
import json
import yaml
import argparse
import logging
import random  # <--- ДОБАВЛЕНО
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("train2")

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for LoRA")
    
    # Гиперпараметры эксперимента
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--quantization", type=str, choices=["fp16", "nf4"], default="nf4", help="FP16 (LoRA) or NF4 (QLoRA)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    # Системные параметры
    parser.add_argument("--output_dir", type=str, default="outputs2/exp_default")
    parser.add_argument("--dataset_path", type=str, default="data/prompts2.jsonl")
    parser.add_argument("--model_name", type=str, default=None, help="Если не задано, берется из config.yaml")
    
    return parser.parse_args()

def load_config():
    """Загрузка базового конфига для промптов и имени модели."""
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def fix_json(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return {}
    except:
        return {}

def calculate_metrics_custom(model, tokenizer, dataset, device):
    """
    Ручной подсчет метрик: JSON Validity, Exact Match, Slot-F1.
    """
    model.eval()
    valid_count = 0
    exact_match = 0
    y_true_flat = []
    y_pred_flat = []
    
    # Ключи для F1
    keys = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "number"]
    
    logger.info("Начало валидации...")
    prompts = dataset["text_prompt"] # Мы сохраним чистый промпт при создании датасета
    gts = dataset["labels_json"]
    
    # Инференс батчами (по 1 для надежности в eval)
    for i in range(len(prompts)):
        prompt = prompts[i]
        gt = gts[i]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Отрезаем промпт из ответа, если модель его повторила (хотя skip_special_tokens часто помогает)
        # Llama часто повторяет промпт, поэтому ищем маркер ответа или вырезаем начало
        if "**Ответ:**" in generated_text:
            response = generated_text.split("**Ответ:**")[-1].strip()
        else:
            # Fallback: просто пытаемся найти JSON в тексте
            response = generated_text
        
        # 1. Validity
        try:
            pred_json = fix_json(response)
            if pred_json:
                valid_count += 1
        except:
            pred_json = {}

        # Нормализация для сравнения
        norm_pred = {k: str(pred_json.get(k, "")).strip() for k in keys}
        norm_gt = {k: str(gt.get(k, "")).strip() for k in keys}
        
        # 2. Exact Match
        if norm_pred == norm_gt:
            exact_match += 1
            
        # 3. Data for F1
        for k in keys:
            y_true_flat.append(norm_gt[k])
            y_pred_flat.append(norm_pred[k])

    total = len(dataset)
    metrics = {
        "json_validity": valid_count / total if total > 0 else 0,
        "exact_match": exact_match / total if total > 0 else 0,
        "slot_f1": f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    }
    return metrics

def main():
    args = parse_args()
    cfg = load_config()
    
    model_id = args.model_name if args.model_name else cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    system_prompt = cfg.get("system_prompt", "Ты помощник.")
    
    # 1. Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Подготовка модели (Quantization)
    bnb_config = None
    torch_dtype = torch.float16
    
    if args.quantization == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    if args.quantization == "nf4":
        model = prepare_model_for_kbit_training(model)

    # 3. LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)

    # 4. Датасет
    def format_ds(sample):
        prompt_text = f"{system_prompt}\n\n**Запрос:** {sample['prompt']}\n\n**Ответ:**"
        json_text = json.dumps(sample['filters'], ensure_ascii=False)
        full_text = prompt_text + " " + json_text
        
        tokenized = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        tokenized["text_prompt"] = prompt_text
        tokenized["labels_json"] = sample['filters']
        return tokenized

    raw_data = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    # Сплит
    random.seed(42) # Теперь это сработает
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.9)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    
    train_ds = Dataset.from_list(train_data).map(format_ds)
    eval_ds = Dataset.from_list(val_data).map(format_ds)
    
    train_ds_formatted = train_ds.remove_columns(["prompt", "filters", "text_prompt", "labels_json"])

    # 5. Аргументы обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="no",
        fp16=(args.quantization == "fp16"),
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_formatted,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    # 6. Обучение
    trainer.train()

    # 7. Валидация
    metrics = calculate_metrics_custom(model, tokenizer, eval_ds, model.device)
    
    results = {
        "lora_r": args.lora_r,
        "quantization": args.quantization,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "json_validity": metrics["json_validity"],
        "exact_match": metrics["exact_match"],
        "slot_f1": metrics["slot_f1"]
    }
    
    print(f"__RESULT_JSON__{json.dumps(results)}__RESULT_JSON__")

if __name__ == "__main__":
    main()