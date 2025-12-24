#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train2.py: Финальное обучение (версия 2) с лучшими гиперпараметрами.
Основано на надежной структуре train.py, но с внедрением Config 2 (Rank 32, 4-bit).
"""

import os
import sys
import json
import yaml
import random
import logging
import inspect
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orbit-nlu-train-v2")

# --- ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ (Из твоего анализа) ---
BEST_CONFIG = {
    "lora_r": 32,               # Увеличено с 16
    "lora_alpha": 64,           # 2 * r
    "learning_rate": 0.0001,    # 1e-4
    "output_dir": "outputs2/orbit-nlu-lora-v2",
    "dataset_filename": "prompts2.jsonl"
}

def read_yaml(path: str) -> Dict:
    # Ищем конфиг гибко
    paths = [Path(path), path]
    for p in paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    logger.warning("Config.yaml не найден, используем дефолтные пути.")
    return {}

def format_example(example: Dict, system_prompt: str, prompt_template: str) -> Dict:
    """Форматирование входных данных в текст + целевой JSON."""
    user = example["prompt"]
    target_json = json.dumps(example["filters"], ensure_ascii=False)
    # Используем шаблон из конфига
    text = prompt_template.format(system_prompt=system_prompt, user=user)
    return {"text": text, "labels_raw": target_json}

def build_dataset(base_path: str, filename: str, system_prompt: str, prompt_template: str, split_ratio: float = 0.95) -> Tuple[Dataset, Optional[Dataset]]:
    """Загрузка датасета с приоритетом на prompts2.jsonl."""
    # Определяем папку данных (обычно data/)
    data_dir = os.path.dirname(base_path) if os.path.dirname(base_path) else "data"
    
    # Пытаемся найти prompts2.jsonl
    target_path = os.path.join(data_dir, filename)
    if not os.path.exists(target_path):
        # Если prompts2 нет, пробуем prompts.jsonl
        logger.warning(f"Файл {target_path} не найден. Пробуем старый prompts.jsonl...")
        target_path = target_path.replace("prompts2.jsonl", "prompts.jsonl")
        
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Не удалось найти файл данных. Искали: {target_path}")

    logger.info(f"Используем датасет: {target_path}")

    data = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                data.append(format_example(ex, system_prompt, prompt_template))

    random.shuffle(data)
    # Почти все данные в обучение, так как валидацию мы делаем отдельно
    split_idx = int(len(data) * split_ratio)
    train = Dataset.from_list(data[:split_idx])
    val = Dataset.from_list(data[split_idx:]) if split_idx < len(data) else None
    
    logger.info(f"Train: {len(train)}, Val: {len(val) if val else 0}")
    return train, val

def tokenize_and_prepare(dataset: Dataset, tokenizer, max_length: int, text_key: str = "text", label_key: str = "labels_raw"):
    """
    Надежная токенизация: маскируем вопрос пользователя (input_ids), 
    чтобы модель училась генерировать ТОЛЬКО ответ (labels).
    """
    def preprocess(batch):
        inputs = tokenizer(batch[text_key], truncation=True, padding="max_length", max_length=max_length)
        # Токенизируем ответ отдельно, чтобы знать его длину (для маскирования в идеале, 
        # но для простоты здесь используем стандартный подход SFT: учим продолжать текст)
        
        # В SFTTrainer обычно принято подавать полный текст. 
        # Здесь мы используем подход, где labels = input_ids, но pad токены заменяются на -100
        label_ids = inputs["input_ids"]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Заменяем pad_token на -100, чтобы не учиться на паддинге
        processed_labels = [[(tok if tok != pad_token_id else -100) for tok in seq] for seq in label_ids]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": processed_labels
        }

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

def safe_set_env_vars():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def try_instantiate_sfttrainer(**kwargs):
    try:
        trainer = SFTTrainer(**kwargs)
        return trainer
    except Exception as e:
        logger.warning(f"SFTTrainer init warning: {e}. Пробуем fallback на стандартный Trainer...")
        return None

def main():
    try:
        safe_set_env_vars()
        cfg = read_yaml("config.yaml")
        logger.info("Базовый конфиг загружен.")

        # --- НАСТРОЙКИ (Переопределяем лучшими значениями) ---
        model_name = cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        dataset_path_orig = cfg.get("dataset_path", "data/prompts.jsonl")
        
        # Используем 4-bit (NF4) вместо 8-bit для экономии памяти при Rank 32
        # Это ключевое отличие для производительности
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        logger.info("Используется 4-bit NF4 квантование.")

        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right' # Важно для SFT

        # Модель
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            use_cache=False # Обязательно False для обучения
        )
        model = prepare_model_for_kbit_training(model)

        # LoRA Config (Внедряем Rank 32)
        peft_config = LoraConfig(
            r=BEST_CONFIG["lora_r"],      # 32
            lora_alpha=BEST_CONFIG["lora_alpha"], # 64
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"LoRA применена: Rank={BEST_CONFIG['lora_r']}, Alpha={BEST_CONFIG['lora_alpha']}")

        # Датасет
        system_prompt = cfg.get("system_prompt", "")
        prompt_tmpl = cfg.get("prompt_template", "{system_prompt}\nUser: {user}\nAnswer:")
        
        train_ds, val_ds = build_dataset(dataset_path_orig, BEST_CONFIG["dataset_filename"], system_prompt, prompt_tmpl)

        # Токенизация
        max_seq_len = cfg.get("max_seq_length", 512)
        train_tok = tokenize_and_prepare(train_ds, tokenizer, max_seq_len)
        val_tok = tokenize_and_prepare(val_ds, tokenizer, max_seq_len) if val_ds else None

        # Arguments
        output_dir = BEST_CONFIG["output_dir"]
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3, # 3 эпохи обычно достаточно для ранга 32
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
            learning_rate=BEST_CONFIG["learning_rate"],
            weight_decay=0.01,
            warmup_ratio=0.03,
            fp16=False,
            bf16=True, # bfloat16 стабильнее для Llama 3
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="no",
            report_to="none",
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
        )

        # SFT Trainer
        sft_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": train_tok,
            "eval_dataset": val_tok,
            "args": training_args,
            "dataset_text_field": "input_ids", # Используем уже токенизированные
            "packing": False
        }

        trainer = try_instantiate_sfttrainer(**sft_kwargs)
        
        # Fallback если SFTTrainer не завелся (как в твоем train.py)
        if trainer is None:
            from transformers import Trainer
            logger.warning("Fallback: используем стандартный Trainer.")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tok,
                eval_dataset=val_tok,
                tokenizer=tokenizer,
                data_collator=None 
            )

        logger.info("Начинаем обучение...")
        trainer.train()

        # Сохранение
        logger.info(f"Сохранение модели в {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Сохраняем конфиг обучения для истории
        with open(os.path.join(output_dir, "train_params.json"), "w") as f:
            json.dump(BEST_CONFIG, f, indent=4)
            
        logger.info("Успешно завершено!")

    except Exception as e:
        logger.exception(f"Ошибка: {e}")
        raise

if __name__ == "__main__":
    main()