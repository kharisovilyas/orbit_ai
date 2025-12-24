#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train2.py: Финальное обучение лучшей модели (Config 2).
Параметры зафиксированы на основе экспериментальной таблицы:
- LoRA Rank: 32
- Quantization: 4-bit (nf4)
- Learning Rate: 1e-4
- Warmup: 0.03
"""

import os
import sys
import json
import yaml
import random
import logging
import inspect
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer 

# Логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orbit-nlu-train-v2")

# --- ГИПЕРПАРАМЕТРЫ CONFIG 2 ---
BEST_PARAMS = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "learning_rate": 0.0001,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "batch_size": 2,
    "grad_accum": 4,
    "epochs": 3
}

def read_yaml(path: str) -> Dict:
    # Пытаемся найти конфиг
    paths = [Path(path), path]
    for p in paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    logger.warning("Config.yaml не найден, используем дефолтные настройки путей.")
    return {}

def format_example(example: Dict, system_prompt: str, prompt_template: str) -> Dict:
    user = example["prompt"]
    target_json = json.dumps(example["filters"], ensure_ascii=False)
    text = prompt_template.format(system_prompt=system_prompt, user=user)
    return {"text": text, "labels_raw": target_json}

def build_dataset(path: str, system_prompt: str, prompt_template: str, split_ratio: float = 0.95) -> Tuple[Dataset, Optional[Dataset]]:
    data = []
    # Проверка существования файла
    if not os.path.exists(path):
        # Фолбек на старый файл, если prompts2 нет
        alt_path = path.replace("prompts2.jsonl", "prompts.jsonl")
        if os.path.exists(alt_path):
            logger.warning(f"Файл {path} не найден. Используем {alt_path}")
            path = alt_path
        else:
            raise FileNotFoundError(f"Не найден датасет по пути: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                data.append(format_example(ex, system_prompt, prompt_template))

    random.shuffle(data)
    # Для финального обучения берем почти все данные в train, оставляем крохи для val
    split_idx = int(len(data) * split_ratio)
    train = Dataset.from_list(data[:split_idx])
    val = Dataset.from_list(data[split_idx:]) if split_idx < len(data) else None
    
    logger.info(f"Датасет '{path}': Train={len(train)}, Val={len(val) if val else 0}")
    return train, val

def main():
    try:
        # 1. Читаем конфиг только ради путей (model_name, dataset_path)
        cfg = read_yaml("config.yaml")
        
        # Переопределяем параметры "Лучшей модели"
        model_name = cfg.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Пути к файлам версии 2
        dataset_path = "data/prompts2.jsonl"
        output_dir = "outputs2/orbit-nlu-best-rank32" # Новая папка
        
        system_prompt = cfg.get("system_prompt", "You are an AI assistant.")
        prompt_tmpl = cfg.get("prompt_template", "{system_prompt}\nUser: {user}\nAnswer:")
        
        # 2. Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right' # Важно для SFTTrainer
        
        # 3. Квантование 4-bit (NF4) - Как в таблице результатов
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        logger.info("Используется 4-bit NF4 квантование (как в Best Config).")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            # use_cache=False нужно для обучения (gradient checkpointing)
            use_cache=False 
        )
        model = prepare_model_for_kbit_training(model)

        # 4. LoRA Config (Rank 32)
        peft_config = LoraConfig(
            r=BEST_PARAMS["lora_r"],
            lora_alpha=BEST_PARAMS["lora_alpha"],
            lora_dropout=BEST_PARAMS["lora_dropout"],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Обучаем все линейные слои для качества
            task_type="CAUSAL_LM",
            bias="none"
        )
        
        # 5. Датасет
        train_ds, val_ds = build_dataset(dataset_path, system_prompt, prompt_tmpl)

        # 6. Параметры обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=BEST_PARAMS["epochs"],
            per_device_train_batch_size=BEST_PARAMS["batch_size"],
            gradient_accumulation_steps=BEST_PARAMS["grad_accum"],
            learning_rate=BEST_PARAMS["learning_rate"],
            weight_decay=BEST_PARAMS["weight_decay"],
            warmup_ratio=BEST_PARAMS["warmup_ratio"],
            fp16=False,
            bf16=True, # Используем bfloat16 для стабильности Llama 3
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="no", # Экономим время, валидацию сделали в evaluate_ft.py
            report_to="none",
            optim="paged_adamw_8bit", # Экономит память
            gradient_checkpointing=True,
        )

        # 7. Запуск SFTTrainer
        logger.info(f"Начинаем обучение с параметрами: {BEST_PARAMS}")
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            dataset_text_field="text", # Поле с полным текстом
            max_seq_length=512,
            peft_config=peft_config,
            args=training_args,
            packing=False, # True может ускорить, но усложняет
        )

        trainer.train()

        # 8. Сохранение
        logger.info(f"Сохраняем финальную модель в {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Сохраняем мету о конфиге
        with open(os.path.join(output_dir, "training_params.json"), "w") as f:
            json.dump(BEST_PARAMS, f, indent=4)
            
        logger.info("ГОТОВО! Можно запускать evaluate_ft.py с новым путем.")

    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        raise

if __name__ == "__main__":
    main()