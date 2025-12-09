#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Дообучение LLM с LoRA для маппинга русских промтов на JSON-фильтры (SFT).
Устойчивый и совместимый вариант, с fallback'ами для старых/новых версий transformers/trl/peft.
Оптимизирован под 16GB GPU (параметры берутся из config.yaml).
"""

import os
import sys
import json
import yaml
import random
import logging
import inspect
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
from trl import SFTTrainer  # используем, если доступен

# ----------------- РЕКОМЕНДУЕМЫЕ ВЕРСИИ -----------------
# transformers >= 4.44.0
# peft >= 0.12.0
# trl >= 0.9.6 (или новее)
# datasets >= 2.20.0
# bitsandbytes >= 0.39.0
# ------------------------------------------------------

# Логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orbit-nlu-train")


def read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_example(example: Dict, system_prompt: str, prompt_template: str) -> Dict:
    user = example["prompt"]
    target_json = json.dumps(example["filters"], ensure_ascii=False)
    text = prompt_template.format(system_prompt=system_prompt, user=user)
    # Для SFT: мы будем строить вход как text и требовать продолжение JSON.
    # В датасете хранить отдельно целевой ответ в поле "labels_raw"
    return {"text": text, "labels_raw": target_json}


def build_dataset(path: str, system_prompt: str, prompt_template: str, split_ratio: float = 0.9) -> Tuple[Dataset, Optional[Dataset]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            data.append(format_example(ex, system_prompt, prompt_template))

    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train = Dataset.from_list(data[:split_idx])
    val = Dataset.from_list(data[split_idx:]) if split_idx < len(data) else None
    logger.info(f"Загружено {len(train)} train примеров и {(len(val) if val else 0)} val примеров.")
    return train, val


def tokenize_and_prepare(dataset: Dataset, tokenizer, max_length: int, text_key: str = "text", label_key: str = "labels_raw"):
    """
    Токенизация батчами. Формируем input_ids, attention_mask и labels (для causal LM — сдвигаем не нужно;
    просто используем input_ids для labels, но ставим -100 для паддинга).
    """
    def preprocess(batch):
        # batch[text_key] и batch[label_key] — списки строк
        inputs = tokenizer(batch[text_key], truncation=True, padding="max_length", max_length=max_length)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch[label_key], truncation=True, padding="max_length", max_length=max_length)
        # labels["input_ids"] — заменить pad_token_id на -100
        label_ids = labels["input_ids"]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        label_ids = [[(tok if tok != pad_token_id else -100) for tok in seq] for seq in label_ids]

        result = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": label_ids
        }
        return result

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def safe_set_env_vars():
    # Рекомендации по управлению аллокатором PyTorch
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # уменьшает фрагментацию
        logger.info("Установлен PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")


def try_instantiate_sfttrainer(**kwargs):
    # Попытка создать SFTTrainer; если не получится — вернём None
    try:
        trainer = SFTTrainer(**kwargs)
        return trainer
    except Exception as e:
        logger.warning(f"SFTTrainer init failed: {e}. Попробуем fallback.")
        return None


def custom_generate_eval(model, tokenizer, eval_dataset: Dataset, device: str, batch_size: int = 1, max_new_tokens: int = 256):
    """
    Если evaluate с predict_with_generate недоступен, генерируем ответы вручную по eval_dataset.
    Возвращаем список (preds, refs)
    """
    model.eval()
    preds = []
    refs = []
    loader = eval_dataset.to_dict()  # Dataset -> dict of lists
    total = len(eval_dataset)
    for i in range(0, total, batch_size):
        batch_texts = loader["text"][i:i+batch_size]
        batch_labels = loader["labels_raw"][i:i+batch_size]
        # build prompts
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend(gen_texts)
        refs.extend(batch_labels)
    return preds, refs


def main():
    try:
        safe_set_env_vars()
        cfg = read_yaml("config.yaml")
        logger.info("Конфигурация загружена.")

        # Основные настройки
        model_name = cfg["model_name"]
        load_8bit = bool(cfg.get("load_8bit", False))
        dataset_path = cfg["dataset_path"]
        val_dataset_path = cfg.get("val_dataset_path")
        split_ratio = cfg.get("split_ratio", 0.9)
        model_max_length = cfg.get("model_max_length", cfg.get("max_seq_length", 512))
        output_dir = cfg.get("output_dir", "outputs")
        per_device_train_batch_size = int(cfg.get("per_device_train_batch_size", 1))
        per_device_eval_batch_size = int(cfg.get("per_device_eval_batch_size", 1))
        gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))

        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = model_max_length
        logger.info(f"Токенизатор загружен: {model_name}; model_max_length={model_max_length}")

        # Bits&Bytes конфиг (если 8-bit)
        quant_config = None
        if load_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Используется 8-bit квантование (bitsandbytes).")

        # Загрузка базовой модели
        # Выбираем dtype аккуратно — если load_8bit=True, то модель сама будет в k-bit, dtype не обязателен.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config
        )
        logger.info(f"Модель {model_name} загружена.")

        # Подготовка к LoRA-обучению
        if load_8bit:
            model = prepare_model_for_kbit_training(model)

        # LoRA конфигурация
        lora_cfg = cfg.get("lora", {})
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        # Включаем checkpointing для экономии памяти
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            logger.debug("gradient_checkpointing_enable не доступен для этой модели/версии.")

        logger.info("LoRA адаптация применена.")

        # Датасет
        if val_dataset_path and os.path.exists(val_dataset_path):
            train_ds, val_ds = build_dataset(dataset_path, cfg["system_prompt"], cfg["prompt_template"], split_ratio=1.0)
            # когда val задан явно — build_dataset не нужен, читаем отдельно
            val_ds, _ = build_dataset(val_dataset_path, cfg["system_prompt"], cfg["prompt_template"], split_ratio=0.0)
        else:
            train_ds, val_ds = build_dataset(dataset_path, cfg["system_prompt"], cfg["prompt_template"], split_ratio=split_ratio)

        # Токенизация и подготовка (labels)
        logger.info("Токенизация и подготовка train/val...")
        train_tok = tokenize_and_prepare(train_ds, tokenizer, model_max_length)
        val_tok = tokenize_and_prepare(val_ds, tokenizer, model_max_length) if val_ds else None

        # TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(cfg.get("num_train_epochs", 3)),
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=float(cfg.get("learning_rate", 2e-5)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 500)),
            eval_strategy="steps" if val_tok is not None else "no",
            eval_steps=int(cfg.get("eval_steps", 500)) if val_tok is not None else None,
            fp16=not load_8bit,
            bf16=False,
            report_to="none",
            gradient_checkpointing=True,
            dataloader_num_workers=int(cfg.get("dataset_num_proc", 1)),
            remove_unused_columns=False,
        )

        logger.info("TrainingArguments подготовлены.")

        # Попытка создать SFTTrainer (если trl установлена и SFTTrainer доступен)
        sft_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": train_tok,
            "eval_dataset": val_tok,
            "args": training_args,
            # В зависимости от версии trl, SFTTrainer может ожидать разные аргументы.
            # Укажем dataset_text_field, чтобы SFTTrainer знал, где брать входную строку (старые реализации).
            "dataset_text_field": "input_ids"  # в случае проблем SFTTrainer сам обработает токены
        }

        trainer = try_instantiate_sfttrainer(**sft_kwargs)
        if trainer is None:
            # Фоллбек: попытаемся использовать transformers.Trainer как запасной вариант
            from transformers import Trainer
            logger.warning("Используем transformers.Trainer в качестве fallback'а (без SFTTrainer).")
            data_collator = None
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tok,
                eval_dataset=val_tok,
                tokenizer=tokenizer,
                data_collator=data_collator
            )

        # Попытка evaluate с генерацией: гибко — если метод не принимает predict_with_generate, делаем fallback.
        try:
            # Проверяем, можно ли передать predict_with_generate
            sig = inspect.signature(trainer.evaluate)
            if "predict_with_generate" in sig.parameters:
                logger.info("Вызов trainer.evaluate(predict_with_generate=True).")
                eval_metrics = trainer.evaluate(predict_with_generate=True)
            else:
                logger.info("trainer.evaluate не поддерживает predict_with_generate; вызываем без параметра.")
                eval_metrics = trainer.evaluate()
            logger.info(f"Eval metrics: {eval_metrics}")
        except TypeError as e:
            logger.warning(f"Не удалось вызвать trainer.evaluate с predict_with_generate: {e}. Запускаем custom_eval.")
            if val_ds is not None:
                device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
                preds, refs = custom_generate_eval(model, tokenizer, val_ds, device=device, batch_size=per_device_eval_batch_size, max_new_tokens=256)
                # простая метрика exact match доли:
                exact = sum(1 if p.strip() == r.strip() else 0 for p, r in zip(preds, refs)) / max(1, len(preds))
                logger.info(f"Custom generate exact match ratio: {exact:.4f}")
            else:
                logger.info("Нет eval-датасета — пропускаем валидацию.")

        # Обучение
        logger.info("Начинаем обучение...")
        trainer.train()
        logger.info("Обучение завершено.")

        # Сохранение
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Сохраняем модель и токенизатор в {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Сохранение завершено.")

    except Exception as e:
        logger.exception(f"Ошибка при обучении: {e}")
        raise


if __name__ == "__main__":
    main()
