#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import yaml
import torch
import logging
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def load_cfg():
    """Чтение конфигурации."""
    config_path = Path("orbit_nlu/config.yaml")
    if not config_path.exists():
        config_path = Path("config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_prompt(template, system_prompt, user):
    """Формирование промпта."""
    if not template:
        template = "{system_prompt}\n\n**Запрос:** {user}\n\n**Ответ:**"
    return template.format(system_prompt=system_prompt, user=user)

def fix_json(raw_text: str) -> dict:
    """Извлечение JSON из ответа (поиск скобок)."""
    try:
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = raw_text[start:end]
            return json.loads(json_str)
        return {}
    except:
        return {}

def sanitize_filters(filters: dict) -> dict:
    """
    Умная нормализация: приводит синонимы к единому стандарту, 
    чтобы метрики считались честно, а не падали из-за опечаток.
    """
    if not isinstance(filters, dict): return {}
    
    sanitized = {}
    
    # 1. Исправляем расхождение в именах ключей
    # Если модель предсказала numberOfSatellites, а в GT number - мапим.
    if "numberOfSatellites" in filters:
        filters["number"] = filters["numberOfSatellites"]

    keys_to_clean = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "scale", "tleDate", "number"]
    
    for key, value in filters.items():
        if key not in keys_to_clean:
            continue
            
        val_str = str(value).strip()
        val_lower = val_str.lower()
        
        # Пропуск пустых значений
        if val_str == "0" or val_str == "":
            sanitized[key] = ""
            continue
            
        # --- ЛОГИКА НОРМАЛИЗАЦИИ ЗНАЧЕНИЙ ---
        
        # Status: "актив", "ф" -> "активен"; "не", "вышед" -> "неактивен"
        if key == "status":
            if any(x in val_lower for x in ["актив", "работ", "ф", "function"]):
                sanitized[key] = "активен"
            elif any(x in val_lower for x in ["не", "вышед", "inact"]):
                sanitized[key] = "неактивен"
            else:
                sanitized[key] = val_str

        # OrbitType: исправляем обрывки слов и синонимы
        elif key == "orbitType":
            if any(x in val_lower for x in ["geo", "гео", "ге", "geostationary"]): sanitized[key] = "GEO"
            elif any(x in val_lower for x in ["leo", "low", "низ"]): sanitized[key] = "LEO"
            elif any(x in val_lower for x in ["sso", "солн", "с", "sun"]): sanitized[key] = "SSO"
            elif any(x in val_lower for x in ["meo", "сред", "medium"]): sanitized[key] = "MEO"
            elif "мол" in val_lower: sanitized[key] = "Molniya"
            elif "heo" in val_lower or "выс" in val_lower: sanitized[key] = "HEO"
            else: sanitized[key] = val_str
            
        # Coverage: исправляем сокращения стран
        elif key == "coverage":
            if val_lower in ["кнр", "к", "china"]: sanitized[key] = "Китай"
            elif val_lower in ["рф", "россии", "russia", "rus"]: sanitized[key] = "Россия"
            elif val_lower in ["а", "africa"]: sanitized[key] = "Африка"
            else: sanitized[key] = val_str

        else:
            sanitized[key] = val_str
            
    return sanitized

def classify_error(pred: dict, gt: dict) -> list:
    """Классификация типа ошибки."""
    if not pred and gt:
        return ["JSON Syntax Error / Empty"]
    
    errors = []
    # Сравниваем только те ключи, что есть в схеме
    expected_keys = {"orbitType", "coverage", "altitude", "mass", "status", "formFactor", "scale", "tleDate", "number"}
    
    for k in expected_keys:
        val_p = str(pred.get(k, "")).strip()
        val_g = str(gt.get(k, "")).strip()
        
        if val_p == val_g:
            continue
            
        if val_p and not val_g:
            errors.append(f"Hallucination ({k}: '{val_p}')")
        elif not val_p and val_g:
            errors.append(f"Missing Slot ({k}: exp '{val_g}')")
        elif val_p != val_g:
            errors.append(f"Wrong Value ({k}: got '{val_p}' exp '{val_g}')")
            
    return errors

# --- ОСНОВНАЯ ЛОГИКА ---
def main():
    logger.info("--- ЗАПУСК АНАЛИЗА ОШИБОК (ANALYZE ERRORS) ---")
    
    # 1. Загрузка конфига
    cfg = load_cfg()
    base_model_name = cfg.get("model_name")
    adapter_path = cfg.get("output_dir")
    system_prompt = cfg.get("system_prompt", "")
    prompt_tmpl = cfg.get("prompt_template", "")

    if not adapter_path or not Path(adapter_path).exists():
        logger.error(f"Адаптер не найден: {adapter_path}. Сначала запустите train.py")
        return

    # 2. Загрузка модели
    logger.info(f"Загрузка Base: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    logger.info("Модель готова к тесту.")

    # 3. Подготовка данных (последние 10%)
    data_path = Path("orbit_nlu/data/prompts.jsonl")
    if not data_path.exists(): data_path = Path("prompts.jsonl")
    if not data_path.exists(): data_path = Path("data/prompts.jsonl")
    
    full_dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): full_dataset.append(json.loads(line))
            
    test_size = int(len(full_dataset) * 0.005)
    if test_size < 10: test_size = len(full_dataset) # Если данных мало, берем все
    test_set = full_dataset[-test_size:]
    
    logger.info(f"Тестирование на выборке: {len(test_set)} примеров")

    results = []
    error_counter = Counter()
    exact_match_count = 0

    # 4. Цикл инференса с прогресс-баром
    for item in tqdm(test_set, desc="Processing"):
        user_text = item['prompt']
        # Важно: нормализуем Ground Truth тоже, чтобы формат совпадал
        gt_filters = sanitize_filters(item.get('filters', {}))
        
        full_prompt = build_prompt(prompt_tmpl, system_prompt, user_text)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Отключаем рандом для анализа
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            
        output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        raw_response = output_text.split("**Ответ:**")[-1].strip()
        
        # Парсинг и очистка предсказания
        pred_filters = sanitize_filters(fix_json(raw_response))
        
        # Классификация ошибок
        error_types = classify_error(pred_filters, gt_filters)
        
        error_cat = "No Error"
        if not error_types:
            exact_match_count += 1
        else:
            if "JSON Syntax" in error_types[0]: error_cat = "Syntax"
            elif any("Hallucination" in e for e in error_types): error_cat = "Hallucination"
            elif any("Missing" in e for e in error_types): error_cat = "Missing Slot"
            elif any("Wrong Value" in e for e in error_types): error_cat = "Wrong Value"
            
            for e in error_types:
                main_type = e.split("(")[0].strip()
                error_counter[main_type] += 1
                if "(" in e:
                    field = e.split("(")[1].split(":")[0]
                    error_counter[f"Field: {field}"] += 1

        results.append({
            "prompt": user_text,
            "gt_json": json.dumps(gt_filters, ensure_ascii=False),
            "predicted_json": json.dumps(pred_filters, ensure_ascii=False),
            "error_category": error_cat,
            "details": "; ".join(error_types)
        })

    # 5. Сохранение и отчет
    df = pd.DataFrame(results)
    df.to_csv("errors_detailed.csv", index=False, encoding='utf-8-sig')
    
    accuracy = exact_match_count / len(test_set)
    
    print("\n" + "="*40)
    print("TOP 10 TИПОВ ОШИБОК И ПРОБЛЕМНЫХ ПОЛЕЙ")
    print("="*40)
    for k, v in error_counter.most_common(10):
        print(f"{k:<30}: {v}")
    
    print("-" * 40)
    print(f"Accuracy (Exact Match): {accuracy:.2%}")
    print(f"Детальный отчет сохранен в: errors_detailed.csv")
    print("="*40)

if __name__ == "__main__":
    main()