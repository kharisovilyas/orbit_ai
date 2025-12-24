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
    # Ищем конфиг в текущей папке или в orbit_nlu
    config_paths = [Path("config.yaml"), Path("orbit_nlu/config.yaml")]
    for p in config_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml не найден!")

def build_prompt(template, system_prompt, user):
    """Формирование промпта (должно совпадать с обучением)."""
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
    Нормализация значений для честного сравнения.
    Превращает '0', 'None' в пустые строки, приводит статусы к стандарту.
    """
    if not isinstance(filters, dict): return {}
    
    sanitized = {}
    
    # Маппинг нестандартных имен ключей (если модель ошиблась в названии ключа)
    key_mapping = {
        "numberOfSatellites": "number",
        "num": "number",
        "type": "orbitType"
    }
    
    # Ключи, которые мы ожидаем в финальном JSON
    valid_keys = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "scale", "tleDate", "number"]
    
    for key, value in filters.items():
        # Исправляем имя ключа
        clean_key = key_mapping.get(key, key)
        
        if clean_key not in valid_keys:
            continue
            
        val_str = str(value).strip()
        val_lower = val_str.lower()
        
        # Пропуск пустых значений
        if val_str in ["0", "", "None", "null"]:
            sanitized[clean_key] = ""
            continue
            
        # --- ЛОГИКА НОРМАЛИЗАЦИИ ЗНАЧЕНИЙ ---
        
        # Status
        if clean_key == "status":
            if any(x in val_lower for x in ["актив", "работ", "operat", "function"]):
                sanitized[clean_key] = "активен"
            elif any(x in val_lower for x in ["не", "вышед", "inact", "decomm"]):
                sanitized[clean_key] = "неактивен"
            else:
                sanitized[clean_key] = val_str

        # OrbitType
        elif clean_key == "orbitType":
            if any(x in val_lower for x in ["geo", "гео"]): sanitized[clean_key] = "GEO"
            elif any(x in val_lower for x in ["leo", "low", "низ"]): sanitized[clean_key] = "LEO"
            elif any(x in val_lower for x in ["meo", "сред", "medium"]): sanitized[clean_key] = "MEO"
            elif any(x in val_lower for x in ["heo", "выс", "high"]): sanitized[clean_key] = "HEO"
            elif any(x in val_lower for x in ["sso", "солн"]): sanitized[clean_key] = "SSO"
            else: sanitized[clean_key] = val_str
            
        # Coverage
        elif clean_key == "coverage":
            if val_lower in ["кнр", "china"]: sanitized[clean_key] = "Китай"
            elif val_lower in ["рф", "rus", "russia"]: sanitized[clean_key] = "Россия"
            else: sanitized[clean_key] = val_str

        else:
            sanitized[clean_key] = val_str
            
    return sanitized

def classify_error(pred: dict, gt: dict) -> list:
    """Классификация типа ошибки для каждого слота."""
    if not pred and gt:
        return ["JSON Syntax Error / Empty"]
    
    errors = []
    # Сравниваем по ключам Ground Truth + Keys in Prediction
    all_keys = set(gt.keys()) | set(pred.keys())
    
    for k in all_keys:
        val_p = str(pred.get(k, "")).strip()
        val_g = str(gt.get(k, "")).strip()
        
        if val_p == val_g:
            continue
            
        if val_p and not val_g:
            errors.append(f"Hallucination ({k}: got '{val_p}')")
        elif not val_p and val_g:
            errors.append(f"Missing Slot ({k}: exp '{val_g}')")
        elif val_p != val_g:
            errors.append(f"Wrong Value ({k}: got '{val_p}' | exp '{val_g}')")
            
    return errors

# --- ОСНОВНАЯ ЛОГИКА ---
def main():
    logger.info("--- ЗАПУСК АНАЛИЗА ОШИБОК (DETAILED ERROR ANALYSIS) ---")
    
    # 1. Загрузка конфига
    cfg = load_cfg()
    base_model_name = cfg.get("model_name")
    # Пробуем найти путь к адаптеру
    possible_dirs = [Path(cfg.get("output_dir")), cfg.get("output_dir")]
    adapter_path = next((p for p in possible_dirs if p.exists()), None)

    if not adapter_path:
        logger.error(f"Адаптер не найден! Проверьте путь в config.yaml: {cfg.get('output_dir')}")
        return

    # 2. Загрузка модели (4-bit для экономии памяти)
    logger.info(f"Base Model: {base_model_name}")
    logger.info(f"Adapter: {adapter_path}")
    
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
    logger.info("Модель готова.")

    # 3. Подготовка данных
    # Пытаемся найти prompts.jsonl
    data_paths = [
        Path("data/prompts2.jsonl"),
        Path("prompts.jsonl")
    ]
    data_path = next((p for p in data_paths if p.exists()), None)
    
    full_dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): full_dataset.append(json.loads(line))
            
    # Берем последние 50 примеров для анализа
    test_set = full_dataset[-50:]
    logger.info(f"Анализ последних {len(test_set)} примеров из датасета.")

    results = []
    error_counter = Counter()
    exact_match_count = 0

    # 4. Цикл обработки
    for item in tqdm(test_set, desc="Analyzing"):
        user_text = item['prompt']
        # Нормализуем эталон, чтобы сравнивать "чистые" данные
        gt_filters = sanitize_filters(item.get('filters', {}))
        
        full_prompt = build_prompt(cfg.get("prompt_template", ""), cfg.get("system_prompt", ""), user_text)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Отключаем рандом для детерминизма
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            
        output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        raw_response = output_text.split("**Ответ:**")[-1].strip()
        
        # Парсинг и очистка предсказания
        pred_filters = sanitize_filters(fix_json(raw_response))
        
        # КЛАССИФИКАЦИЯ ОШИБОК
        error_types = classify_error(pred_filters, gt_filters)
        
        error_cat = "No Error"
        if not error_types:
            exact_match_count += 1
        else:
            # Определяем категорию ошибки
            if "JSON Syntax" in error_types[0]: 
                error_cat = "Syntax"
            elif any("Hallucination" in e for e in error_types): 
                error_cat = "Hallucination (Extra info)"
            elif any("Missing" in e for e in error_types): 
                error_cat = "Missing Slot (Lost info)"
            elif any("Wrong Value" in e for e in error_types): 
                error_cat = "Wrong Value (Mismatch)"
            
            # Считаем статистику
            for e in error_types:
                main_type = e.split("(")[0].strip() # Hallucination, Wrong Value...
                error_counter[main_type] += 1
                
                # Считаем, какое поле чаще всего сбоит
                if "(" in e:
                    field_name = e.split("(")[1].split(":")[0].strip()
                    error_counter[f"Field: {field_name}"] += 1

        results.append({
            "Prompt": user_text,
            "Ground Truth": json.dumps(gt_filters, ensure_ascii=False),
            "Prediction": json.dumps(pred_filters, ensure_ascii=False),
            "Error Category": error_cat,
            "Details": "; ".join(error_types)
        })

    # 5. Сохранение результатов
    df = pd.DataFrame(results)
    output_csv = "errors_analysis_report.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    accuracy = exact_match_count / len(test_set)
    
    print("\n" + "="*50)
    print(f"📊 ОТЧЕТ ПО ОШИБКАМ (Accuracy: {accuracy:.1%})")
    print("="*50)
    print("Топ-5 проблемных мест:")
    for k, v in error_counter.most_common(5):
        print(f"  🔴 {k}: {v} раз(а)")
    
    print("-" * 50)
    print(f"Подробная таблица сохранена в: {output_csv}")
    print("="*50)

if __name__ == "__main__":
    main()