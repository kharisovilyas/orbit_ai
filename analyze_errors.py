#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import yaml
import torch
import csv
import logging
import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- КОНФИГУРАЦИЯ И УТИЛИТЫ ---
def load_cfg():
    if Path("orbit_nlu/config.yaml").exists(): return yaml.safe_load(open("orbit_nlu/config.yaml"))
    with open("config.yaml", "r", encoding="utf-8") as f: return yaml.safe_load(f)

def build_prompt(template, system_prompt, user):
    if not template: template = "{system_prompt}\n\n**Запрос:** {user}\n\n**Ответ:**"
    return template.format(system_prompt=system_prompt, user=user)

def fix_json(raw_text: str) -> dict:
    try:
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(raw_text[start:end])
        return {}
    except:
        return {}

def sanitize_filters(filters: dict) -> dict:
    # Та же логика, что в server.py, чтобы оценка была честной
    if not isinstance(filters, dict): return {}
    sanitized = {}
    keys_to_clean = ["orbitType", "coverage", "altitude", "status", "scale", "tleDate", "formFactor"]
    for key, value in filters.items():
        val_str = str(value).strip()
        if key in keys_to_clean and val_str == "0":
            sanitized[key] = ""
        elif key == "status" and "актив" in val_str and "не" not in val_str:
            sanitized[key] = "активен"
        else:
            sanitized[key] = val_str
    return sanitized

def classify_error(pred: dict, gt: dict) -> list:
    """Определяет тип ошибки для конкретной пары предсказание-эталон."""
    if not pred and gt:
        return ["JSON Syntax Error / Empty"]
    
    errors = []
    all_keys = set(pred.keys()) | set(gt.keys())
    
    # Игнорируем ключи, которых нет в схеме, если они пустые
    expected_keys = {"orbitType", "coverage", "altitude", "mass", "status", "formFactor", "scale", "tleDate", "numberOfSatellites"}
    
    for k in expected_keys:
        val_p = str(pred.get(k, "")).strip()
        val_g = str(gt.get(k, "")).strip()
        
        if val_p == val_g:
            continue
            
        # Логика классификации
        if val_p and not val_g:
            errors.append(f"Hallucination ({k}: '{val_p}')")
        elif not val_p and val_g:
            errors.append(f"Missing Slot ({k}: exp '{val_g}')")
        elif val_p != val_g:
            # Проверка на неконсистентность (например, числовой формат)
            errors.append(f"Wrong Value ({k}: got '{val_p}' exp '{val_g}')")
            
    return errors

# --- MAIN ---
def main():
    cfg = load_cfg()
    base_model_name = cfg.get("model_name")
    adapter_path = cfg.get("output_dir")
    system_prompt = cfg.get("system_prompt", "")
    prompt_tmpl = cfg.get("prompt_template", "")

    # 1. Загрузка модели
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", quantization_config=quant_config)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # 2. Подготовка данных (берем последние 10% как тест)
    data_path = Path("prompts.jsonl")
    if not data_path.exists(): data_path = Path("data/prompts.jsonl")
    
    full_dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): full_dataset.append(json.loads(line))
    
    # Отложенная выборка (10%)
    test_size = int(len(full_dataset) * 0.1)
    if test_size == 0: test_size = len(full_dataset) # Если данных мало
    test_set = full_dataset[-test_size:]
    
    logger.info(f"Анализ ошибок на {len(test_set)} примерах...")
    
    results = []
    error_counter = Counter()

    # 3. Инференс
    for item in test_set:
        user_text = item['prompt']
        gt_filters = sanitize_filters(item.get('filters', {})) # Нормализуем GT тоже
        
        full_prompt = build_prompt(prompt_tmpl, system_prompt, user_text)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        raw_resp = output_text.split("**Ответ:**")[-1].strip()
        pred_filters = sanitize_filters(fix_json(raw_resp))
        
        # 4. Классификация ошибок
        error_types = classify_error(pred_filters, gt_filters)
        
        # Агрегация статистики
        if not error_types:
            error_cat = "No Error"
        else:
            # Для упрощения таблицы берем первую или объединяем
            if "JSON Syntax" in error_types[0]: error_cat = "Syntax"
            elif any("Hallucination" in e for e in error_types): error_cat = "Hallucination"
            elif any("Missing" in e for e in error_types): error_cat = "Missing Slot"
            elif any("Wrong Value" in e for e in error_types): error_cat = "Wrong Value"
            else: error_cat = "Other"
            
            # Считаем конкретные поля ошибок
            for e in error_types:
                error_counter[e.split("(")[0].strip()] += 1
                # Считаем поля
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

    # 5. Сохранение таблицы
    df = pd.DataFrame(results)
    df.to_csv("errors_detailed.csv", index=False, encoding='utf-8-sig')
    logger.info("Детальная таблица сохранена в errors_detailed.csv")
    
    # 6. Вывод статистики
    print("\n" + "="*40)
    print("ТОП 10 TИПОВ ОШИБОК И ПРОБЛЕМНЫХ ПОЛЕЙ")
    print("="*40)
    for k, v in error_counter.most_common(10):
        print(f"{k:<30}: {v}")
    
    print("-" * 40)
    accuracy = len(df[df["error_category"] == "No Error"]) / len(df)
    print(f"Accuracy (Exact Match): {accuracy:.2%}")

if __name__ == "__main__":
    main()