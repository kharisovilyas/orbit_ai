#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import yaml
import torch
import logging
import time
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ИЗ SERVER.PY) ---
def load_cfg():
    """Чтение конфигурации."""
    config_path = Path("orbit_nlu/config.yaml")
    if not config_path.exists():
        config_path = Path("config.yaml") # Если запускаем из папки orbit_nlu
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_prompt(template, system_prompt, user):
    """Формирование промпта (должно совпадать с server.py)."""
    # Если в конфиге нет шаблона, используем дефолтный из server.py
    if not template:
        template = "{system_prompt}\n\n**Запрос:** {user}\n\n**Ответ:**"
    return template.format(system_prompt=system_prompt, user=user)

def fix_json(raw_text: str) -> dict:
    """Извлечение JSON из ответа."""
    try:
        # Пытаемся найти границы JSON
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
    Нормализация (как в server.py).
    Критически важно для метрик: превращает '0' в '', 'актив' в 'активен'.
    """
    if not isinstance(filters, dict): return {}
    
    sanitized = {}
    keys_to_clean = ["orbitType", "coverage", "altitude", "status", "scale", "tleDate", "formFactor"]
    
    for key, value in filters.items():
        val_str = str(value).strip()
        
        # Логика очистки
        if key in keys_to_clean and val_str == "0":
            sanitized[key] = ""
        elif key == "status" and "актив" in val_str and "не" not in val_str: # Упрощенная логика server.py
            sanitized[key] = "активен"
        else:
            sanitized[key] = val_str
            
    return sanitized

def calculate_metrics(predictions, ground_truth):
    valid_cnt = 0
    exact_match_cnt = 0
    y_true_flat = []
    y_pred_flat = []

    # Ключи, которые мы ожидаем (схема данных)
    expected_keys = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "number"]

    for pred, true_item in zip(predictions, ground_truth):
        true_filters = true_item.get("filters", {})
        
        # 1. Validity
        if pred: valid_cnt += 1
        
        # 2. Нормализация для сравнения
        # Приводим к строкам и заполняем пропуски пустыми строками, чтобы сравнивать словари целиком
        norm_pred = {k: str(pred.get(k, "")).strip() for k in expected_keys}
        norm_true = {k: str(true_filters.get(k, "")).strip() for k in expected_keys}
        
        # Exact Match
        if norm_pred == norm_true:
            exact_match_cnt += 1
        
        # Slot-F1 Data Preparation
        for k in expected_keys:
            y_true_flat.append(norm_true[k])
            y_pred_flat.append(norm_pred[k])

    total = len(ground_truth)
    return {
        "JSON Validity": valid_cnt / total if total > 0 else 0,
        "Exact Match (EM)": exact_match_cnt / total if total > 0 else 0,
        "Slot-F1 Score": f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    }

# --- ОСНОВНАЯ ЛОГИКА ---
def main():
    logger.info("--- ЗАПУСК ОЦЕНКИ ОБУЧЕННОЙ МОДЕЛИ (FT EVALUATION) ---")
    
    # 1. Загрузка конфига
    cfg = load_cfg()
    base_model_name = cfg.get("model_name")
    adapter_path = cfg.get("output_dir") # Путь к обученным весам (LoRA)
    system_prompt = cfg.get("system_prompt", "")
    prompt_tmpl = cfg.get("prompt_template", "")

    if not adapter_path or not Path(adapter_path).exists():
        logger.error(f"Адаптер не найден по пути: {adapter_path}")
        logger.error("Сначала запустите обучение (train.py)!")
        return

    # 2. Загрузка модели (ТОЧНО КАК В SERVER.PY)
    logger.info(f"Загрузка Base: {base_model_name}")
    logger.info(f"Загрузка Adapter: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Квантование 4-bit (для экономии памяти, как на сервере)
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
    
    # Подключение адаптера
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    logger.info("Модель Fine-Tuned успешно собрана.")

    # 3. Загрузка тестовых данных
    # Если есть test.jsonl - лучше использовать его. Если нет - берем конец prompts.jsonl
    data_path = Path("orbit_nlu/data/prompts.jsonl")
    if not data_path.exists(): data_path = Path("data/prompts.jsonl")
    
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): dataset.append(json.loads(line))
            
    # Используем последние 20 примеров для валидации (или весь датасет, если хочешь)
    test_set = dataset[-20:]
    logger.info(f"Оценка на выборке из {len(test_set)} примеров...")

    predictions = []

    # 4. Инференс цикл
    start_time = time.time()
    for i, item in enumerate(test_set):
        user_text = item['prompt']
        
        # Формируем промпт
        full_prompt = build_prompt(prompt_tmpl, system_prompt, user_text)
        
        # Токенизация
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        # Генерация
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Важно: для метрик лучше отключать сэмплирование (Greedy)
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Декодинг
        output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        raw_response = output_text.split("**Ответ:**")[-1].strip()
        
        # Парсинг + Санитизация (очистка)
        parsed_json = fix_json(raw_response)
        clean_json = sanitize_filters(parsed_json)
        
        predictions.append(clean_json)
        
        # Лог для контроля (первые 3 примера)
        if i < 3:
            print(f"\n[DEBUG] Query: {user_text}")
            print(f"[DEBUG] Raw: {raw_response}")
            print(f"[DEBUG] Clean: {clean_json}")
            print(f"[DEBUG] Target: {item['filters']}")

    # 5. Расчет метрик
    metrics = calculate_metrics(predictions, test_set)
    duration = time.time() - start_time
    
    print("\n" + "#"*40)
    print("РЕЗУЛЬТАТЫ FINE-TUNED МОДЕЛИ (LAB 3)")
    print("#"*40)
    print(f"Обработано примеров: {len(test_set)}")
    print(f"Время выполнения: {duration:.2f} сек")
    print("-" * 20)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("#"*40)

if __name__ == "__main__":
    main()