import json
import logging
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from datetime import datetime
from reg_pipeline import RAGPipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- ФУНКЦИИ ОЧИСТКИ И САНАЦИИ (ОСТАВЛЕНЫ БЕЗ ИЗМЕНЕНИЙ ДЛЯ ЧЕСТНОГО ТЕСТА) ---

def sanitize_filters(filters: dict) -> dict:
    if not isinstance(filters, dict): return {}
    sanitized = {}

    # Синхронизация имен ключей (RAG может возвращать numberOfSatellites или number)
    if "numberOfSatellites" in filters:
        filters["number"] = filters["numberOfSatellites"]

    keys_to_clean = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "scale", "tleDate", "number"]

    for key in keys_to_clean:
        value = filters.get(key, "")
        val_str = str(value).strip()
        val_lower = val_str.lower()

        if val_str in ["0", "", "None", "null"]:
            sanitized[key] = ""
            continue

        if key == "status":
            if any(x in val_lower for x in ["актив", "работ", "ф", "function"]):
                sanitized[key] = "активен"
            elif any(x in val_lower for x in ["не", "вышед", "inact"]):
                sanitized[key] = "неактивен"
            else:
                sanitized[key] = val_str
        elif key == "orbitType":
            if any(x in val_lower for x in ["geo", "гео", "geostationary"]):
                sanitized[key] = "GEO"
            elif any(x in val_lower for x in ["leo", "низ"]):
                sanitized[key] = "LEO"
            elif any(x in val_lower for x in ["sso", "солн"]):
                sanitized[key] = "SSO"
            elif any(x in val_lower for x in ["meo", "сред"]):
                sanitized[key] = "MEO"
            else:
                sanitized[key] = val_str
        elif key == "coverage":
            if val_lower in ["кнр", "china"]:
                sanitized[key] = "Китай"
            elif val_lower in ["рф", "россия", "rus"]:
                sanitized[key] = "Россия"
            else:
                sanitized[key] = val_str
        else:
            sanitized[key] = val_str
    return sanitized


def classify_error(pred: dict, gt: dict) -> list:
    errors = []
    expected_keys = {"orbitType", "coverage", "altitude", "mass", "status", "formFactor", "scale", "tleDate", "number"}

    for k in expected_keys:
        val_p = str(pred.get(k, "")).strip()
        val_g = str(gt.get(k, "")).strip()

        if val_p == val_g: continue
        if val_p and not val_g:
            errors.append(f"Hallucination ({k}: '{val_p}')")
        elif not val_p and val_g:
            errors.append(f"Missing Slot ({k}: exp '{val_g}')")
        elif val_p != val_g:
            errors.append(f"Wrong Value ({k}: got '{val_p}' exp '{val_g}')")

    return errors


# --- ОСНОВНАЯ ЛОГИКА ТЕСТИРОВАНИЯ ---
def main():
    logger.info("--- ЗАПУСК ТЕСТИРОВАНИЯ RAG PIPELINE ---")

    # 1. Инициализация пайплайна
    # Использует Config по умолчанию, который берет API ключ из env
    pipeline = RAGPipeline()
    logger.info("Инициализация индексов и моделей...")
    if not pipeline.initialize():
        logger.error("Не удалось инициализировать RAG Pipeline")
        return

    # 2. Подготовка тестовых данных
    # Берем те же последние 0.5% данных, что и в тесте FT-модели
    data_path = Path("prompts.jsonl")
    if not data_path.exists(): data_path = Path("data/prompts.jsonl")

    full_dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): full_dataset.append(json.loads(line))

    test_size = int(len(full_dataset) * 0.005)
    if test_size < 10: test_size = len(full_dataset)
    test_set = full_dataset[-test_size:]

    logger.info(f"Выборка для теста: {len(test_set)} примеров")

    results = []
    error_counter = Counter()
    exact_match_count = 0

    # 3. Цикл тестирования
    for item in tqdm(test_set, desc="RAG Inference"):
        user_text = item['prompt']
        gt_filters = sanitize_filters(item.get('filters', {}))

        try:
            # Вызываем ваш основной метод пайплайна
            rag_output = pipeline.rag_inference(user_text, k=5)

            # Извлекаем финальные фильтры (они уже в dict формате)
            pred_raw = rag_output.get("final_filters", {})
            pred_filters = sanitize_filters(pred_raw)

            # проверка на пустой ответ
            if not pred_filters and gt_filters:
                error_types = ["JSON Syntax Error / Empty"]
            else:
                error_types = classify_error(pred_filters, gt_filters)

        except Exception as e:
            logger.error(f"Ошибка при обработке '{user_text[:30]}...': {e}")
            pred_filters = {}
            error_types = ["Pipeline Crash"]

        # Статистика
        error_cat = "No Error"
        if not error_types:
            exact_match_count += 1
        else:
            error_cat = error_types[0].split(" ")[0]  # Берем первое слово категории
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
            "details": "; ".join(error_types),
            "rewritten_query": rag_output.get("clarified_query", "")
        })

    # 4. Сохранение отчета
    df = pd.DataFrame(results)
    output_file = "errors_rag_detailed.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    accuracy = exact_match_count / len(test_set)

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ RAG")
    print("=" * 40)
    for k, v in error_counter.most_common(10):
        print(f"{k:<30}: {v}")

    print("-" * 40)
    print(f"Accuracy (Exact Match): {accuracy:.2%}")
    print(f"Отчет сохранен в: {output_file}")
    print("=" * 40)


if __name__ == "__main__":
    main()