# benchmark.py
"""
Универсальный бенчмарк для сравнения RAG и Fine-tuned моделей
"""

import json
import time
import pandas as pd
from typing import Dict, List, Callable, Any
from pathlib import Path
from collections import Counter
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Benchmark:
    """Класс для сравнения разных подходов"""

    def __init__(self, test_data_path: str = "test_queries.jsonl"):
        self.test_data = self.load_test_data(test_data_path)
        self.results = []

    def load_test_data(self, path: str) -> List[Dict]:
        """Загрузка тестовых данных"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Загружено тестовых запросов: {len(data)}")
        return data

    def sanitize_filters(self, filters: Dict) -> Dict:
        """Нормализация фильтров (как у коллеги)"""
        if not isinstance(filters, dict):
            return {}

        sanitized = {}
        keys_to_clean = ["orbitType", "coverage", "altitude", "mass", "status",
                         "formFactor", "scale", "tleDate", "number"]

        for key in keys_to_clean:
            value = str(filters.get(key, "")).strip().lower()

            if not value or value == "0":
                sanitized[key] = ""
                continue

            # Нормализация значений
            if key == "status":
                if any(x in value for x in ["актив", "работ", "function"]):
                    sanitized[key] = "активен"
                elif any(x in value for x in ["не", "вышед", "inact"]):
                    sanitized[key] = "неактивен"
                else:
                    sanitized[key] = value

            elif key == "orbitType":
                if any(x in value for x in ["geo", "гео", "geostationary"]):
                    sanitized[key] = "GEO"
                elif any(x in value for x in ["leo", "low", "низ"]):
                    sanitized[key] = "LEO"
                elif any(x in value for x in ["sso", "солн", "sun"]):
                    sanitized[key] = "SSO"
                elif any(x in value for x in ["meo", "сред", "medium"]):
                    sanitized[key] = "MEO"
                else:
                    sanitized[key] = value

            elif key == "coverage":
                if value in ["кнр", "china", "китай"]:
                    sanitized[key] = "Китай"
                elif value in ["рф", "russia", "россии"]:
                    sanitized[key] = "Россия"
                elif value in ["africa", "африка"]:
                    sanitized[key] = "Африка"
                else:
                    sanitized[key] = value

            else:
                sanitized[key] = value

        return sanitized

    def calculate_metrics(self, predicted: Dict, expected: Dict) -> Dict:
        """Расчет метрик для одного примера"""

        pred_norm = self.sanitize_filters(predicted)
        exp_norm = self.sanitize_filters(expected)

        # Ключи для сравнения
        keys_to_check = ["orbitType", "coverage", "mass", "status", "formFactor", "number"]

        # Метрики
        exact_match = pred_norm == exp_norm

        # Поэлементное сравнение
        correct_slots = 0
        total_slots = 0
        errors = []

        for key in keys_to_check:
            pred_val = pred_norm.get(key, "")
            exp_val = exp_norm.get(key, "")

            if exp_val:  # Считаем только слоты, которые должны быть заполнены
                total_slots += 1
                if pred_val == exp_val:
                    correct_slots += 1
                else:
                    if not pred_val:
                        errors.append(f"Missing:{key}")
                    elif pred_val != exp_val:
                        errors.append(f"Wrong:{key}({pred_val}!={exp_val})")

        slot_accuracy = correct_slots / total_slots if total_slots > 0 else 1.0

        # Классификация ошибок
        error_type = "None"
        if errors:
            if any("Missing" in e for e in errors):
                error_type = "Missing"
            elif any("Wrong" in e for e in errors):
                error_type = "Wrong"

        return {
            "exact_match": exact_match,
            "slot_accuracy": slot_accuracy,
            "correct_slots": correct_slots,
            "total_slots": total_slots,
            "error_type": error_type,
            "errors": "; ".join(errors)
        }

    def test_approach(self,
                      approach_name: str,
                      inference_function: Callable[[str], Dict],
                      sample_size: int = None) -> Dict:
        """Тестирование одного подхода"""

        test_samples = self.test_data
        if sample_size:
            test_samples = test_samples[:sample_size]

        logger.info(f"Тестирование подхода: {approach_name} ({len(test_samples)} запросов)")

        total_metrics = {
            "exact_matches": 0,
            "total_slots_correct": 0,
            "total_slots_expected": 0,
            "processing_times": [],
            "error_counts": Counter()
        }

        results = []

        for i, item in enumerate(test_samples):
            query = item["query"]
            expected = item["expected_filters"]

            try:
                # Замер времени
                start_time = time.time()
                predicted = inference_function(query)
                processing_time = time.time() - start_time

                # Расчет метрик
                metrics = self.calculate_metrics(predicted, expected)

                # Агрегация
                total_metrics["exact_matches"] += 1 if metrics["exact_match"] else 0
                total_metrics["total_slots_correct"] += metrics["correct_slots"]
                total_metrics["total_slots_expected"] += metrics["total_slots"]
                total_metrics["processing_times"].append(processing_time)

                if metrics["error_type"] != "None":
                    total_metrics["error_counts"][metrics["error_type"]] += 1

                # Сохраняем детали
                results.append({
                    "approach": approach_name,
                    "query": query,
                    "processing_time": processing_time,
                    "exact_match": metrics["exact_match"],
                    "slot_accuracy": metrics["slot_accuracy"],
                    "errors": metrics["errors"],
                    "predicted": json.dumps(predicted, ensure_ascii=False),
                    "expected": json.dumps(expected, ensure_ascii=False)
                })

                if (i + 1) % 10 == 0:
                    logger.info(f"  Обработано {i + 1}/{len(test_samples)}")

            except Exception as e:
                logger.error(f"Ошибка при обработке запроса {i}: {e}")
                results.append({
                    "approach": approach_name,
                    "query": query,
                    "processing_time": None,
                    "error": str(e)
                })

        # Итоговые метрики
        avg_processing_time = sum(total_metrics["processing_times"]) / len(total_metrics["processing_times"]) if \
        total_metrics["processing_times"] else 0

        overall_metrics = {
            "approach": approach_name,
            "total_queries": len(test_samples),
            "exact_match_accuracy": total_metrics["exact_matches"] / len(test_samples),
            "slot_accuracy": total_metrics["total_slots_correct"] / total_metrics["total_slots_expected"] if
            total_metrics["total_slots_expected"] > 0 else 0,
            "avg_processing_time": avg_processing_time,
            "total_processing_time": sum(total_metrics["processing_times"]),
            "error_distribution": dict(total_metrics["error_counts"])
        }

        # Сохраняем детальные результаты
        self._save_detailed_results(approach_name, results)

        return overall_metrics

    def _save_detailed_results(self, approach_name: str, results: List[Dict]):
        """Сохранение детальных результатов"""
        filename = f"results_{approach_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Детальные результаты сохранены в {filename}")

    def compare_approaches(self, approaches: Dict[str, Callable]) -> pd.DataFrame:
        """Сравнение нескольких подходов"""
        comparison_results = []

        for name, func in approaches.items():
            metrics = self.test_approach(name, func)
            comparison_results.append(metrics)

        # Создаем сводную таблицу
        df = pd.DataFrame(comparison_results)

        # Сохраняем сравнение
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f"comparison_{timestamp}.csv", index=False)
        df.to_excel(f"comparison_{timestamp}.xlsx", index=False)

        # Выводим красивую таблицу
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ ПОДХОДОВ")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        return df


# ========================= ВАШИ ПОДХОДЫ ДЛЯ ТЕСТИРОВАНИЯ =========================

def create_rag_approach(rag_pipeline):
    """Создание функции для RAG-пайплайна"""

    def rag_inference(query: str) -> Dict:
        # Используем ваш существующий пайплайн
        result = rag_pipeline.rag_inference(query, k=3)
        # Извлекаем фильтры из результата
        filters = result.get("final_filters", {})
        return filters

    return rag_inference


def create_llama_approach():
    """Создание функции для Fine-tuned Llama (как у коллеги)"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    # Загрузка модели (настройте под свою)
    model_path = "meta-llama/Llama-3.1-8B-Instruct"  # Или путь к адаптеру
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Используем 8-битную загрузку
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Промпт как у коллеги
    system_prompt = """Ты — высокоточный NLU-парсер..."""  # Полный промпт из конфига

    def llama_inference(query: str) -> Dict:
        full_prompt = f"{system_prompt}\n\n**Запрос:** {query}\n\n**Ответ:**"

        response = pipe(
            full_prompt,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False
        )

        # Извлечение JSON
        text = response[0]['generated_text']
        start = text.find('{')
        end = text.rfind('}') + 1

        if start != -1 and end != 0:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except:
                return {}

        return {}

    return llama_inference


def create_simple_baseline():
    """Простой baseline для сравнения (например, rule-based)"""

    def baseline_inference(query: str) -> Dict:
        # Простая эвристика
        filters = {}

        if "геостационар" in query.lower():
            filters["orbitType"] = "GEO"
        elif "низкой" in query.lower() or "leo" in query:
            filters["orbitType"] = "LEO"

        if "актив" in query.lower():
            filters["status"] = "активен"
        elif "неработа" in query.lower():
            filters["status"] = "неактивен"

        # ... другие правила

        return filters

    return baseline_inference


# ========================= ЗАПУСК ТЕСТИРОВАНИЯ =========================

if __name__ == "__main__":
    # 1. Создаем тестовый набор (если еще нет)
    test_file = Path("test_queries.jsonl")
    if not test_file.exists():
        logger.info("Создание тестового набора...")
        from create_tests_query import create_test_set

        create_test_set(test_size=10)

    # 2. Инициализируем бенчмарк
    benchmark = Benchmark("test_queries.jsonl")

    # 3. Создаем подходы для сравнения
    approaches = {}

    # Ваш RAG подход (нужно инициализировать пайплайн)
    try:
        from reg_pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize(rebuild=False)
        approaches["RAG_Pipeline"] = create_rag_approach(rag_pipeline)
    except Exception as e:
        logger.warning(f"Не удалось загрузить RAG пайплайн: {e}")

    # Fine-tuned Llama подход
    try:
        approaches["FineTuned_Llama"] = create_llama_approach()
    except Exception as e:
        logger.warning(f"Не удалось загрузить Llama: {e}")

    # Baseline
    approaches["RuleBased_Baseline"] = create_simple_baseline()

    # 4. Запускаем сравнение
    if approaches:
        results_df = benchmark.compare_approaches(approaches)

        # 5. Генерируем отчет
        print("\n📊 ОТЧЕТ ПО МЕТРИКАМ:")
        print("-" * 80)

        for idx, row in results_df.iterrows():
            print(f"\n{row['approach']}:")
            print(f"  Exact Match Accuracy: {row['exact_match_accuracy']:.1%}")
            print(f"  Slot Accuracy: {row['slot_accuracy']:.1%}")
            print(f"  Avg Time/Query: {row['avg_processing_time']:.2f} сек")
            print(f"  Total Time: {row['total_processing_time']:.1f} сек")
            print(f"  Errors: {row['error_distribution']}")
    else:
        logger.error("Нет доступных подходов для тестирования")