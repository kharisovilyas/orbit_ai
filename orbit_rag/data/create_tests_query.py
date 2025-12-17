# create_test_set.py
import json
import random
from pathlib import Path


def create_test_set(input_path: str = "prompts.jsonl",
                    output_path: str = "test_queries.jsonl",
                    test_size: int = 10):
    """Создание тестового набора из промптов"""

    # Читаем все промпты
    all_prompts = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                all_prompts.append(data)

    # Выбираем случайные для теста
    test_set = random.sample(all_prompts, min(test_size, len(all_prompts)))

    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in test_set:
            # Сохраняем как "запрос" → "ожидаемые фильтры"
            test_item = {
                "query": item["prompt"],
                "expected_filters": item["filters"]
            }
            f.write(json.dumps(test_item, ensure_ascii=False) + '\n')

    print(f"Создан тестовый набор: {len(test_set)} запросов -> {output_path}")


if __name__ == "__main__":
    create_test_set(test_size=10)