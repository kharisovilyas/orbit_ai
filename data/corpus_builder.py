import json
import re
from typing import Dict, Any


def clean_text(text: str) -> str:
    """
    Очистка текста: удаление двойных пробелов, обрезка пробелов по краям,
    приведение к строковому типу.
    """
    if not isinstance(text, str):
        text = str(text)

    # Удаление двойных (и более) пробелов
    text = re.sub(r'\s+', ' ', text)
    # Обрезка пробелов в начале и конце
    text = text.strip()
    # Приведение к UTF-8 (в Python 3 строка уже в Unicode)
    return text


def build_corpus(input_path: str, output_path: str) -> None:
    """
    Основная функция для построения корпуса данных.

    Args:
        input_path: Путь к входному файлу prompts.jsonl
        output_path: Путь к выходному файлу documents.jsonl
    """
    documents = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Пропускаем пустые строки
                    if not line.strip():
                        continue

                    # Парсим JSON строку
                    data = json.loads(line)

                    # Извлекаем prompt и filters
                    prompt = data.get('prompt', '')
                    filters = data.get('filters', {})

                    # Очищаем текст запроса
                    cleaned_prompt = clean_text(prompt)

                    # Создаем метаданные на основе фильтров
                    metadata = {
                        "hasOrbit": bool(filters.get('orbitType', '').strip()),
                        "hasMass": bool(filters.get('mass', '').strip()),
                        "hasCoverage": bool(filters.get('coverage', '').strip()),
                        "hasStatus": bool(filters.get('status', '').strip())
                    }

                    # Формируем объект документа
                    document = {
                        "id": line_num,
                        "text": cleaned_prompt,
                        "json": filters,
                        "metadata": metadata
                    }

                    documents.append(document)

                except json.JSONDecodeError as e:
                    print(f"Ошибка парсинга JSON в строке {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Ошибка обработки строки {line_num}: {e}")
                    continue

        # Сохраняем результат в файл
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        print(f"Успешно обработано {len(documents)} документов")
        print(f"Результат сохранен в {output_path}")

    except FileNotFoundError:
        print(f"Файл {input_path} не найден")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    # Точка входа для тестирования
    input_file = "prompts.jsonl"
    output_file = "documents.jsonl"
    build_corpus(input_file, output_file)