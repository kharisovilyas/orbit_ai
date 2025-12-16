import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse


def generate_embeddings(input_file: str, output_vectors: str, output_model_info: str):
    """
    Генерирует эмбеддинги для текстов из документов JSONL.

    Args:
        input_file: Путь к входному файлу documents.jsonl
        output_vectors: Путь для сохранения векторов (vectors.npy)
        output_model_info: Путь для сохранения информации о модели (embedding_model_name.txt)
    """

    # 1. Загрузка текстов
    texts = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['text'])
        print(f"Загружено {len(texts)} текстов из {input_file}")
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        return

    # 2. Загрузка модели
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"Загрузка модели {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # 3. Генерация эмбеддингов с батчингом
    print("Генерация эмбеддингов...")
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,  # Пакетная обработка для оптимизации
            normalize_embeddings=True,  # Нормализация векторов
            show_progress_bar=True  # Индикатор прогресса
        )
    except Exception as e:
        print(f"Ошибка генерации эмбеддингов: {e}")
        return

    # 4. Проверка результатов
    print(f"\nПроверка эмбеддингов:")
    print(f"  Количество векторов: {len(embeddings)}")
    print(f"  Размерность вектора: {embeddings[0].shape}")
    print(f"  Форма массива: {embeddings.shape}")

    # Проверка нормализации
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Норма векторов: min={norms.min():.6f}, max={norms.max():.6f}")
    if np.allclose(norms, 1.0, atol=1e-6):
        print("  ✅ Все векторы нормализованы (длина ~1.0)")
    else:
        print("  ⚠️ Векторы не полностью нормализованы")

    # 5. Сохранение результатов
    # Сохранение векторов
    try:
        np.save(output_vectors, embeddings)
        print(f"\nВекторы сохранены в {output_vectors}")
    except Exception as e:
        print(f"Ошибка сохранения векторов: {e}")
        return

    # Сохранение информации о модели
    try:
        with open(output_model_info, 'w', encoding='utf-8') as f:
            f.write(model_name)
        print(f"Информация о модели сохранена в {output_model_info}")
    except Exception as e:
        print(f"Ошибка сохранения информации о модели: {e}")
        return

    print("\n✅ Генерация эмбеддингов завершена успешно!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генерация эмбеддингов для текстов')
    parser.add_argument('--input', type=str, default='documents.jsonl',
                        help='Путь к входному файлу JSONL')
    parser.add_argument('--output', type=str, default='vectors.npy',
                        help='Путь для сохранения векторов')

    args = parser.parse_args()

    # Автоматическое создание имени файла для информации о модели
    model_info_file = args.output.replace('.npy', '_model_name.txt')
    if model_info_file == args.output:
        model_info_file = 'embedding_model_name.txt'

    generate_embeddings(args.input, args.output, model_info_file)