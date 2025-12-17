#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для улучшения датасета (Data Augmentation / Cleaning) с помощью RAG и Llama-70b.
Проходит по prompts.jsonl, генерирует эталонные JSON-фильтры через LLM и сохраняет улучшенный датасет.
"""

import os
import json
import numpy as np
import faiss
import openai
import time
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Получаем ключ из окружения
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY не найден! Установите его: export GROQ_API_KEY='gsk_...'")

# ========================= КОНФИГУРАЦИЯ =========================
@dataclass
class Config:
    # Пути
    input_dataset: str = "prompts.jsonl"      # Исходный грязный датасет
    output_dataset: str = "prompts_augmented.jsonl" # Чистый датасет
    documents_path: str = "rag_data/documents.jsonl"
    vectors_path: str = "rag_data/vectors.npy"
    index_path: str = "rag_data/index.faiss"

    # Модели
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = GROQ_API_KEY

    # Параметры
    embedding_dim: int = 768
    batch_size: int = 32

config = Config()
os.makedirs("rag_data", exist_ok=True)

# ========================= МОДУЛИ RAG (Упрощенные для скрипта) =====================

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=config.batch_size, normalize_embeddings=True, show_progress_bar=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True).astype(np.float32)

class FAISSIndex:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None
        self.documents = []

    def build_or_load(self, documents: List[Dict], embedder: Embedder):
        """Создает индекс, если его нет, или загружает существующий"""
        if os.path.exists(self.index_path) and os.path.exists(config.vectors_path):
            logger.info("Загрузка существующего индекса FAISS...")
            self.index = faiss.read_index(self.index_path)
            self.documents = documents
            return

        logger.info("Генерация векторов и индекса...")
        texts = [doc['text'] for doc in documents]
        vectors = embedder.generate_embeddings(texts).astype(np.float32)
        
        self.index = faiss.IndexFlatIP(config.embedding_dim)
        self.index.add(vectors)
        faiss.write_index(self.index, self.index_path)
        np.save(config.vectors_path, vectors)
        self.documents = documents
        logger.info(f"Индекс создан: {self.index.ntotal} элементов")

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict]:
        distances, indices = self.index.search(query_vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({"document": self.documents[idx], "similarity": float(dist)})
        return results

class LLMProcessor:
    def __init__(self):
        self.client = openai.OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)

    def refine_filters(self, query: str, context_docs: List[Dict]) -> Dict:
        # Формируем промпт (без изменений)
        examples = ""
        for i, res in enumerate(context_docs, 1):
            doc = res["document"]
            examples += f"Пример {i}:\nЗапрос: {doc['text']}\nОтвет JSON: {json.dumps(doc['json'], ensure_ascii=False)}\n\n"

        prompt = f"""Ты эксперт по извлечению параметров спутников.
Задача: Исправить JSON-фильтры.
1. `status`: "неактивный"/"сломан" -> "неактивен". Иначе -> "активен".
2. `coverage`: КНР->Китай, РФ->Россия.
3. `orbitType`: GEO, LEO, MEO, SSO, HEO, Molniya.
4. `number`: число спутников.

Контекст:
{examples}

ЗАПРОС: "{query}"

Верни ТОЛЬКО JSON."""

        max_retries = 5
        wait_time = 5  # Начальное ожидание
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=300
                )
                return json.loads(response.choices[0].message.content)
            
            except openai.RateLimitError as e:
                error_msg = str(e)
                # Ищем время ожидания в тексте ошибки "Please try again in 5m47.328s"
                import re
                match = re.search(r"try again in (\d+)m(\d+\.?\d*)s", error_msg)
                if match:
                    minutes = float(match.group(1))
                    seconds = float(match.group(2))
                    sleep_seconds = minutes * 60 + seconds + 2 # +2 сек про запас
                    logger.warning(f"Лимит токенов! Ждем {sleep_seconds:.1f} сек...")
                    time.sleep(sleep_seconds)
                else:
                    # Если не смогли распарсить время, просто ждем с экспонентой
                    logger.warning(f"Rate Limit (429). Ждем {wait_time} сек...")
                    time.sleep(wait_time)
                    wait_time *= 2 # Увеличиваем время ожидания
                    
            except Exception as e:
                logger.error(f"Ошибка LLM: {e}")
                return {}
        
        return {}
# ========================= MAIN LOGIC =========================

def load_raw_data():
    """Загружает исходный датасет"""
    if not os.path.exists(config.input_dataset):
        # Попробуем найти в data/
        alt_path = "data/" + config.input_dataset
        if os.path.exists(alt_path):
            config.input_dataset = alt_path
        else:
            raise FileNotFoundError(f"Нет файла {config.input_dataset}")

    docs = []
    with open(config.input_dataset, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                js = json.loads(line)
                docs.append({"text": js['prompt'], "json": js.get('filters', {})})
    return docs

def main():
    if not config.llm_api_key:
        print("❌ ОШИБКА: Нет API ключа. Введите: export GROQ_API_KEY='ваш_ключ'")
        return

    logger.info("1. Загрузка исходного датасета...")
    raw_docs = load_raw_data()
    logger.info(f"Загружено {len(raw_docs)} примеров.")

    # Используем 20% датасета как базу знаний (корпус для RAG), 
    # а прогонять будем остальные, или прогоним ВСЕ для полной чистки.
    # Для улучшения качества лучше прогнать ВСЕ через LLM.
    
    logger.info("2. Инициализация RAG (Embedder + FAISS)...")
    embedder = Embedder(config.embedding_model)
    index = FAISSIndex(config.index_path)
    
    # Строим индекс по всему текущему датасету (чтобы искать похожие формулировки)
    index.build_or_load(raw_docs, embedder)
    
    llm = LLMProcessor()
    augmented_data = []
    
    logger.info("3. Запуск генерации улучшенных меток (Data Cleaning)...")
    
    # Обработаем, например, первые 200 (или весь, если есть время) для теста
    # Чтобы обработать все - уберите [:200]
    process_limit = 200 
    logger.info(f"Обрабатываем первые {process_limit} записей (для теста)...")
    
    for doc in tqdm(raw_docs[:process_limit]):
        query = doc['text']
        original_json = doc['json']
        
        # 1. Ищем похожие примеры (чтобы LLM видела паттерны)
        q_vec = embedder.embed_query(query)
        context = index.search(q_vec, k=3)
        
        # 2. Генерируем идеальный JSON
        new_json = llm.refine_filters(query, context)
        
        # 3. Валидация (если LLM вернула пустой JSON, оставляем старый)
        if not new_json:
            final_json = original_json
        else:
            final_json = new_json
            
        augmented_data.append({
            "prompt": query,
            "filters": final_json,
            "meta": "augmented_by_llama70b"
        })
        
        # Небольшая пауза для rate limits Groq (бесплатный тариф)
        time.sleep(0.5)

    # Сохранение
    with open(config.output_dataset, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"✅ Готово! Улучшенный датасет сохранен в {config.output_dataset}")
    logger.info("Теперь запустите обучение (train.py) на ЭТОМ файле.")

if __name__ == "__main__":
    main()