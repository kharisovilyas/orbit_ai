"""
RAG Pipeline для поиска спутников
Объединяет: подготовку корпуса, векторизацию, FAISS поиск, перефразирование и генерацию ответов
"""
import os
import json
import numpy as np
import faiss
import openai
import time
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========================= КОНФИГУРАЦИЯ =========================
@dataclass
class Config:
    """Конфигурация всей RAG-системы"""
    # Пути к данным
    prompts_path: str = "prompts.jsonl"
    documents_path: str = "documents.jsonl"
    vectors_path: str = "vectors.npy"
    index_path: str = "index.faiss"
    model_info_path: str = "embedding_model_name.txt"

    # Модели
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "llama-3.3-70b-versatile"  # Groq

    llm_base_url: str = "https://api.groq.com/openai/v1"

    # Параметры
    default_top_k: int = 5
    embedding_dim: int = 768
    batch_size: int = 32


config = Config()


# ========================= МОДУЛЬ 1: КОРПУС =====================
class CorpusBuilder:
    """Создание и управление корпусом документов (ваш модуль 1)"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста (ваш код)"""
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def build_from_prompts(cls, input_path: str, output_path: str) -> int:
        """
        Построение корпуса из промптов
        Возвращает количество обработанных документов
        """
        documents = []

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        prompt = data.get('prompt', '')
                        filters = data.get('filters', {})

                        cleaned_prompt = cls.clean_text(prompt)

                        metadata = {
                            "hasOrbit": bool(filters.get('orbitType', '').strip()),
                            "hasMass": bool(filters.get('mass', '').strip()),
                            "hasCoverage": bool(filters.get('coverage', '').strip()),
                            "hasStatus": bool(filters.get('status', '').strip())
                        }

                        documents.append({
                            "id": line_num,
                            "text": cleaned_prompt,
                            "json": filters,
                            "metadata": metadata
                        })

                    except json.JSONDecodeError:
                        logger.warning(f"Ошибка JSON в строке {line_num}")
                        continue

            # Сохранение
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')

            logger.info(f"Создан корпус: {len(documents)} документов -> {output_path}")
            return len(documents)

        except Exception as e:
            logger.error(f"Ошибка построения корпуса: {e}")
            return 0


# ========================= МОДУЛЬ 2: ВЕКТОРИЗАЦИЯ ==============
class Embedder:
    """Генерация и управление эмбеддингами (ваш модуль 2)"""

    def __init__(self, model_name: str = config.embedding_model):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def generate_embeddings(self, input_file: str, output_vectors: str) -> Optional[np.ndarray]:
        """Генерация эмбеддингов для документов"""
        try:
            # Загрузка текстов
            texts = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data['text'])

            logger.info(f"Загружено {len(texts)} текстов")

            # Генерация эмбеддингов
            embeddings = self.model.encode(
                texts,
                batch_size=config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True
            )

            # Сохранение
            np.save(output_vectors, embeddings)
            logger.info(f"Векторы сохранены: {embeddings.shape} -> {output_vectors}")

            # Сохранение информации о модели
            with open(config.model_info_path, 'w', encoding='utf-8') as f:
                f.write(self.model_name)

            return embeddings

        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддингов: {e}")
            return None

    def embed_query(self, query: str) -> np.ndarray:
        """Векторизация одного запроса"""
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding.astype(np.float32)


# ========================= МОДУЛЬ 3: FAISS ПОИСК ================
class FAISSIndex:
    """Управление FAISS индексом (ваш модуль 3)"""

    def __init__(self, index_path: str = config.index_path):
        self.index_path = index_path
        self.index = None
        self.documents = []

    def build_index(self, vectors: np.ndarray) -> bool:
        """Построение индекса из векторов"""
        try:
            if vectors.shape[1] != config.embedding_dim:
                raise ValueError(f"Ожидалась размерность {config.embedding_dim}")

            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)

            # Создание индекса для косинусного сходства
            self.index = faiss.IndexFlatIP(config.embedding_dim)
            self.index.add(vectors)

            # Сохранение
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Индекс создан: {self.index.ntotal} векторов -> {self.index_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка построения индекса: {e}")
            return False

    def load_index(self) -> bool:
        """Загрузка индекса из файла"""
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Индекс загружен: {self.index.ntotal} векторов")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")
            return False

    def load_documents(self, documents_path: str = config.documents_path) -> bool:
        """Загрузка документов для поиска"""
        try:
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = [json.loads(line) for line in f]
            logger.info(f"Документы загружены: {len(self.documents)}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")
            return False

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """Поиск k ближайших документов"""
        if self.index is None or not self.documents:
            raise ValueError("Индекс или документы не загружены")

        # Поиск в FAISS
        distances, indices = self.index.search(query_vector, k)

        # Формирование результатов
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "document": doc,
                    "similarity": float(dist),  # Косинусное сходство
                    "rank": len(results) + 1
                })

        return results


# ========================= МОДУЛЬ 4: ПЕРЕФРАЗИРОВАНИЕ ==========
class QueryRewriter:
    """Перефразирование запросов (ваш модуль 4)"""

    def __init__(self, api_key: str = config.llm_api_key,
                 base_url: str = config.llm_base_url,
                 model: str = config.llm_model):

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model

    def rewrite(self, query: str) -> str:
        """Перефразирование запроса для улучшения поиска"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты помощник для перефразирования запросов о спутниках. "
                            "Перефразируй запрос так, чтобы он максимально точно передавал "
                            "задачу выбора спутника. Уточни: тип орбиты, массу, "
                            "форм-фактор, статус, покрытие. "
                            "Не добавляй новые данные — только переформулируй существующие. "
                            "Не придумывай значения, которых нет в исходном запросе. "
                            "Сохрани исходный смысл полностью."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=150
            )

            rewritten = response.choices[0].message.content.strip()
            logger.info(f"Перефразирование: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Ошибка перефразирования: {e}")
            return query  # Возвращаем оригинал при ошибке


# ========================= МОДУЛЬ 5: RAG ГЕНЕРАЦИЯ =============
class RAGGenerator:
    """Генерация ответов на основе найденных документов"""

    def __init__(self, rewriter: QueryRewriter):
        self.rewriter = rewriter
        self.client = rewriter.client
        self.model = rewriter.model

    def build_rag_prompt(self, original_query: str,
                         rewritten_query: str,
                         retrieved_docs: List[Dict]) -> str:
        """Построение промпта для LLM с контекстом"""

        # Форматирование контекста
        context_lines = []
        for i, result in enumerate(retrieved_docs, 1):
            doc = result["document"]
            context_lines.append(
                f"{i}. [Сходство: {result['similarity']:.3f}] "
                f"Запрос: {doc['text']}\n"
                f"   Фильтры: {json.dumps(doc['json'], ensure_ascii=False)}"
            )

        context = "\n\n".join(context_lines)

        prompt = f"""Ты - ассистент по поиску спутников. Используй предоставленный контекст и ответь строго в формате JSON.

ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {original_query}
ПЕРЕФРАЗИРОВАННЫЙ ЗАПРОС (для поиска): {rewritten_query}

КОНТЕКСТ (найденные похожие запросы и их фильтры):
{context}

АНАЛИЗ И ОТВЕТ:
1. Проанализируй, какие фильтры из контекста соответствуют исходному запросу
2. Сформулируй итоговый JSON с параметрами для поиска спутников
3. Учитывай все указанные параметры: орбиту, массу, покрытие, статус
4. Если параметр не указан явно - оставь пустым

ФОРМАТ ОТВЕТА (JSON):
{{
    "original_query": "строка",
    "rewritten_query": "строка",
    "final_filters": {{
        "orbitType": "строка или пусто",
        "mass": "строка или пусто",
        "coverage": "строка или пусто",
        "status": "строка или пусто",
        "formFactor": "строка или пусто"
    }},
    "retrieved_count": число,
    "average_similarity": число (среднее сходство),
    "reasoning": "строка с объяснением логики выбора фильтров",
    "timestamp": "2025-01-15T12:00:00Z"
}}

Верни ТОЛЬКО JSON, без пояснений и кода:"""

        return prompt

    def generate_response(self, prompt: str) -> Dict:
        """Генерация ответа LLM с валидацией JSON"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Низкая для детерминированности
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content.strip()

                # Пытаемся распарсить JSON
                try:
                    result = json.loads(result_text)

                    # Базовая валидация структуры
                    required_fields = ["original_query", "final_filters", "reasoning"]
                    if all(field in result for field in required_fields):
                        logger.info(f"Успешная генерация (попытка {attempt + 1})")
                        return result
                    else:
                        logger.warning(f"Неполный JSON, пробуем снова...")

                except json.JSONDecodeError as e:
                    logger.warning(f"Невалидный JSON (попытка {attempt + 1}): {e}")

                # Self-correction prompt для повторной попытки
                if attempt < max_retries - 1:
                    correction_prompt = f"""Твой предыдущий ответ содержал невалидный JSON. 
                    Исправь ошибки и верни ТОЛЬКО корректный JSON в указанном формате.

                    Формат:
                    {{
                        "original_query": "строка",
                        "rewritten_query": "строка",
                        "final_filters": {{...}},
                        "retrieved_count": число,
                        "average_similarity": число,
                        "reasoning": "строка",
                        "timestamp": "строка ISO"
                    }}

                    Невалидный ответ: {result_text}

                    Исправленный JSON:"""

                    # Используем тот же промпт для повторной попытки
                    prompt = correction_prompt

            except Exception as e:
                logger.error(f"Ошибка генерации (попытка {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise

        # Fallback ответ
        return {
            "original_query": "",
            "rewritten_query": "",
            "final_filters": {},
            "retrieved_count": 0,
            "average_similarity": 0.0,
            "reasoning": "Ошибка генерации ответа",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# ========================= ГЛАВНЫЙ ПАЙПЛАЙН ====================
class RAGPipeline:
    """Основной пайплайн, объединяющий все модули"""

    def __init__(self, config: Config = config):
        self.config = config
        self.corpus_builder = CorpusBuilder()
        self.embedder = Embedder(config.embedding_model)
        self.faiss_index = FAISSIndex(config.index_path)
        self.query_rewriter = QueryRewriter(
            config.llm_api_key,
            config.llm_base_url,
            config.llm_model
        )
        self.rag_generator = RAGGenerator(self.query_rewriter)

        # Состояние системы
        self.is_initialized = False

    def initialize(self, rebuild: bool = False) -> bool:
        """Инициализация пайплайна (загрузка или пересборка)"""
        try:
            if rebuild or not all([
                os.path.exists(self.config.documents_path),
                os.path.exists(self.config.vectors_path),
                os.path.exists(self.config.index_path)
            ]):
                logger.info("Пересборка RAG системы...")

                # 1. Построение корпуса
                doc_count = self.corpus_builder.build_from_prompts(
                    self.config.prompts_path,
                    self.config.documents_path
                )
                if doc_count == 0:
                    return False

                # 2. Генерация эмбеддингов
                embeddings = self.embedder.generate_embeddings(
                    self.config.documents_path,
                    self.config.vectors_path
                )
                if embeddings is None:
                    return False

                # 3. Построение FAISS индекса
                if not self.faiss_index.build_index(embeddings):
                    return False

            # Загрузка индекса и документов
            if not self.faiss_index.load_index():
                return False

            if not self.faiss_index.load_documents(self.config.documents_path):
                return False

            self.is_initialized = True
            logger.info("✅ RAG система инициализирована")
            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            return False

    def rag_inference(self, query: str, k: int = 5) -> Dict:
        """
        Основной метод пайплайна

        Args:
            query: Пользовательский запрос
            k: Количество возвращаемых похожих документов

        Returns:
            Словарь с результатами в формате JSON
        """
        if not self.is_initialized:
            raise RuntimeError("Пайплайн не инициализирован. Вызовите initialize() сначала.")

        logger.info(f"🚀 Запуск RAG-пайплайна для запроса: '{query}'")
        start_time = time.time()

        # Шаг 1: Перефразирование запроса
        rewritten_query = self.query_rewriter.rewrite(query)

        # Шаг 2: Векторизация запроса
        query_vector = self.embedder.embed_query(rewritten_query)

        # Шаг 3: Поиск в FAISS
        retrieved_results = self.faiss_index.search(query_vector, k)

        # Шаг 4: Построение RAG промпта
        prompt = self.rag_generator.build_rag_prompt(
            query,
            rewritten_query,
            retrieved_results
        )

        # Шаг 5: Генерация ответа
        response = self.rag_generator.generate_response(prompt)

        # Дополняем результат метаданными
        response["pipeline_metadata"] = {
            "processing_time_seconds": round(time.time() - start_time, 2),
            "retrieved_documents": len(retrieved_results),
            "query_embedding_dim": query_vector.shape[1],
            "model_used": self.config.llm_model
        }

        # Добавляем информацию о найденных документах (для отладки)
        response["retrieved_docs_preview"] = [
            {
                "text": r["document"]["text"][:100] + "..." if len(r["document"]["text"]) > 100 else r["document"][
                    "text"],
                "similarity": round(r["similarity"], 3),
                "has_filters": r["document"]["metadata"]
            }
            for r in retrieved_results[:3]  # Только первые 3 для краткости
        ]

        logger.info(f"✅ Пайплайн завершен за {response['pipeline_metadata']['processing_time_seconds']} сек")
        return response


# ========================= ИСПОЛЬЗОВАНИЕ =======================
def main():
    """Пример использования полного пайплайна"""

    # Создаем пайплайн
    pipeline = RAGPipeline()

    # Инициализируем (загружаем или пересобираем данные)
    print("🔄 Инициализация RAG системы...")
    if not pipeline.initialize(rebuild=False):  # rebuild=True для пересборки
        print("❌ Ошибка инициализации")
        return

    # Тестовые запросы
    test_queries = [
        # "Найди спутники на низкой орбите массой до 100 кг"#,
        # "Покажи активные CubeSat с глобальным покрытием",
        # "Спутники на геостационарной орбите",
        # "Неработающие спутники с покрытием Европы"
        "Найди 5 легкие спутники на низкой орбите"
    ]

    # Обработка запросов
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"ЗАПРОС {i}: {query}")
        print(f"{'=' * 60}")

        try:
            result = pipeline.rag_inference(query, k=3)

            # Красивый вывод
            print(f"\n📊 РЕЗУЛЬТАТ:")
            print(f"Оригинальный запрос: {result['original_query']}")
            print(f"Перефразированный: {result['rewritten_query']}")
            print(f"\n🎯 ФИЛЬТРЫ:")
            for key, value in result['final_filters'].items():
                if value:
                    print(f"  {key}: {value}")

            print(f"\n🤔 ОБЪЯСНЕНИЕ: {result['reasoning'][:200]}...")
            print(f"\n⚡ МЕТАДАННЫЕ: {result['pipeline_metadata']['processing_time_seconds']} сек")

        except Exception as e:
            print(f"❌ Ошибка при обработке запроса: {e}")

    print(f"\n{'=' * 60}")
    print("🎯 ПАЙПЛАЙН УСПЕШНО ПРОТЕСТИРОВАН")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import os

    main()
