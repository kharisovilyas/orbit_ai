"""
RAG Pipeline для поиска спутников с локальной Llama 3.1 8B
Оптимизировано для RTX 2070 Ti (16GB VRAM)
"""

import os
import sys
import json
import numpy as np
import faiss
import time
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from datetime import datetime
import logging

# Настройка логирования - только важные сообщения
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Отключаем лишние логи
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# ========================= КОНФИГУРАЦИЯ =========================
@dataclass
class Config:
    """Конфигурация RAG-системы с локальной Llama"""

    # Пути к данным
    prompts_path: str = "prompts.jsonl"
    documents_path: str = "documents.jsonl"
    vectors_path: str = "vectors.npy"
    index_path: str = "index.faiss"

    # Модель для эмбеддингов (быстрая)
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768

    # Локальная Llama модель
    llama_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"  # или локальный путь
    use_local_llama: bool = True

    # Параметры
    default_top_k: int = 5
    batch_size: int = 32


config = Config()


# ========================= МОДУЛЬ 1: КОРПУС =====================
class CorpusBuilder:
    """Создание и управление корпусом документов"""

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def build_from_prompts(cls, input_path: str, output_path: str) -> int:
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
                        continue

            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')

            logger.info(f"Создан корпус: {len(documents)} документов")
            return len(documents)

        except Exception as e:
            logger.error(f"Ошибка построения корпуса: {e}")
            return 0


# ========================= МОДУЛЬ 2: ВЕКТОРИЗАЦИЯ ==============
class Embedder:
    """Генерация и управление эмбеддингами"""

    def __init__(self, model_name: str = config.embedding_model):
        logger.info(f"Загрузка модели эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.model_name = model_name

    def generate_embeddings(self, input_file: str, output_vectors: str) -> Optional[np.ndarray]:
        try:
            texts = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data['text'])

            logger.info(f"Загружено текстов: {len(texts)}")

            embeddings = self.model.encode(
                texts,
                batch_size=config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            np.save(output_vectors, embeddings)
            logger.info(f"Векторы сохранены: {embeddings.shape}")

            return embeddings

        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддингов: {e}")
            return None

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding.astype(np.float32)


# ========================= МОДУЛЬ 3: FAISS ПОИСК ================
class FAISSIndex:
    """Управление FAISS индексом"""

    def __init__(self, index_path: str = config.index_path):
        self.index_path = index_path
        self.index = None
        self.documents = []

    def build_index(self, vectors: np.ndarray) -> bool:
        try:
            if vectors.shape[1] != config.embedding_dim:
                raise ValueError(f"Ожидалась размерность {config.embedding_dim}")

            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)

            self.index = faiss.IndexFlatIP(config.embedding_dim)
            self.index.add(vectors)

            faiss.write_index(self.index, self.index_path)
            logger.info(f"Индекс создан: {self.index.ntotal} векторов")
            return True

        except Exception as e:
            logger.error(f"Ошибка построения индекса: {e}")
            return False

    def load_index(self) -> bool:
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Индекс загружен: {self.index.ntotal} векторов")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")
            return False

    def load_documents(self, documents_path: str = config.documents_path) -> bool:
        try:
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = [json.loads(line) for line in f]
            logger.info(f"Документы загружены: {len(self.documents)}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")
            return False

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index is None or not self.documents:
            raise ValueError("Индекс или документы не загружены")

        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "document": doc,
                    "similarity": float(dist),
                    "rank": len(results) + 1
                })

        return results


# ========================= МОДУЛЬ 4: ЛОКАЛЬНАЯ LLAMA ============
class LocalLlamaParser:
    """Парсер на основе локальной Llama 3.1 8B"""

    def __init__(self, model_path: str = config.llama_model_path):
        logger.info(f"Инициализация локальной Llama: {model_path}")

        try:
            # Используем transformers для локальной модели
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            # Загрузка модели с 8-битной квантовкой для экономии памяти
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # Автоматически распределит по GPU/CPU
                load_in_8bit=True,  # 8-битная квантовка для 16GB VRAM
                torch_dtype="auto"
            )

            # Создаем пайплайн
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )

            # Системный промпт из вашей конфигурации
            self.system_prompt = """Ты — высокоточный NLU-парсер. Твоя задача — извлекать из запроса пользователя фильтры и возвращать их строго в формате JSON.
Твой ответ ДОЛЖЕН содержать ТОЛЬКО валидный JSON-объект и ничего больше.
Если какой-либо фильтр не упоминается в запросе, его значением ДОЛЖНА быть пустая строка "".

### Пример 1
**Запрос:** Найди 2 активных спутника форм-фактора 12U для России.
**Ответ:**
{"coverage": "Россия", "altitude": "", "orbitType": "", "status": "активен", "formFactor": "12U", "mass": "", "scale": "", "tleDate": "", "numberOfSatellites": "2"}"""

            logger.info("Llama модель загружена успешно")

        except ImportError:
            logger.error("Установите transformers: pip install transformers accelerate bitsandbytes")
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки Llama: {e}")
            raise

    def parse_query(self, query: str, max_retries: int = 2) -> Dict:
        """Парсинг запроса в JSON фильтров"""

        prompt = f"""{self.system_prompt}

**Запрос:** {query}

**Ответ:**"""

        for attempt in range(max_retries):
            try:
                # Генерация ответа
                response = self.pipe(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )

                result_text = response[0]['generated_text'].strip()

                # Извлекаем JSON из ответа
                json_str = self._extract_json(result_text)

                # Парсим JSON
                parsed = json.loads(json_str)

                # Базовая валидация
                if isinstance(parsed, dict):
                    logger.info(f"Успешный парсинг (попытка {attempt + 1})")
                    return parsed
                else:
                    logger.warning(f"Ответ не является словарем, пробуем снова...")

            except Exception as e:
                logger.warning(f"Ошибка парсинга (попытка {attempt + 1}): {e}")

                # Self-correction промпт
                if attempt < max_retries - 1:
                    correction_prompt = f"""Исправь этот JSON на основе ошибки:

Ошибка: {str(e)}

Невалидный ответ: {result_text if 'result_text' in locals() else 'Нет ответа'}

Верни ТОЛЬКО исправленный JSON в формате:
{{"coverage": "", "altitude": "", "orbitType": "", "status": "", "formFactor": "", "mass": "", "scale": "", "tleDate": "", "numberOfSatellites": ""}}

Исправленный JSON:"""

                    prompt = correction_prompt

        # Fallback
        return self._create_empty_filters()

    def _extract_json(self, text: str) -> str:
        """Извлечение JSON из текста ответа"""
        # Ищем JSON объект
        start = text.find('{')
        end = text.rfind('}') + 1

        if start != -1 and end != 0:
            return text[start:end]

        # Если не нашли, возвращаем как есть
        return text

    def _create_empty_filters(self) -> Dict:
        """Создание пустых фильтров"""
        return {
            "coverage": "",
            "altitude": "",
            "orbitType": "",
            "status": "",
            "formFactor": "",
            "mass": "",
            "scale": "",
            "tleDate": "",
            "numberOfSatellites": ""
        }


# ========================= МОДУЛЬ 5: RAG ГЕНЕРАЦИЯ ==============
class RAGGenerator:
    """Генерация финального ответа на основе поиска"""

    def __init__(self, llama_parser: LocalLlamaParser):
        self.llama_parser = llama_parser

    def generate_final_response(self, original_query: str,
                                rewritten_query: str,
                                retrieved_results: List[Dict]) -> Dict:
        """Генерация финального ответа"""

        # 1. Парсим запрос через Llama
        parsed_filters = self.llama_parser.parse_query(original_query)

        # 2. Анализируем найденные документы
        filter_stats = self._analyze_filters(retrieved_results)

        # 3. Создаем финальный ответ
        response = {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "final_filters": parsed_filters,
            "retrieved_count": len(retrieved_results),
            "average_similarity": round(
                sum(r["similarity"] for r in retrieved_results) / len(retrieved_results)
                if retrieved_results else 0, 3
            ),
            "filter_analysis": filter_stats,
            "retrieved_docs_sample": [
                {
                    "text": r["document"]["text"][:80] + "..." if len(r["document"]["text"]) > 80 else r["document"][
                        "text"],
                    "similarity": round(r["similarity"], 3),
                    "filters": r["document"]["json"]
                }
                for r in retrieved_results[:2]  # Только 2 для примера
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        return response

    def _analyze_filters(self, results: List[Dict]) -> Dict:
        """Анализ частоты фильтров в найденных документах"""
        filter_counts = {}

        for result in results:
            filters = result["document"]["json"]
            for key, value in filters.items():
                if value and str(value).strip():
                    if key not in filter_counts:
                        filter_counts[key] = {}
                    if value not in filter_counts[key]:
                        filter_counts[key][value] = 0
                    filter_counts[key][value] += 1

        return filter_counts


# ========================= ГЛАВНЫЙ ПАЙПЛАЙН ====================
class RAGPipeline:
    """Основной пайплайн с локальной Llama"""

    def __init__(self, config: Config = config):
        self.config = config
        self.corpus_builder = CorpusBuilder()
        self.embedder = None
        self.faiss_index = FAISSIndex(config.index_path)
        self.llama_parser = None
        self.rag_generator = None

        self.is_initialized = False

    def initialize(self, rebuild: bool = False) -> bool:
        """Инициализация пайплайна"""
        logger.info("Инициализация RAG системы с локальной Llama")
        total_start = time.time()

        try:
            # 1. Загрузка эмбеддинг модели
            logger.info("Шаг 1: Загрузка модели эмбеддингов")
            self.embedder = Embedder(config.embedding_model)

            # 2. Загрузка или пересборка индекса
            if rebuild or not os.path.exists(config.index_path):
                logger.info("Шаг 2: Пересборка индекса")
                if not os.path.exists(config.documents_path):
                    self.corpus_builder.build_from_prompts(
                        config.prompts_path,
                        config.documents_path
                    )

                embeddings = self.embedder.generate_embeddings(
                    config.documents_path,
                    config.vectors_path
                )

                if embeddings is not None:
                    self.faiss_index.build_index(embeddings)

            # 3. Загрузка индекса и документов
            logger.info("Шаг 3: Загрузка FAISS индекса")
            if not self.faiss_index.load_index():
                return False

            logger.info("Шаг 4: Загрузка документов")
            if not self.faiss_index.load_documents(config.documents_path):
                return False

            # 4. Загрузка локальной Llama (только если нужно)
            if config.use_local_llama:
                logger.info("Шаг 5: Загрузка локальной Llama модели")
                self.llama_parser = LocalLlamaParser(config.llama_model_path)
                self.rag_generator = RAGGenerator(self.llama_parser)

            self.is_initialized = True
            total_time = time.time() - total_start
            logger.info(f"Инициализация завершена за {total_time:.1f} секунд")
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
            Словарь с результатами
        """
        if not self.is_initialized:
            raise RuntimeError("Пайплайн не инициализирован")

        logger.info(f"Обработка запроса: '{query}'")
        start_time = time.time()

        try:
            # Шаг 1: Векторизация запроса
            query_vector = self.embedder.embed_query(query)

            # Шаг 2: Поиск в FAISS
            retrieved_results = self.faiss_index.search(query_vector, k)

            # Шаг 3: Перефразирование (простое, без LLM)
            rewritten_query = self._simple_rewrite(query)

            # Шаг 4: Генерация ответа через Llama
            if self.llama_parser:
                response = self.rag_generator.generate_final_response(
                    query, rewritten_query, retrieved_results
                )
            else:
                # Fallback без Llama
                response = self._create_basic_response(
                    query, rewritten_query, retrieved_results
                )

            # Метаданные
            response["processing_time_seconds"] = round(time.time() - start_time, 2)
            response["model_used"] = config.llama_model_path if config.use_local_llama else "no_llm"

            logger.info(f"Запрос обработан за {response['processing_time_seconds']} сек")
            return response

        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "error": str(e),
                "original_query": query,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def _simple_rewrite(self, query: str) -> str:
        """Простое перефразирование без LLM"""
        replacements = {
            "найди": "найти",
            "покажи": "показать",
            "выведи": "вывести",
            "спутник": "космический аппарат",
            "спутники": "космические аппараты"
        }

        result = query
        for old, new in replacements.items():
            if old in result.lower():
                result = result.replace(old, new)

        return result

    def _create_basic_response(self, query: str, rewritten: str,
                               results: List[Dict]) -> Dict:
        """Создание базового ответа без Llama"""
        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "retrieved_count": len(results),
            "average_similarity": round(
                sum(r["similarity"] for r in results) / len(results) if results else 0, 3
            ),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# ========================= ИСПОЛЬЗОВАНИЕ =======================
def main():
    """Основная функция"""
    print("=" * 60)
    print("RAG ПАЙПЛАЙН С ЛОКАЛЬНОЙ LLAMA 3.1 8B")
    print("=" * 60)

    # Создаем пайплайн
    pipeline = RAGPipeline()

    # Инициализируем
    print("\nИнициализация системы...")
    if not pipeline.initialize(rebuild=False):
        print("Ошибка инициализации")
        return

    # Тестовые запросы
    test_queries = [
        "Найди 5 легких спутников на низкой орбите",
        "Покажи активные CubeSat с глобальным покрытием",
        "Спутники на геостационарной орбите для Европы",
        "Неработающие спутники массой до 1000 кг"
    ]

    # Обработка запросов
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"ЗАПРОС {i}: {query}")

        try:
            result = pipeline.rag_inference(query, k=3)

            # Вывод результатов
            if "error" in result:
                print(f"ОШИБКА: {result['error']}")
                continue

            print(f"\nРЕЗУЛЬТАТ:")
            print(f"Оригинал: {result['original_query']}")
            print(f"Перефразированный: {result['rewritten_query']}")

            if 'final_filters' in result:
                print(f"\nФИЛЬТРЫ:")
                for key, value in result['final_filters'].items():
                    if value:
                        print(f"  {key}: {value}")

            print(f"\nСТАТИСТИКА:")
            print(f"  Найдено документов: {result['retrieved_count']}")
            print(f"  Среднее сходство: {result.get('average_similarity', 0)}")
            print(f"  Время обработки: {result['processing_time_seconds']} сек")

        except Exception as e:
            print(f"ОШИБКА: {e}")

    print(f"\n{'=' * 60}")
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()