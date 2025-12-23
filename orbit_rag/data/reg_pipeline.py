# # """
# # RAG Pipeline для поиска спутников
# # Объединяет: подготовку корпуса, векторизацию, FAISS поиск, перефразирование и генерацию ответов
# # """
# # import os
# # import json
# # import numpy as np
# # import faiss
# # import openai
# # import time
# # import re
# # from typing import List, Dict, Any, Optional
# # from sentence_transformers import SentenceTransformer
# # from dataclasses import dataclass
# # from datetime import datetime
# # import logging
# # from query_rewriter import rewrite_query
# #
# # # Настройка логирования
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)
# #
# #
# # # ========================= КОНФИГУРАЦИЯ =========================
# # @dataclass
# # class Config:
# #     """Конфигурация всей RAG-системы"""
# #     # Пути к данным
# #     prompts_path: str = "prompts.jsonl"
# #     documents_path: str = "documents.jsonl"
# #     vectors_path: str = "vectors.npy"
# #     index_path: str = "index.faiss"
# #     model_info_path: str = "embedding_model_name.txt"
# #
# #     # Модели
# #     embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
# #     llm_model: str = "llama-3.3-70b-versatile"  # Groq
# #     llm_api_key: str = os.getenv("GROQ_API_KEY", "")
# #
# #     llm_base_url: str = "https://api.groq.com/openai/v1"
# #
# #     # Параметры
# #     default_top_k: int = 5
# #     embedding_dim: int = 768
# #     batch_size: int = 32
# #
# #
# # config = Config()
# #
# #
# # # ========================= МОДУЛЬ 1: КОРПУС =====================
# # class CorpusBuilder:
# #     """Создание и управление корпусом документов (ваш модуль 1)"""
# #
# #     @staticmethod
# #     def clean_text(text: str) -> str:
# #         """Очистка текста (ваш код)"""
# #         if not isinstance(text, str):
# #             text = str(text)
# #         text = re.sub(r'\s+', ' ', text)
# #         return text.strip()
# #
# #     @classmethod
# #     def build_from_prompts(cls, input_path: str, output_path: str) -> int:
# #         """
# #         Построение корпуса из промптов
# #         Возвращает количество обработанных документов
# #         """
# #         documents = []
# #
# #         try:
# #             with open(input_path, 'r', encoding='utf-8') as f:
# #                 for line_num, line in enumerate(f, 1):
# #                     if not line.strip():
# #                         continue
# #
# #                     try:
# #                         data = json.loads(line)
# #                         prompt = data.get('prompt', '')
# #                         filters = data.get('filters', {})
# #
# #                         cleaned_prompt = cls.clean_text(prompt)
# #
# #                         metadata = {
# #                             "hasOrbit": bool(filters.get('orbitType', '').strip()),
# #                             "hasMass": bool(filters.get('mass', '').strip()),
# #                             "hasCoverage": bool(filters.get('coverage', '').strip()),
# #                             "hasStatus": bool(filters.get('status', '').strip())
# #                         }
# #
# #                         documents.append({
# #                             "id": line_num,
# #                             "text": cleaned_prompt,
# #                             "json": filters,
# #                             "metadata": metadata
# #                         })
# #
# #                     except json.JSONDecodeError:
# #                         logger.warning(f"Ошибка JSON в строке {line_num}")
# #                         continue
# #
# #             # Сохранение
# #             with open(output_path, 'w', encoding='utf-8') as f:
# #                 for doc in documents:
# #                     f.write(json.dumps(doc, ensure_ascii=False) + '\n')
# #
# #             logger.info(f"Создан корпус: {len(documents)} документов -> {output_path}")
# #             return len(documents)
# #
# #         except Exception as e:
# #             logger.error(f"Ошибка построения корпуса: {e}")
# #             return 0
# #
# #
# # # ========================= МОДУЛЬ 2: ВЕКТОРИЗАЦИЯ ==============
# # class Embedder:
# #     """Генерация и управление эмбеддингами (ваш модуль 2)"""
# #
# #     def __init__(self, model_name: str = config.embedding_model):
# #         self.model = SentenceTransformer(model_name)
# #         self.model_name = model_name
# #
# #     def generate_embeddings(self, input_file: str, output_vectors: str) -> Optional[np.ndarray]:
# #         """Генерация эмбеддингов для документов"""
# #         try:
# #             # Загрузка текстов
# #             texts = []
# #             with open(input_file, 'r', encoding='utf-8') as f:
# #                 for line in f:
# #                     data = json.loads(line)
# #                     texts.append(data['text'])
# #
# #             logger.info(f"Загружено {len(texts)} текстов")
# #
# #             # Генерация эмбеддингов
# #             embeddings = self.model.encode(
# #                 texts,
# #                 batch_size=config.batch_size,
# #                 normalize_embeddings=True,
# #                 show_progress_bar=True
# #             )
# #
# #             # Сохранение
# #             np.save(output_vectors, embeddings)
# #             logger.info(f"Векторы сохранены: {embeddings.shape} -> {output_vectors}")
# #
# #             # Сохранение информации о модели
# #             with open(config.model_info_path, 'w', encoding='utf-8') as f:
# #                 f.write(self.model_name)
# #
# #             return embeddings
# #
# #         except Exception as e:
# #             logger.error(f"Ошибка генерации эмбеддингов: {e}")
# #             return None
# #
# #     def embed_query(self, query: str) -> np.ndarray:
# #         """Векторизация одного запроса"""
# #         embedding = self.model.encode([query], normalize_embeddings=True)
# #         return embedding.astype(np.float32)
# #
# #
# # # ========================= МОДУЛЬ 3: FAISS ПОИСК ================
# # class FAISSIndex:
# #     """Управление FAISS индексом (ваш модуль 3)"""
# #
# #     def __init__(self, index_path: str = config.index_path):
# #         self.index_path = index_path
# #         self.index = None
# #         self.documents = []
# #
# #     def build_index(self, vectors: np.ndarray) -> bool:
# #         """Построение индекса из векторов"""
# #         try:
# #             if vectors.shape[1] != config.embedding_dim:
# #                 raise ValueError(f"Ожидалась размерность {config.embedding_dim}")
# #
# #             if vectors.dtype != np.float32:
# #                 vectors = vectors.astype(np.float32)
# #
# #             # Создание индекса для косинусного сходства
# #             self.index = faiss.IndexFlatIP(config.embedding_dim)
# #             self.index.add(vectors)
# #
# #             # Сохранение
# #             faiss.write_index(self.index, self.index_path)
# #             logger.info(f"Индекс создан: {self.index.ntotal} векторов -> {self.index_path}")
# #             return True
# #
# #         except Exception as e:
# #             logger.error(f"Ошибка построения индекса: {e}")
# #             return False
# #
# #     def load_index(self) -> bool:
# #         """Загрузка индекса из файла"""
# #         try:
# #             self.index = faiss.read_index(self.index_path)
# #             logger.info(f"Индекс загружен: {self.index.ntotal} векторов")
# #             return True
# #         except Exception as e:
# #             logger.error(f"Ошибка загрузки индекса: {e}")
# #             return False
# #
# #     def load_documents(self, documents_path: str = config.documents_path) -> bool:
# #         """Загрузка документов для поиска"""
# #         try:
# #             with open(documents_path, 'r', encoding='utf-8') as f:
# #                 self.documents = [json.loads(line) for line in f]
# #             logger.info(f"Документы загружены: {len(self.documents)}")
# #             return True
# #         except Exception as e:
# #             logger.error(f"Ошибка загрузки документов: {e}")
# #             return False
# #
# #     def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
# #         """Поиск k ближайших документов"""
# #         if self.index is None or not self.documents:
# #             raise ValueError("Индекс или документы не загружены")
# #
# #         # Поиск в FAISS
# #         distances, indices = self.index.search(query_vector, k)
# #
# #         # Формирование результатов
# #         results = []
# #         for dist, idx in zip(distances[0], indices[0]):
# #             if idx < len(self.documents):
# #                 doc = self.documents[idx]
# #                 results.append({
# #                     "document": doc,
# #                     "similarity": float(dist),  # Косинусное сходство
# #                     "rank": len(results) + 1
# #                 })
# #
# #         return results
# #
# #
# # # ========================= МОДУЛЬ 4: ПЕРЕФРАЗИРОВАНИЕ ==========
# # class QueryRewriter:
# #     """Перефразирование запросов (ваш модуль 4)"""
# #
# #     def __init__(self, api_key: str = config.llm_api_key,
# #                  base_url: str = config.llm_base_url,
# #                  model: str = config.llm_model):
# #
# #         self.client = openai.OpenAI(
# #             base_url=base_url,
# #             api_key=api_key
# #         )
# #         self.model = model
# #
# #     def rewrite(self, query: str) -> str:
# #         """Перефразирование запроса для улучшения поиска"""
# #         try:
# #             response = self.client.chat.completions.create(
# #                 model=self.model,
# #                 messages=[
# #                     {
# #                         "role": "system",
# #                         "content": (
# #                             "Ты помощник для перефразирования запросов о спутниках. "
# #                             "Перефразируй запрос так, чтобы он максимально точно передавал "
# #                             "задачу выбора спутника. Уточни: тип орбиты, массу, "
# #                             "форм-фактор, статус, покрытие. "
# #                             "Не добавляй новые данные — только переформулируй существующие. "
# #                             "Не придумывай значения, которых нет в исходном запросе. "
# #                             "Сохрани исходный смысл полностью."
# #                         )
# #                     },
# #                     {"role": "user", "content": query}
# #                 ],
# #                 temperature=0.3,
# #                 max_tokens=150
# #             )
# #
# #             rewritten = response.choices[0].message.content.strip()
# #             logger.info(f"Перефразирование: '{query}' -> '{rewritten}'")
# #             return rewritten
# #
# #         except Exception as e:
# #             logger.error(f"Ошибка перефразирования: {e}")
# #             return query  # Возвращаем оригинал при ошибке
# #
# #
# # # ========================= МОДУЛЬ 5: RAG ГЕНЕРАЦИЯ =============
# # class RAGGenerator:
# #     """Генерация ответов на основе найденных документов"""
# #
# #     def __init__(self, rewriter: QueryRewriter):
# #         self.rewriter = rewriter
# #         self.client = rewriter.client
# #         self.model = rewriter.model
# #
# #     def build_rag_prompt(self, original_query: str,
# #                          rewritten_query: str,
# #                          retrieved_docs: List[Dict]) -> str:
# #         """Построение промпта для LLM с контекстом"""
# #
# #         # Форматирование контекста
# #         context_lines = []
# #         for i, result in enumerate(retrieved_docs, 1):
# #             doc = result["document"]
# #             context_lines.append(
# #                 f"{i}. [Сходство: {result['similarity']:.3f}] "
# #                 f"Запрос: {doc['text']}\n"
# #                 f"   Фильтры: {json.dumps(doc['json'], ensure_ascii=False)}"
# #             )
# #
# #         context = "\n\n".join(context_lines)
# #
# #         prompt = f"""Ты - ассистент по поиску спутников. Используй предоставленный контекст и ответь строго в формате JSON.
# #
# # ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {original_query}
# # ПЕРЕФРАЗИРОВАННЫЙ ЗАПРОС (для поиска): {rewritten_query}
# #
# # КОНТЕКСТ (найденные похожие запросы и их фильтры):
# # {context}
# #
# # АНАЛИЗ И ОТВЕТ:
# # 1. Проанализируй, какие фильтры из контекста соответствуют исходному запросу
# # 2. Сформулируй итоговый JSON с параметрами для поиска спутников
# # 3. Учитывай все указанные параметры: орбиту, массу, покрытие, статус
# # 4. Если параметр не указан явно - оставь пустым
# #
# # ФОРМАТ ОТВЕТА (JSON):
# # {{
# #     "original_query": "строка",
# #     "rewritten_query": "строка",
# #     "final_filters": {{
# #         "orbitType": "строка или пусто",
# #         "mass": "строка или пусто",
# #         "coverage": "строка или пусто",
# #         "status": "строка или пусто",
# #         "formFactor": "строка или пусто"
# #     }},
# #     "retrieved_count": число,
# #     "average_similarity": число (среднее сходство),
# #     "reasoning": "строка с объяснением логики выбора фильтров",
# #     "timestamp": "2025-01-15T12:00:00Z"
# # }}
# #
# # Верни ТОЛЬКО JSON, без пояснений и кода:"""
# #
# #         return prompt
# #
# #     def generate_response(self, prompt: str) -> Dict:
# #         """Генерация ответа LLM с валидацией JSON"""
# #         max_retries = 3
# #
# #         for attempt in range(max_retries):
# #             try:
# #                 response = self.client.chat.completions.create(
# #                     model=self.model,
# #                     messages=[
# #                         {"role": "user", "content": prompt}
# #                     ],
# #                     temperature=0.1,  # Низкая для детерминированности
# #                     max_tokens=500,
# #                     response_format={"type": "json_object"}
# #                 )
# #
# #                 result_text = response.choices[0].message.content.strip()
# #
# #                 # Пытаемся распарсить JSON
# #                 try:
# #                     result = json.loads(result_text)
# #
# #                     # Базовая валидация структуры
# #                     required_fields = ["original_query", "final_filters", "reasoning"]
# #                     if all(field in result for field in required_fields):
# #                         logger.info(f"Успешная генерация (попытка {attempt + 1})")
# #                         return result
# #                     else:
# #                         logger.warning(f"Неполный JSON, пробуем снова...")
# #
# #                 except json.JSONDecodeError as e:
# #                     logger.warning(f"Невалидный JSON (попытка {attempt + 1}): {e}")
# #
# #                 # Self-correction prompt для повторной попытки
# #                 if attempt < max_retries - 1:
# #                     correction_prompt = f"""Твой предыдущий ответ содержал невалидный JSON.
# #                     Исправь ошибки и верни ТОЛЬКО корректный JSON в указанном формате.
# #
# #                     Формат:
# #                     {{
# #                         "original_query": "строка",
# #                         "rewritten_query": "строка",
# #                         "final_filters": {{...}},
# #                         "retrieved_count": число,
# #                         "average_similarity": число,
# #                         "reasoning": "строка",
# #                         "timestamp": "строка ISO"
# #                     }}
# #
# #                     Невалидный ответ: {result_text}
# #
# #                     Исправленный JSON:"""
# #
# #                     # Используем тот же промпт для повторной попытки
# #                     prompt = correction_prompt
# #
# #             except Exception as e:
# #                 logger.error(f"Ошибка генерации (попытка {attempt + 1}): {e}")
# #                 if attempt == max_retries - 1:
# #                     raise
# #
# #         # Fallback ответ
# #         return {
# #             "original_query": "",
# #             "rewritten_query": "",
# #             "final_filters": {},
# #             "retrieved_count": 0,
# #             "average_similarity": 0.0,
# #             "reasoning": "Ошибка генерации ответа",
# #             "timestamp": datetime.utcnow().isoformat() + "Z"
# #         }
# #
# #
# # # ========================= ГЛАВНЫЙ ПАЙПЛАЙН ====================
# # class RAGPipeline:
# #     """Основной пайплайн, объединяющий все модули"""
# #
# #     def __init__(self, config: Config = config):
# #         self.config = config
# #         self.corpus_builder = CorpusBuilder()
# #         self.embedder = Embedder(config.embedding_model)
# #         self.faiss_index = FAISSIndex(config.index_path)
# #         self.query_rewriter = QueryRewriter(
# #             config.llm_api_key,
# #             config.llm_base_url,
# #             config.llm_model
# #         )
# #         self.rag_generator = RAGGenerator(self.query_rewriter)
# #
# #         # Состояние системы
# #         self.is_initialized = False
# #
# #     def initialize(self, rebuild: bool = False) -> bool:
# #         """Инициализация пайплайна (загрузка или пересборка)"""
# #         try:
# #             if rebuild or not all([
# #                 os.path.exists(self.config.documents_path),
# #                 os.path.exists(self.config.vectors_path),
# #                 os.path.exists(self.config.index_path)
# #             ]):
# #                 logger.info("Пересборка RAG системы...")
# #
# #                 # 1. Построение корпуса
# #                 doc_count = self.corpus_builder.build_from_prompts(
# #                     self.config.prompts_path,
# #                     self.config.documents_path
# #                 )
# #                 if doc_count == 0:
# #                     return False
# #
# #                 # 2. Генерация эмбеддингов
# #                 embeddings = self.embedder.generate_embeddings(
# #                     self.config.documents_path,
# #                     self.config.vectors_path
# #                 )
# #                 if embeddings is None:
# #                     return False
# #
# #                 # 3. Построение FAISS индекса
# #                 if not self.faiss_index.build_index(embeddings):
# #                     return False
# #
# #             # Загрузка индекса и документов
# #             if not self.faiss_index.load_index():
# #                 return False
# #
# #             if not self.faiss_index.load_documents(self.config.documents_path):
# #                 return False
# #
# #             self.is_initialized = True
# #             logger.info("✅ RAG система инициализирована")
# #             return True
# #
# #         except Exception as e:
# #             logger.error(f"Ошибка инициализации: {e}")
# #             return False
# #
# #     def rag_inference(self, query: str, k: int = 5) -> Dict:
# #         """
# #         Основной метод пайплайна
# #
# #         Args:
# #             query: Пользовательский запрос
# #             k: Количество возвращаемых похожих документов
# #
# #         Returns:
# #             Словарь с результатами в формате JSON
# #         """
# #         if not self.is_initialized:
# #             raise RuntimeError("Пайплайн не инициализирован. Вызовите initialize() сначала.")
# #
# #         logger.info(f"🚀 Запуск RAG-пайплайна для запроса: '{query}'")
# #         start_time = time.time()
# #
# #         # Шаг 1: Перефразирование запроса
# #         rewritten_query = self.query_rewriter.rewrite(query)
# #
# #         # Шаг 2: Векторизация запроса
# #         query_vector = self.embedder.embed_query(rewritten_query)
# #
# #         # Шаг 3: Поиск в FAISS
# #         retrieved_results = self.faiss_index.search(query_vector, k)
# #
# #         # Шаг 4: Построение RAG промпта
# #         prompt = self.rag_generator.build_rag_prompt(
# #             query,
# #             rewritten_query,
# #             retrieved_results
# #         )
# #
# #         # Шаг 5: Генерация ответа
# #         response = self.rag_generator.generate_response(prompt)
# #
# #         # Дополняем результат метаданными
# #         response["pipeline_metadata"] = {
# #             "processing_time_seconds": round(time.time() - start_time, 2),
# #             "retrieved_documents": len(retrieved_results),
# #             "query_embedding_dim": query_vector.shape[1],
# #             "model_used": self.config.llm_model
# #         }
# #
# #         # Добавляем информацию о найденных документах (для отладки)
# #         response["retrieved_docs_preview"] = [
# #             {
# #                 "text": r["document"]["text"][:100] + "..." if len(r["document"]["text"]) > 100 else r["document"][
# #                     "text"],
# #                 "similarity": round(r["similarity"], 3),
# #                 "has_filters": r["document"]["metadata"]
# #             }
# #             for r in retrieved_results[:3]  # Только первые 3 для краткости
# #         ]
# #
# #         logger.info(f"✅ Пайплайн завершен за {response['pipeline_metadata']['processing_time_seconds']} сек")
# #         return response
# #
# #
# # # ========================= ИСПОЛЬЗОВАНИЕ =======================
# # def main():
# #     """Пример использования полного пайплайна"""
# #
# #     # Создаем пайплайн
# #     pipeline = RAGPipeline()
# #
# #     # Инициализируем (загружаем или пересобираем данные)
# #     print("🔄 Инициализация RAG системы...")
# #     if not pipeline.initialize(rebuild=False):  # rebuild=True для пересборки
# #         print("❌ Ошибка инициализации")
# #         return
# #
# #     # Тестовые запросы
# #     test_queries = [
# #         # "Найди спутники на низкой орбите массой до 100 кг"#,
# #         # "Покажи активные CubeSat с глобальным покрытием",
# #         # "Спутники на геостационарной орбите",
# #         # "Неработающие спутники с покрытием Европы"
# #         "Найди 5 легкие спутники на низкой орбите"
# #     ]
# #
# #     # Обработка запросов
# #     for i, query in enumerate(test_queries, 1):
# #         print(f"\n{'=' * 60}")
# #         print(f"ЗАПРОС {i}: {query}")
# #         print(f"{'=' * 60}")
# #
# #         try:
# #             result = pipeline.rag_inference(query, k=3)
# #
# #             # Красивый вывод
# #             print(f"\n📊 РЕЗУЛЬТАТ:")
# #             print(f"Оригинальный запрос: {result['original_query']}")
# #             print(f"Перефразированный: {result['rewritten_query']}")
# #             print(f"\n🎯 ФИЛЬТРЫ:")
# #             for key, value in result['final_filters'].items():
# #                 if value:
# #                     print(f"  {key}: {value}")
# #
# #             print(f"\n🤔 ОБЪЯСНЕНИЕ: {result['reasoning'][:200]}...")
# #             print(f"\n⚡ МЕТАДАННЫЕ: {result['pipeline_metadata']['processing_time_seconds']} сек")
# #
# #         except Exception as e:
# #             print(f"❌ Ошибка при обработке запроса: {e}")
# #
# #     print(f"\n{'=' * 60}")
# #     print("🎯 ПАЙПЛАЙН УСПЕШНО ПРОТЕСТИРОВАН")
# #     print(f"{'=' * 60}")
# #
# #
# # if __name__ == "__main__":
# #     import os
# #
# #     main()
#
#
# """
# RAG Pipeline для поиска спутников
# Объединяет: подготовку корпуса, векторизацию, FAISS поиск, перефразирование и генерацию ответов
# """
# import os
# import json
# import numpy as np
# import faiss
# import openai
# import time
# import re
# from typing import List, Dict, Any, Optional
# from sentence_transformers import SentenceTransformer
# from dataclasses import dataclass
# from datetime import datetime
# import logging
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# # ========================= КОНФИГУРАЦИЯ =========================
# @dataclass
# class Config:
#     """Конфигурация всей RAG-системы"""
#     # Пути к данным
#     prompts_path: str = "prompts.jsonl"
#     documents_path: str = "documents.jsonl"
#     vectors_path: str = "vectors.npy"
#     index_path: str = "index.faiss"
#     model_info_path: str = "embedding_model_name.txt"
#
#     # Модели
#     embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
#     llm_model: str = "llama-3.3-70b-versatile"  # Groq
#     llm_api_key: str = os.getenv("GROQ_API_KEY", "")
#
#     llm_base_url: str = "https://api.groq.com/openai/v1"
#
#     # Параметры
#     default_top_k: int = 5
#     embedding_dim: int = 768
#     batch_size: int = 32
#
#
# config = Config()
#
#
# # ========================= МОДУЛЬ 1: КОРПУС =====================
# class CorpusBuilder:
#     """Создание и управление корпусом документов (ваш модуль 1)"""
#
#     @staticmethod
#     def clean_text(text: str) -> str:
#         """Очистка текста (ваш код)"""
#         if not isinstance(text, str):
#             text = str(text)
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip()
#
#     @classmethod
#     def build_from_prompts(cls, input_path: str, output_path: str) -> int:
#         """
#         Построение корпуса из промптов
#         Возвращает количество обработанных документов
#         """
#         documents = []
#
#         try:
#             with open(input_path, 'r', encoding='utf-8') as f:
#                 for line_num, line in enumerate(f, 1):
#                     if not line.strip():
#                         continue
#
#                     try:
#                         data = json.loads(line)
#                         prompt = data.get('prompt', '')
#                         filters = data.get('filters', {})
#
#                         cleaned_prompt = cls.clean_text(prompt)
#
#                         metadata = {
#                             "hasOrbit": bool(filters.get('orbitType', '').strip()),
#                             "hasMass": bool(filters.get('mass', '').strip()),
#                             "hasCoverage": bool(filters.get('coverage', '').strip()),
#                             "hasStatus": bool(filters.get('status', '').strip())
#                         }
#
#                         documents.append({
#                             "id": line_num,
#                             "text": cleaned_prompt,
#                             "json": filters,
#                             "metadata": metadata
#                         })
#
#                     except json.JSONDecodeError:
#                         logger.warning(f"Ошибка JSON в строке {line_num}")
#                         continue
#
#             # Сохранение
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 for doc in documents:
#                     f.write(json.dumps(doc, ensure_ascii=False) + '\n')
#
#             logger.info(f"Создан корпус: {len(documents)} документов -> {output_path}")
#             return len(documents)
#
#         except Exception as e:
#             logger.error(f"Ошибка построения корпуса: {e}")
#             return 0
#
#
# # ========================= МОДУЛЬ 2: ВЕКТОРИЗАЦИЯ ==============
# class Embedder:
#     """Генерация и управление эмбеддингами (ваш модуль 2)"""
#
#     def __init__(self, model_name: str = config.embedding_model):
#         self.model = SentenceTransformer(model_name)
#         self.model_name = model_name
#
#     def generate_embeddings(self, input_file: str, output_vectors: str) -> Optional[np.ndarray]:
#         """Генерация эмбеддингов для документов"""
#         try:
#             # Загрузка текстов
#             texts = []
#             with open(input_file, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     data = json.loads(line)
#                     texts.append(data['text'])
#
#             logger.info(f"Загружено {len(texts)} текстов")
#
#             # Генерация эмбеддингов
#             embeddings = self.model.encode(
#                 texts,
#                 batch_size=config.batch_size,
#                 normalize_embeddings=True,
#                 show_progress_bar=True
#             )
#
#             # Сохранение
#             np.save(output_vectors, embeddings)
#             logger.info(f"Векторы сохранены: {embeddings.shape} -> {output_vectors}")
#
#             # Сохранение информации о модели
#             with open(config.model_info_path, 'w', encoding='utf-8') as f:
#                 f.write(self.model_name)
#
#             return embeddings
#
#         except Exception as e:
#             logger.error(f"Ошибка генерации эмбеддингов: {e}")
#             return None
#
#     def embed_query(self, query: str) -> np.ndarray:
#         """Векторизация одного запроса"""
#         embedding = self.model.encode([query], normalize_embeddings=True)
#         return embedding.astype(np.float32)
#
#
# # ========================= МОДУЛЬ 3: FAISS ПОИСК ================
# class FAISSIndex:
#     """Управление FAISS индексом (ваш модуль 3)"""
#
#     def __init__(self, index_path: str = config.index_path):
#         self.index_path = index_path
#         self.index = None
#         self.documents = []
#     def build_index(self, vectors: np.ndarray) -> bool:
#         """Построение индекса из векторов"""
#         try:
#             if vectors.shape[1] != config.embedding_dim:
#                 raise ValueError(f"Ожидалась размерность {config.embedding_dim}")
#
#             if vectors.dtype != np.float32:
#                 vectors = vectors.astype(np.float32)
#
#             # Создание индекса для косинусного сходства
#             self.index = faiss.IndexFlatIP(config.embedding_dim)
#             self.index.add(vectors)
#
#             # Сохранение
#             faiss.write_index(self.index, self.index_path)
#             logger.info(f"Индекс создан: {self.index.ntotal} векторов -> {self.index_path}")
#             return True
#
#         except Exception as e:
#             logger.error(f"Ошибка построения индекса: {e}")
#             return False
#
#     def load_index(self) -> bool:
#         """Загрузка индекса из файла"""
#         try:
#             self.index = faiss.read_index(self.index_path)
#             logger.info(f"Индекс загружен: {self.index.ntotal} векторов")
#             return True
#         except Exception as e:
#             logger.error(f"Ошибка загрузки индекса: {e}")
#             return False
#
#     def load_documents(self, documents_path: str = config.documents_path) -> bool:
#         """Загрузка документов для поиска"""
#         try:
#             with open(documents_path, 'r', encoding='utf-8') as f:
#                 self.documents = [json.loads(line) for line in f]
#             logger.info(f"Документы загружены: {len(self.documents)}")
#             return True
#         except Exception as e:
#             logger.error(f"Ошибка загрузки документов: {e}")
#             return False
#
#     def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
#         """Поиск k ближайших документов"""
#         if self.index is None or not self.documents:
#             raise ValueError("Индекс или документы не загружены")
#
#         # Поиск в FAISS
#         distances, indices = self.index.search(query_vector, k)
#
#         # Формирование результатов
#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx < len(self.documents):
#                 doc = self.documents[idx]
#                 results.append({
#                     "document": doc,
#                     "similarity": float(dist),  # Косинусное сходство
#                     "rank": len(results) + 1
#                 })
#
#         return results
#
#
# # ========================= МОДУЛЬ 4: ПЕРЕФРАЗИРОВАНИЕ ==========
# class FastQueryRewriter:
#     """Быстрый перефразировщик на основе вашего работающего кода из query_rewriter.py"""
#
#     def __init__(self, api_key: str = config.llm_api_key,
#                  base_url: str = config.llm_base_url,
#                  model: str = config.llm_model):
#
#         self.client = openai.OpenAI(
#             base_url=base_url,
#             api_key=api_key,
#             timeout=30.0,
#             max_retries=1
#         )
#         self.model = model
#
#     def rewrite(self, query: str) -> str:
#         """Перефразирование запроса с использованием вашего рабочего промпта"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "Ты помощник для перефразирования запросов о спутниках. Перефразируй запрос так, чтобы он максимально точно передавал задачу выбора спутника. Уточни: тип орбиты, массу, форм-фактор, статус, покрытие. Не добавляй новые данные — только переформулируй существующие. Не придумывай значения, которых нет в исходном запросе. Сохрани исходный смысл полностью."
#                     },
#                     {"role": "user", "content": query}
#                 ],
#                 temperature=0.3,
#                 max_tokens=150,
#                 timeout=15.0
#             )
#
#             rewritten = response.choices[0].message.content.strip()
#             logger.info(f"Перефразирование: '{query}' -> '{rewritten}'")
#             return rewritten
#
#         except Exception as e:
#             logger.warning(f"Ошибка перефразирования, используем оригинал: {e}")
#             return query  # Возвращаем оригинал при ошибке
#
#
# # ========================= МОДУЛЬ 5: RAG ГЕНЕРАЦИЯ =============
# class RAGGenerator:
#     """Генерация ответов на основе найденных документов"""
#
#     def __init__(self, rewriter: FastQueryRewriter):
#         self.rewriter = rewriter
#         self.client = rewriter.client
#         self.model = rewriter.model
#
#     def build_rag_prompt(self, original_query: str,
#                          rewritten_query: str,
#                          retrieved_docs: List[Dict]) -> str:
#         """Построение промпта для LLM с контекстом"""
#
#         # Форматирование контекста
#         context_lines = []
#         for i, result in enumerate(retrieved_docs, 1):
#             doc = result["document"]
#             context_lines.append(
#                 f"{i}. [Сходство: {result['similarity']:.3f}] "
#                 f"Запрос: {doc['text']}\n"
#                 f"   Фильтры: {json.dumps(doc['json'], ensure_ascii=False)}"
#             )
#
#         context = "\n\n".join(context_lines)
#
#         prompt = f"""Ты - ассистент по поиску спутников. Используй предоставленный контекст и ответь строго в формате JSON.
#
# ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {original_query}
# ПЕРЕФРАЗИРОВАННЫЙ ЗАПРОС (для поиска): {rewritten_query}
#
# КОНТЕКСТ (найденные похожие запросы и их фильтры):
# {context}
#
# АНАЛИЗ И ОТВЕТ:
# 1. Проанализируй, какие фильтры из контекста соответствуют исходному запросу
# 2. Сформулируй итоговый JSON с параметрами для поиска спутников
# 3. Учитывай все указанные параметры: орбиту, массу, покрытие, статус
# 4. Если параметр не указан явно - оставь пустым
#
# ФОРМАТ ОТВЕТА (JSON):
# {{
#     "original_query": "строка",
#     "rewritten_query": "строка",
#     "final_filters": {{
#         "orbitType": "строка или пусто",
#         "mass": "строка или пусто",
#         "coverage": "строка или пусто",
#         "status": "строка или пусто",
#         "formFactor": "строка или пусто"
#     }},
#     "retrieved_count": число,
#     "average_similarity": число (среднее сходство),
#     "reasoning": "строка с объяснением логики выбора фильтров",
#     "timestamp": "2025-01-15T12:00:00Z"
# }}
#
# Верни ТОЛЬКО JSON, без пояснений и кода:"""
#
#         return prompt
#
#     def generate_response(self, prompt: str) -> Dict:
#         """Генерация ответа LLM с валидацией JSON"""
#         max_retries = 2  # Уменьшили для скорости
#
#         for attempt in range(max_retries):
#             try:
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.1,
#                     max_tokens=500,
#                     response_format={"type": "json_object"},
#                     timeout=20.0  # Добавили таймаут
#                 )
#
#                 result_text = response.choices[0].message.content.strip()
#
#                 # Пытаемся распарсить JSON
#                 try:
#                     result = json.loads(result_text)
#
#                     # Базовая валидация структуры
#                     required_fields = ["original_query", "final_filters", "reasoning"]
#                     if all(field in result for field in required_fields):
#                         logger.info(f"Успешная генерация (попытка {attempt + 1})")
#                         return result
#                     else:
#                         logger.warning(f"Неполный JSON, пробуем снова...")
#
#                 except json.JSONDecodeError as e:
#                     logger.warning(f"Невалидный JSON (попытка {attempt + 1}): {e}")
#
#                 # Self-correction prompt для повторной попытки
#                 if attempt < max_retries - 1:
#                     correction_prompt = f"""Твой предыдущий ответ содержал невалидный JSON.
#                     Исправь ошибки и верни ТОЛЬКО корректный JSON в указанном формате.
#
#                     Формат:
#                     {{
#                         "original_query": "строка",
#                         "rewritten_query": "строка",
#                         "final_filters": {{...}},
#                         "retrieved_count": число,
#                         "average_similarity": число,
#                         "reasoning": "строка",
#                         "timestamp": "строка ISO"
#                     }}
#
#                     Невалидный ответ: {result_text}
#
#                     Исправленный JSON:"""
#
#                     # Используем тот же промпт для повторной попытки
#                     prompt = correction_prompt
#
#             except Exception as e:
#                 logger.error(f"Ошибка генерации (попытка {attempt + 1}): {e}")
#                 if attempt == max_retries - 1:
#                     # Fallback ответ
#                     return {
#                         "original_query": "",
#                         "rewritten_query": "",
#                         "final_filters": {},
#                         "retrieved_count": 0,
#                         "average_similarity": 0.0,
#                         "reasoning": "Ошибка генерации ответа",
#                         "timestamp": datetime.utcnow().isoformat() + "Z"
#                     }
#
#         # Fallback ответ если все ретраи исчерпаны
#         return {
#             "original_query": "",
#             "rewritten_query": "",
#             "final_filters": {},
#             "retrieved_count": 0,
#             "average_similarity": 0.0,
#             "reasoning": "Ошибка генерации ответа",
#             "timestamp": datetime.utcnow().isoformat() + "Z"
#         }
#
#
# # ========================= ГЛАВНЫЙ ПАЙПЛАЙН ====================
# class RAGPipeline:
#     """Основной пайплайн, объединяющий все модули"""
#
#     def __init__(self, config: Config = config):
#         self.config = config
#         self.corpus_builder = CorpusBuilder()
#         self.embedder = Embedder(config.embedding_model)
#         self.faiss_index = FAISSIndex(config.index_path)
#
#         # Используем быстрый перефразировщик
#         self.query_rewriter = FastQueryRewriter(
#             config.llm_api_key,
#             config.llm_base_url,
#             config.llm_model
#         )
#         self.rag_generator = RAGGenerator(self.query_rewriter)
#
#         # Состояние системы
#         self.is_initialized = False
#
#     def initialize(self, rebuild: bool = False) -> bool:
#         """Инициализация пайплайна (загрузка или пересборка)"""
#         try:
#             if rebuild or not all([
#                 os.path.exists(self.config.documents_path),
#                 os.path.exists(self.config.vectors_path),
#                 os.path.exists(self.config.index_path)
#             ]):
#                 logger.info("Пересборка RAG системы...")
#
#                 # 1. Построение корпуса
#                 doc_count = self.corpus_builder.build_from_prompts(
#                     self.config.prompts_path,
#                     self.config.documents_path
#                 )
#                 if doc_count == 0:
#                     return False
#
#                 # 2. Генерация эмбеддингов
#                 embeddings = self.embedder.generate_embeddings(
#                     self.config.documents_path,
#                     self.config.vectors_path
#                 )
#                 if embeddings is None:
#                     return False
#
#                 # 3. Построение FAISS индекса
#                 if not self.faiss_index.build_index(embeddings):
#                     return False
#
#             # Загрузка индекса и документов
#             if not self.faiss_index.load_index():
#                 return False
#
#             if not self.faiss_index.load_documents(self.config.documents_path):
#                 return False
#
#             self.is_initialized = True
#             logger.info("✅ RAG система инициализирована")
#             return True
#
#         except Exception as e:
#             logger.error(f"Ошибка инициализации: {e}")
#             return False
#
#     def rag_inference(self, query: str, k: int = 5) -> Dict:
#         """
#         Основной метод пайплайна
#
#         Args:
#             query: Пользовательский запрос
#             k: Количество возвращаемых похожих документов
#
#         Returns:
#             Словарь с результатами в формате JSON
#         """
#         if not self.is_initialized:
#             raise RuntimeError("Пайплайн не инициализирован. Вызовите initialize() сначала.")
#
#         logger.info(f"🚀 Запуск RAG-пайплайна для запроса: '{query}'")
#         start_time = time.time()
#
#         timings = {}
#
#         # Шаг 1: Перефразирование запроса
#         step_start = time.time()
#         rewritten_query = self.query_rewriter.rewrite(query)
#         timings["rewriting"] = time.time() - step_start
#
#         # Шаг 2: Векторизация запроса
#         step_start = time.time()
#         query_vector = self.embedder.embed_query(rewritten_query)
#         timings["embedding"] = time.time() - step_start
#
#         # Шаг 3: Поиск в FAISS
#         step_start = time.time()
#         retrieved_results = self.faiss_index.search(query_vector, k)
#         timings["faiss_search"] = time.time() - step_start
#
#         # Шаг 4: Построение RAG промпта
#         step_start = time.time()
#         prompt = self.rag_generator.build_rag_prompt(
#             query,
#             rewritten_query,
#             retrieved_results
#         )
#         timings["prompt_building"] = time.time() - step_start
#
#         # Шаг 5: Генерация ответа
#         step_start = time.time()
#         response = self.rag_generator.generate_response(prompt)
#         timings["generation"] = time.time() - step_start
#
#         # Дополняем результат метаданными
#         total_time = time.time() - start_time
#         response["pipeline_metadata"] = {
#             "processing_time_seconds": round(total_time, 2),
#             "timings": {
#                 "rewriting": round(timings.get("rewriting", 0), 2),
#                 "embedding": round(timings.get("embedding", 0), 2),
#                 "faiss_search": round(timings.get("faiss_search", 0), 2),
#                 "prompt_building": round(timings.get("prompt_building", 0), 2),
#                 "generation": round(timings.get("generation", 0), 2)
#             },
#             "retrieved_documents": len(retrieved_results),
#             "query_embedding_dim": query_vector.shape[1],
#             "model_used": self.config.llm_model
#         }
#
#         # Добавляем информацию о найденных документах (для отладки)
#         response["retrieved_docs_preview"] = [
#             {
#                 "text": r["document"]["text"][:100] + "..." if len(r["document"]["text"]) > 100 else r["document"][
#                     "text"],
#                 "similarity": round(r["similarity"], 3),
#                 "has_filters": r["document"]["metadata"]
#             }
#             for r in retrieved_results[:3]  # Только первые 3 для краткости
#         ]
#
#         logger.info(f"✅ Пайплайн завершен за {response['pipeline_metadata']['processing_time_seconds']} сек")
#         logger.info(f"⏱️ Тайминги: "
#                     f"перефразирование={timings.get('rewriting', 0):.1f}s, "
#                     f"векторизация={timings.get('embedding', 0):.1f}s, "
#                     f"поиск={timings.get('faiss_search', 0):.1f}s, "
#                     f"генерация={timings.get('generation', 0):.1f}s")
#
#         return response
#
#
# # ========================= ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС ====================
# def interactive_mode(pipeline: RAGPipeline):
#     """Интерактивный режим с консольным вводом"""
#     print("\n" + "=" * 70)
#     print("🎯 ИНТЕРАКТИВНЫЙ RAG-ПОИСК СПУТНИКОВ")
#     print("=" * 70)
#     print("Команды:")
#     print("  • Введите запрос для поиска (например: 'спутники на низкой орбите массой до 100 кг')")
#     print("  • 'k=число' - изменить количество возвращаемых документов (по умолчанию: 5)")
#     print("  • 'exit', 'quit', 'выход' - завершить работу")
#     print("  • 'stats' - показать статистику системы")
#     print("=" * 70 + "\n")
#
#     current_k = 5
#
#     while True:
#         try:
#             # Ввод запроса
#             user_input = input("\n🔍 Введите запрос (или команду): ").strip()
#
#             if not user_input:
#                 continue
#
#             # Обработка команд
#             if user_input.lower() in ['exit', 'quit', 'выход']:
#                 print("\n👋 Завершение работы...")
#                 break
#
#             if user_input.lower() == 'stats':
#                 print(f"\n📊 СТАТИСТИКА СИСТЕМЫ:")
#                 print(f"   • Загружено документов: {len(pipeline.faiss_index.documents)}")
#                 print(
#                     f"   • Размер FAISS индекса: {pipeline.faiss_index.index.ntotal if pipeline.faiss_index.index else 0} векторов")
#                 print(f"   • Текущий параметр k: {current_k}")
#                 print(f"   • Модель эмбеддингов: {pipeline.config.embedding_model}")
#                 print(f"   • Модель LLM: {pipeline.config.llm_model}")
#                 continue
#
#             # Проверка команды изменения k
#             if user_input.startswith('k='):
#                 try:
#                     new_k = int(user_input.split('=')[1])
#                     if 1 <= new_k <= 50:
#                         current_k = new_k
#                         print(f"✅ Установлен новый параметр k = {current_k}")
#                     else:
#                         print("❌ Параметр k должен быть от 1 до 50")
#                 except ValueError:
#                     print("❌ Неверный формат команды. Используйте: k=число")
#                 continue
#
#             # Обычный запрос - обработка
#             print(f"\n{'=' * 60}")
#             print(f"🔎 ОБРАБОТКА ЗАПРОСА: '{user_input}'")
#             print(f"📂 Будет возвращено документов: {current_k}")
#             print(f"{'=' * 60}")
#
#             start_time = time.time()
#
#             try:
#                 # Выполнение RAG-пайплайна
#                 result = pipeline.rag_inference(user_input, k=current_k)
#
#                 # Вывод результатов
#                 print(f"\n✅ РЕЗУЛЬТАТ ПОИСКА ({result['pipeline_metadata']['processing_time_seconds']} сек):")
#                 print(f"\n📝 Исходный запрос: {result.get('original_query', user_input)}")
#
#                 if 'rewritten_query' in result and result['rewritten_query'] != user_input:
#                     print(f"🔄 Перефразированный запрос: {result['rewritten_query']}")
#
#                 print(f"\n🎯 ФИЛЬТРЫ ДЛЯ ПОИСКА:")
#                 filters = result.get('final_filters', {})
#                 if filters:
#                     for key, value in filters.items():
#                         if value:  # Выводим только непустые фильтры
#                             print(f"   • {key}: {value}")
#                 else:
#                     print("   (нет явно указанных фильтров)")
#
#                 print(f"\n📊 МЕТАДАННЫЕ:")
#                 print(f"   • Найдено похожих запросов: {result.get('retrieved_count', 0)}")
#                 print(f"   • Среднее сходство: {result.get('average_similarity', 0):.3f}")
#
#                 if 'retrieved_docs_preview' in result and result['retrieved_docs_preview']:
#                     print(f"\n📄 ОБРАЗЦЫ НАЙДЕННЫХ ДОКУМЕНТОВ:")
#                     for i, doc in enumerate(result['retrieved_docs_preview'], 1):
#                         print(f"   {i}. {doc['text']}")
#                         print(f"      Сходство: {doc['similarity']:.3f}")
#
#                 print(f"\n🤔 ОБЪЯСНЕНИЕ ЛОГИКИ:")
#                 reasoning = result.get('reasoning', '')
#                 if reasoning:
#                     # Форматируем длинное объяснение
#                     if len(reasoning) > 300:
#                         print(f"   {reasoning[:300]}...")
#                     else:
#                         print(f"   {reasoning}")
#
#                 print(f"\n⏱️  ТАЙМИНГИ:")
#                 timings = result['pipeline_metadata'].get('timings', {})
#                 for stage, t in timings.items():
#                     stage_name = {
#                         'rewriting': 'Перефразирование',
#                         'embedding': 'Векторизация',
#                         'faiss_search': 'Поиск FAISS',
#                         'prompt_building': 'Построение промпта',
#                         'generation': 'Генерация ответа'
#                     }.get(stage, stage)
#                     print(f"   • {stage_name}: {t:.2f} сек")
#
#                 # Опционально: сохранить результат в файл
#                 save_option = input(f"\n💾 Сохранить результат в файл? (y/n): ").strip().lower()
#                 if save_option == 'y':
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"rag_result_{timestamp}.json"
#                     with open(filename, 'w', encoding='utf-8') as f:
#                         json.dump(result, f, ensure_ascii=False, indent=2)
#                     print(f"✅ Результат сохранен в {filename}")
#
#             except Exception as e:
#                 print(f"❌ Ошибка при обработке запроса: {e}")
#                 logger.error(f"Ошибка в интерактивном режиме: {e}")
#
#             print(f"\n{'=' * 60}")
#
#         except KeyboardInterrupt:
#             print("\n\n👋 Завершение работы по запросу пользователя...")
#             break
#         except EOFError:
#             print("\n\n👋 Завершение работы...")
#             break
#
#
# # ========================= ОСНОВНАЯ ФУНКЦИЯ =======================
# def main():
#     """Главная функция с интерактивным режимом"""
#
#     # Создаем пайплайн
#     print("🚀 Запуск RAG Pipeline для поиска спутников")
#     print("=" * 60)
#
#     pipeline = RAGPipeline()
#
#     # Инициализируем систему
#     print("🔄 Инициализация RAG системы...")
#     if not pipeline.initialize(rebuild=False):
#         print("❌ Ошибка инициализации RAG системы")
#         print("   Проверьте наличие файлов:")
#         print(f"   • {config.prompts_path}")
#         print(f"   • {config.documents_path}")
#         print(f"   • {config.vectors_path}")
#         print(f"   • {config.index_path}")
#         return
#
#     print("✅ RAG система успешно инициализирована")
#     print(f"   • Документов: {len(pipeline.faiss_index.documents)}")
#     print(f"   • Размер индекса: {pipeline.faiss_index.index.ntotal} векторов")
#
#     # Запускаем интерактивный режим
#     interactive_mode(pipeline)
#
#     print("\n" + "=" * 60)
#     print("🎯 RAG-ПАЙПЛАЙН УСПЕШНО ВЫПОЛНЕН")
#     print("=" * 60)
#
#
# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np
import faiss
import openai
import time
import re
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    prompts_path: str = "prompts.jsonl"
    documents_path: str = "documents.jsonl"
    vectors_path: str = "vectors.npy"
    index_path: str = "index.faiss"
    model_info_path: str = "embedding_model_name.txt"

    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_base_url: str = "https://api.groq.com/openai/v1"

    default_top_k: int = 5
    embedding_dim: int = 768
    batch_size: int = 32


config = Config()


class CorpusBuilder:
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
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        prompt = data.get('prompt', '')
                        filters = data.get('filters', {})
                        cleaned_prompt = cls.clean_text(prompt)

                        documents.append({
                            "id": line_num,
                            "text": cleaned_prompt,
                            "json": filters,
                            "metadata": {
                                "hasOrbit": bool(filters.get('orbitType', '').strip()),
                                "hasMass": bool(filters.get('mass', '').strip()),
                                "hasCoverage": bool(filters.get('coverage', '').strip()),
                                "hasStatus": bool(filters.get('status', '').strip())
                            }
                        })
                    except json.JSONDecodeError:
                        continue

            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            return len(documents)
        except Exception as e:
            logger.error(f"Error building corpus: {e}")
            return 0


class Embedder:
    def __init__(self, model_name: str = config.embedding_model):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def generate_embeddings(self, input_file: str, output_vectors: str) -> Optional[np.ndarray]:
        try:
            texts = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    texts.append(json.loads(line)['text'])

            embeddings = self.model.encode(texts, batch_size=config.batch_size, normalize_embeddings=True)
            np.save(output_vectors, embeddings)

            with open(config.model_info_path, 'w', encoding='utf-8') as f:
                f.write(self.model_name)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True).astype(np.float32)


class FAISSIndex:
    def __init__(self, index_path: str = config.index_path):
        self.index_path = index_path
        self.index = None
        self.documents = []

    def build_index(self, vectors: np.ndarray) -> bool:
        try:
            self.index = faiss.IndexFlatIP(config.embedding_dim)
            self.index.add(vectors.astype(np.float32))
            faiss.write_index(self.index, self.index_path)
            return True
        except Exception as e:
            logger.error(f"Index build error: {e}")
            return False

    def load_index(self) -> bool:
        try:
            self.index = faiss.read_index(self.index_path)
            return True
        except Exception as e:
            return False

    def load_documents(self, documents_path: str = config.documents_path) -> bool:
        try:
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = [json.loads(line) for line in f]
            return True
        except Exception as e:
            return False

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index is None or not self.documents:
            return []
        distances, indices = self.index.search(query_vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({"document": self.documents[idx], "similarity": float(dist)})
        return results


class FastQueryRewriter:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def rewrite(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Перефразируй запрос пользователя о спутниках для улучшения поиска. Уточни параметры: орбита, масса, статус, покрытие. Не добавляй лишней информации."},
                    {"role": "user", "content": query}
                ],
                temperature=0.2,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query


class RAGGenerator:
    def __init__(self, rewriter: FastQueryRewriter):
        self.client = rewriter.client
        self.model = rewriter.model

    def generate_response(self, original_query: str, rewritten_query: str, retrieved_docs: List[Dict]) -> Dict:
        context = "\n".join([f"Doc: {d['document']['text']} Filters: {d['document']['json']}" for d in retrieved_docs])

        prompt = f"""
        Запрос: {original_query}
        Контекст: {context}
        Верни JSON с полями: original_query, final_filters (orbitType, mass, coverage, status, formFactor), reasoning.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e), "original_query": original_query}


class RAGPipeline:
    def __init__(self, config: Config = config):
        self.config = config
        self.embedder = Embedder()
        self.faiss_index = FAISSIndex()
        self.rewriter = FastQueryRewriter(config.llm_api_key, config.llm_base_url, config.llm_model)
        self.generator = RAGGenerator(self.rewriter)
        self.is_initialized = False

    def initialize(self) -> bool:
        if not os.path.exists(self.config.index_path):
            builder = CorpusBuilder()
            builder.build_from_prompts(self.config.prompts_path, self.config.documents_path)
            vectors = self.embedder.generate_embeddings(self.config.documents_path, self.config.vectors_path)
            self.faiss_index.build_index(vectors)

        if self.faiss_index.load_index() and self.faiss_index.load_documents():
            self.is_initialized = True
        return self.is_initialized

    def rag_inference(self, query: str, k: int = 5) -> Dict:
        """Основная функция обработки запроса"""
        if not self.is_initialized:
            self.initialize()

        rewritten = self.rewriter.rewrite(query)
        query_vec = self.embedder.embed_query(rewritten)
        docs = self.faiss_index.search(query_vec, k=k)

        result = self.generator.generate_response(query, rewritten, docs)
        result["metadata"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "retrieved_count": len(docs)
        }
        return result


def main():
    pipeline = RAGPipeline()
    # Пример вызова
    response = pipeline.rag_inference("Спутник связи на ГСО массой более 500 кг")
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()