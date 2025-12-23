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
from dotenv import load_dotenv

load_dotenv()

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


# class RAGGenerator:
#     def __init__(self, rewriter: FastQueryRewriter):
#         self.client = rewriter.client
#         self.model = rewriter.model
#
#     def generate_response(self, original_query: str, rewritten_query: str, retrieved_docs: List[Dict]) -> Dict:
#         context = "\n".join([f"Doc: {d['document']['text']} Filters: {d['document']['json']}" for d in retrieved_docs])
#
#         prompt = f"""
#         Запрос: {original_query}
#         Контекст: {context}
#         Верни JSON с полями: original_query, final_filters (orbitType, mass, coverage, status, formFactor), reasoning.
#         """
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.1,
#                 response_format={"type": "json_object"}
#             )
#             return json.loads(response.choices[0].message.content)
#         except Exception as e:
#             return {"error": str(e), "original_query": original_query}
class RAGGenerator:
    def __init__(self, rewriter: FastQueryRewriter):
        self.client = rewriter.client
        self.model = rewriter.model

    def generate_response(self, original_query: str, rewritten_query: str, retrieved_docs: List[Dict]) -> Dict:
        # Подготовка контекста из найденных документов (примеры правильных фильтров)
        context = "\n".join(
            [f"Пример запроса: {d['document']['text']} | Его фильтры: {d['document']['json']}" for d in retrieved_docs])

        # Промпт теперь опирается на перефразированный (технически уточненный) запрос
        prompt = f"""
        Используя примеры из контекста, извлеки параметры для уточненного запроса.

        ИСХОДНЫЙ ТЕКСТ: "{original_query}"
        УТОЧНЕННЫЙ ТЕХНИЧЕСКИЙ ЗАПРОС: "{rewritten_query}"

        КОНТЕКСТ (ПРИМЕРЫ):
        {context}

        Верни ТОЛЬКО JSON:
        {{
            "original_query": "{original_query}",
            "clarified_query": "{rewritten_query}",
            "final_filters": {{
                "orbitType": "string",
                "mass": "string",
                "coverage": "string",
                "status": "string",
                "formFactor": "string"
            }}
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"error": "failed", "original_query": original_query, "final_filters": {}}

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
        #print(rewritten)
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
    response = pipeline.rag_inference("Дай маленький спутник с орбиты действующий")
    #print(json.dumps(response, ensure_ascii=False, indent=2))
    print(response)


if __name__ == "__main__":
    main()