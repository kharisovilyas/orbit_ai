import faiss
import numpy as np
import os

def build_faiss_index(vectors_path: str, output_path: str) -> None:
    """
    Строит FAISS-индекс IndexFlatIP из векторов и сохраняет его.
    """
    vectors = np.load(vectors_path)
    
    if vectors.shape[1] != 768:
      raise ValueError(f"Ожидалась размерность 768, получено {vectors.shape[1]}")
    
    if vectors.dtype != np.float32:
      vectors = vectors.astype(np.float32)
    
    index = faiss.IndexFlatIP(768)
    
    index.add(vectors)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    faiss.write_index(index, output_path)
    
    print(f"Индекс успешно создан и сохранён в {output_path}")
    print(f"Количество векторов в индексе: {index.ntotal}")


def load_index(index_path: str) -> faiss.Index:
    """
    Загружает готовый FAISS-индекс из файла.
    
    """
    if not os.path.exists(index_path):
      raise FileNotFoundError(f"Индекс не найден: {index_path}")
    
    index = faiss.read_index(index_path)
    print(f"Индекс загружен из {index_path}, векторов: {index.ntotal}")
    return index


#Пример использования (для тестирования в лабе)
if __name__ == "__main__":
    build_faiss_index('../data/vectors.npy', '../data/index.faiss')
    idx = load_index('../data/index.faiss')