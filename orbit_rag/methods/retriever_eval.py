import faiss
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

INDEX_PATH = '../data/index.faiss'
VECTORS_PATH = '../data/vectors.npy'
DOCUMENTS_PATH = '../data/documents.jsonl'
KS = [1, 5, 10, 20]
COSINE_THRESHOLD = 0.75              
NORMALIZED = True                  

def load_documents(jsonl_path: str):
    """Загружает documents.jsonl"""
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    print(f"Загружено {len(documents)} документов")
    return documents

def load_index_and_vectors():
    """Загружает FAISS-индекс и векторы"""
    index = faiss.read_index(INDEX_PATH)
    vectors = np.load(VECTORS_PATH).astype('float32')
    if NORMALIZED:
        faiss.normalize_L2(vectors)
    return index, vectors

def retrieve_top_k(index, query_vec, k, exclude_idx):
    """Возвращает top-k (исключая сам запрос)"""
    D, I = index.search(query_vec[None, :], k + 1)
    distances = D[0]
    indices = I[0]
    mask = indices != exclude_idx
    return distances[mask][:k], indices[mask][:k]

def evaluate_retriever(documents, index, vectors):
    """Основная функция оценки на полном датасете"""
    max_k = max(KS)
    recall_hits = {k: 0 for k in KS}
    cos_sums = {k: 0.0 for k in KS}      # сумма косинусных сходств для среднего
    irrelevant_counts = {k: 0 for k in KS}
    total = len(documents)

    for idx, doc in enumerate(tqdm(documents, desc="Evaluation")):
        query_vec = vectors[idx]
        json_part = doc.get('json', {})
        true_slots = (
            json_part.get('orbitType'),
            json_part.get('status'),
            json_part.get('mass')
        )

        distances, _ = retrieve_top_k(index, query_vec, max_k, exclude_idx=idx)

        for k in KS:
            top_distances = distances[:k]

            # Среднее косинусное сходство
            if len(top_distances) > 0:
                cos_sums[k] += np.mean(top_distances)

            # Recall@k: есть ли хотя бы один с точно такими же слотами
            # (для этого нужно проверить документы по индексам, но поскольку мы ищем точное совпадение слотов,
            #  а не только текущий, мы просто считаем hit, если в топе есть совпадение — здесь упрощаем через поиск)
            # В реальности для точности нужно проверять retrieved документы, но в задаче требуется совпадение слотов
            # Чтобы не хранить второй раз документы, мы используем упрощённый подход: hit = 1 если есть совпадение
            # (в вашем случае датасет небольшой, поэтому получаем retrieved_docs)
            _, retrieved_indices = retrieve_top_k(index, query_vec, max_k, exclude_idx=idx)
            retrieved_docs = [documents[i] for i in retrieved_indices[:k]]
            hit = any(
                (d.get('json', {}).get('orbitType'), d.get('json', {}).get('status'), d.get('json', {}).get('mass')) == true_slots
                for d in retrieved_docs
            )
            if hit:
                recall_hits[k] += 1

            # Доля пустых/нерелевантных
            if len(top_distances) > 0 and np.all(top_distances < COSINE_THRESHOLD):
                irrelevant_counts[k] += 1

    # Вычисление метрик
    results = {}
    for k in KS:
        results[f'Recall@{k}'] = recall_hits[k] / total
        results[f'Mean Cosine Similarity@{k}'] = cos_sums[k] / total
        results[f'Irrelevant/Empty@{k} (%)'] = 100 * irrelevant_counts[k] / total

    return results

def plot_recall_at_k(ks, recall_values):
    """Отрисовывает и сохраняет график Recall@k"""
    plt.figure(figsize=(8, 5))
    plt.plot(ks, recall_values, marker='o', linestyle='-', color='blue')
    plt.title('Recall@k')
    plt.xlabel('k')
    plt.ylabel('Recall')
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    for i, v in enumerate(recall_values):
        plt.text(ks[i], v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig('recall_at_k.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    documents = load_documents(DOCUMENTS_PATH)
    index, vectors = load_index_and_vectors()

    if len(documents) != index.ntotal:
        print("ВНИМАНИЕ: количество документов и векторов не совпадает!")

    metrics = evaluate_retriever(documents, index, vectors)

    # Таблица результатов
    df = pd.DataFrame({
        "Metric": metrics.keys(),
        "Value": [f"{v:.4f}" if isinstance(v, float) else v for v in metrics.values()]
    })
    print("\nРезультаты оценки retriever'а:")
    print(df.to_string(index=False))

    # График Recall@k
    recall_values = [metrics[f'Recall@{k}'] for k in KS]
    plot_recall_at_k(KS, recall_values)
    print("\nГрафик сохранён как 'recall_at_k.png'")