import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer

# --- НАСТРОЙКИ ---
# Определяем пути динамически относительно текущего файла
BASE_DIR = Path(__file__).resolve().parent.parent  # Это папка orbit_nlu
DATA_FILE = BASE_DIR / "data" / "prompts.jsonl"
IMG_OUTPUT_DIR = BASE_DIR / "EDA"  # Куда сохранять графики

# Для токенизации (можно заменить на путь к Llama-3, если есть доступ и веса)
# Используем distilbert как легкую альтернативу для подсчета токенов,
# так как токенайзеры Llama требуют логина в HuggingFace.
MODEL_NAME = "distilbert-base-multilingual-cased" 

def load_data(filepath):
    """Загрузка JSONL файла в DataFrame с разворачиванием фильтров."""
    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Создаем базовый DF
    df = pd.DataFrame(data)
    
    # Разворачиваем (normalize) вложенный словарь 'filters' в отдельные колонки
    filters_df = pd.json_normalize(df['filters'])
    
    # Объединяем промпты и фильтры
    result_df = pd.concat([df['prompt'], filters_df], axis=1)
    
    # Превращаем пустые строки в NaN для корректной статистики
    result_df = result_df.replace(r'^\s*$', None, regex=True)
    
    return result_df

def analyze_sparsity(df):
    """Анализ заполненности слотов (Frequency of filters)."""
    # Считаем непустые значения (исключая prompt)
    slot_counts = df.drop(columns=['prompt']).count().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=slot_counts.index, y=slot_counts.values, hue=slot_counts.index, palette="viridis", legend=False)
    plt.title('Частота заполнения фильтров (Sparsity Check)')
    plt.ylabel('Количество заполненных примеров')
    plt.xlabel('Название слота')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMG_OUTPUT_DIR / "sparsity_analysis.png")
    print(f"[INFO] График заполненности сохранен в {IMG_OUTPUT_DIR}")
    plt.show()

def analyze_tokens(df):
    """Анализ распределения токенов (Token Distribution)."""
    print(f"[INFO] Загрузка токенизатора {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Считаем токены для промптов
        df['prompt_tokens'] = df['prompt'].apply(lambda x: len(tokenizer.encode(x)))
        
        plt.figure(figsize=(10, 5))
        sns.histplot(df['prompt_tokens'], kde=True, color='blue', bins=20)
        plt.title(f'Распределение длины запросов (в токенах {MODEL_NAME})')
        plt.xlabel('Количество токенов')
        plt.ylabel('Частота')
        plt.savefig(IMG_OUTPUT_DIR / "token_distribution.png")
        print(f"[INFO] График токенов сохранен в {IMG_OUTPUT_DIR}")
        plt.show()
        
        print(f"Средняя длина запроса: {df['prompt_tokens'].mean():.2f} токенов")
        print(f"Макс. длина запроса: {df['prompt_tokens'].max()} токенов")
        
    except Exception as e:
        print(f"[WARN] Не удалось загрузить токенизатор: {e}")
        # Фоллбэк: считаем слова
        df['word_count'] = df['prompt'].apply(lambda x: len(x.split()))
        print(f"Средняя длина (в словах): {df['word_count'].mean():.2f}")

def analyze_values(df):
    """Анализ дисбаланса значений внутри классов."""
    # Пример для orbitType и status
    target_cols = ['orbitType', 'status']
    
    for col in target_cols:
        if col in df.columns:
            val_counts = df[col].value_counts()
            if not val_counts.empty:
                print(f"\n--- Распределение значений для {col} ---")
                print(val_counts)
                
                plt.figure(figsize=(8, 4))
                sns.barplot(x=val_counts.index, y=val_counts.values, hue=val_counts.index, palette="magma", legend=False)
                plt.title(f'Распределение значений: {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

def main():
    print("--- ЗАПУСК АУДИТА ДАТАСЕТА (ЛАБ 1) ---")
    print(f"Чтение данных из: {DATA_FILE}")
    
    try:
        df = load_data(DATA_FILE)
        print(f"Успешно загружено {len(df)} записей.")
        print("-" * 30)
        
        # 1. Анализ частоты фильтров
        analyze_sparsity(df)
        
        # 2. Анализ токенов
        analyze_tokens(df)
        
        # 3. Анализ значений
        analyze_values(df)
        
        print("\n[SUCCESS] Анализ завершен.")
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Проверьте, что файл prompts.jsonl существует в папке data.")

if __name__ == "__main__":
    main()