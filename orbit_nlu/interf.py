import json
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- НАСТРОЙКИ ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # Используем Instruct версию, она умнее
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "prompts.jsonl" # Берем оригинальные данные для теста

# Системный промпт, чтобы объяснить модели задачу
SYSTEM_PROMPT = """You are an AI assistant for aerospace engineers. 
Extract structured information from the user query into a JSON object.
Use the following keys: orbitType, coverage, altitude, mass, status, formFactor, number.
If a value is missing, return an empty string "".
Example Output: {"orbitType": "LEO", "mass": "50", ...}
Only return the JSON."""

def load_model():
    print(f"Загрузка модели {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Загружаем в 4 бита для экономии памяти (если есть GPU), иначе убери load_in_4bit
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            # load_in_4bit=True  # Раскомментируй, если мало видеопамяти и стоят bitsandbytes
        )
    except Exception as e:
        print(f"Ошибка GPU, пробуем CPU (будет медленно): {e}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    return tokenizer, model

def predict(model, tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids, 
        max_new_tokens=128, 
        do_sample=False, # Жадная генерация для воспроизводимости
        temperature=0.0
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

def calculate_metrics(predictions, ground_truth):
    valid_json = 0
    exact_matches = 0
    
    y_true_slots = []
    y_pred_slots = []
    
    for pred_str, truth in zip(predictions, ground_truth):
        true_filters = truth.get("filters", {})
        
        # Попытка парсинга JSON
        try:
            # Иногда модель пишет лишний текст, пробуем найти первую { и последнюю }
            start = pred_str.find('{')
            end = pred_str.rfind('}') + 1
            if start != -1 and end != -1:
                clean_json = pred_str[start:end]
                pred_json = json.loads(clean_json)
                valid_json += 1
            else:
                pred_json = {}
        except:
            pred_json = {}

        # Сравнение (Exact Match)
        # Нормализуем ключи (оставляем только те, что есть в эталоне)
        normalized_pred = {k: str(pred_json.get(k, "")).strip() for k in true_filters.keys()}
        normalized_true = {k: str(v).strip() for k, v in true_filters.items()}
        
        if normalized_pred == normalized_true:
            exact_matches += 1
            
        # Для F1
        for k in true_filters.keys():
            y_true_slots.append(normalized_true[k])
            y_pred_slots.append(normalized_pred[k])

    total = len(ground_truth)
    return {
        "JSON Validity %": (valid_json / total) * 100,
        "Exact Match %": (exact_matches / total) * 100,
        "Slot-F1 Score": f1_score(y_true_slots, y_pred_slots, average='macro', zero_division=0)
    }

def main():
    tokenizer, model = load_model()
    
    # Загружаем данные и берем 10 последних примеров для теста
    data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    test_set = data[-10:] 
    print(f"\n--- ЗАПУСК BASELINE НА {len(test_set)} ПРИМЕРАХ ---")
    
    preds = []
    for item in test_set:
        query = item['prompt']
        print(f"\nЗапрос: {query}")
        
        output = predict(model, tokenizer, query)
        preds.append(output)
        
        print(f"Ответ модели: {output}")
        print("-" * 20)

    metrics = calculate_metrics(preds, test_set)
    
    print("\n" + "="*40)
    print("РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ №1 (BASELINE)")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    
    # Сохраняем метрики в файл
    with open(BASE_DIR / "baseline_metrics.txt", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()