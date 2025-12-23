import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report

# --- НАСТРОЙКИ ---
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" # Или твой путь
ADAPTER_PATH = "outputs/orbit-nlu-lora" # Путь к папке с весами ЛУЧШЕГО эксперимента
DATA_FILE = "orbit_nlu/data/prompts.jsonl" # Тестовые данные

def get_errors():
    # 1. Загрузка модели (как обычно)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    # 2. Прогон данных
    errors = []
    
    # Загружаем данные
    with open(DATA_FILE, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    test_data = data[-50:] # Берем 50 примеров для анализа
    
    print("Анализ ошибок...")
    for item in test_data:
        prompt = f"Extract JSON.\n\nQuery: {item['prompt']}\n\nResponse:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
        res = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Парсинг
        try:
            start = res.find('{')
            end = res.rfind('}') + 1
            pred_json = json.loads(res[start:end])
            
            # Очистка значений
            pred_clean = {k: str(v).strip() for k, v in pred_json.items()}
            true_clean = {k: str(v).strip() for k, v in item['filters'].items()}
            
            # Сравнение
            if pred_clean != true_clean:
                # Нашли ошибку!
                diff = {}
                for k in true_clean:
                    if pred_clean.get(k) != true_clean.get(k):
                        diff[k] = {"Expected": true_clean.get(k), "Got": pred_clean.get(k)}
                
                errors.append({
                    "query": item['prompt'],
                    "diff": diff
                })
        except:
            pass # Игнорируем ошибки парсинга, их у нас нет

    # 3. Сохранение отчета
    with open("orbit_nlu/error_report.txt", "w", encoding="utf-8") as f:
        f.write(f"ВСЕГО ОШИБОК НА 50 ПРИМЕРАХ: {len(errors)}\n")
        f.write("="*30 + "\n\n")
        for err in errors:
            f.write(f"Q: {err['query']}\n")
            f.write(f"ERR: {json.dumps(err['diff'], ensure_ascii=False)}\n")
            f.write("-" * 20 + "\n")
            
    print(f"Найдено {len(errors)} ошибок. Подробности в error_report.txt")

if __name__ == "__main__":
    get_errors()