#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_experiments2.py
Оркестратор для запуска серии экспериментов по дообучению.
Формирует сетку параметров, запускает train2.py и сохраняет результаты в CSV.
"""

import subprocess
import csv
import json
import os
import time

# --- КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТОВ ---
# Базовые параметры (Baseline)
BASE_CONFIG = {
    "lora_r": 16,
    "quantization": "nf4",
    "learning_rate": 1e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0
}

EXPERIMENTS = []

# 1. Baseline
EXPERIMENTS.append({**BASE_CONFIG, "note": "Baseline"})

# 2. LoRA Rank variations
EXPERIMENTS.append({**BASE_CONFIG, "lora_r": 4, "note": "Low Rank"})
EXPERIMENTS.append({**BASE_CONFIG, "lora_r": 32, "note": "High Rank"})
EXPERIMENTS.append({**BASE_CONFIG, "lora_r": 64, "note": "Very High Rank"})

# 3. Quantization variations
EXPERIMENTS.append({**BASE_CONFIG, "quantization": "fp16", "note": "No Quant (FP16)"})

# 4. Learning Rate variations
EXPERIMENTS.append({**BASE_CONFIG, "learning_rate": 2e-4, "note": "High LR"})
EXPERIMENTS.append({**BASE_CONFIG, "learning_rate": 5e-5, "note": "Low LR"})

# 5. Warmup variations
EXPERIMENTS.append({**BASE_CONFIG, "warmup_ratio": 0.1, "note": "Medium Warmup"})
EXPERIMENTS.append({**BASE_CONFIG, "warmup_ratio": 0.15, "note": "High Warmup"})

# 6. Weight Decay variations
EXPERIMENTS.append({**BASE_CONFIG, "weight_decay": 0.1, "note": "Medium WD"})
EXPERIMENTS.append({**BASE_CONFIG, "weight_decay": 0.2, "note": "High WD"})

# Дополнительный тест: r=8
EXPERIMENTS.append({**BASE_CONFIG, "lora_r": 8, "note": "Rank 8"})

OUTPUT_CSV = "ft_hparams_results2.csv"
DATASET_FILE = "data/prompts2.jsonl" # Убедись, что этот файл существует!

def run_experiment(cfg, idx):
    print(f"\n>>> Запуск эксперимента {idx+1}/{len(EXPERIMENTS)}: {cfg['note']}")
    print(f"    Params: r={cfg['lora_r']}, q={cfg['quantization']}, lr={cfg['learning_rate']}, warm={cfg['warmup_ratio']}, wd={cfg['weight_decay']}")
    
    output_dir = f"outputs2/exp_{idx}_{cfg['note'].replace(' ', '_')}"
    
    cmd = [
        "python", "train2.py",
        "--lora_r", str(cfg["lora_r"]),
        "--quantization", str(cfg["quantization"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--warmup_ratio", str(cfg["warmup_ratio"]),
        "--weight_decay", str(cfg["weight_decay"]),
        "--output_dir", output_dir,
        "--dataset_path", DATASET_FILE
    ]
    
    try:
        # Запускаем процесс и захватываем вывод
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout
        
        # Парсим результат из stdout (ищем маркер __RESULT_JSON__)
        if "__RESULT_JSON__" in stdout:
            json_str = stdout.split("__RESULT_JSON__")[1]
            metrics = json.loads(json_str)
            return metrics
        else:
            print("ОШИБКА: Не найден JSON с результатами в выводе скрипта.")
            print("Последние строки вывода:")
            print(stdout[-500:])
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"ОШИБКА при выполнении эксперимента {idx}: {e}")
        print(e.stderr)
        return None

def main():
    # Инициализация CSV
    headers = [
        "Config ID", "Note", "LoRA R", "Quantization", "LR", "Warmup", "Weight Decay", 
        "JSON Validity", "Exact Match", "Slot-F1"
    ]
    
    # Проверка наличия датасета
    if not os.path.exists(DATASET_FILE):
        print(f"ВНИМАНИЕ: Файл {DATASET_FILE} не найден. Проверьте путь.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    for i, exp_cfg in enumerate(EXPERIMENTS):
        metrics = run_experiment(exp_cfg, i)
        
        if metrics:
            row = [
                i,
                exp_cfg["note"],
                exp_cfg["lora_r"],
                exp_cfg["quantization"],
                exp_cfg["learning_rate"],
                exp_cfg["warmup_ratio"],
                exp_cfg["weight_decay"],
                f"{metrics['json_validity']:.4f}",
                f"{metrics['exact_match']:.4f}",
                f"{metrics['slot_f1']:.4f}"
            ]
            
            # Пишем в файл сразу (append mode было бы лучше, но здесь перезаписываем список)
            # Чтобы не терять данные при падении, откроем файл на append
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f_append:
                writer_append = csv.writer(f_append)
                writer_append.writerow(row)
                
            print(f"✅ Успех! EM: {metrics['exact_match']:.4f}, F1: {metrics['slot_f1']:.4f}")
        else:
            print("❌ Эксперимент не удался.")
        
        # Небольшая пауза, чтобы GPU остыла/очистилась
        time.sleep(5)

    print(f"\nВсе эксперименты завершены. Результаты в {OUTPUT_CSV}")

if __name__ == "__main__":
    main()