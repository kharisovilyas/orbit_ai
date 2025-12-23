#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_experiments2.py
Оркестратор для запуска серии экспериментов по дообучению.
С визуализацией времени выполнения и прогресс-баром.
"""

import subprocess
import csv
import json
import os
import time
from tqdm import tqdm  # Нужна библиотека tqdm

# --- КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТОВ ---
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
DATASET_FILE = "data/prompts2.jsonl" 

def run_experiment(cfg, idx):
    # Формируем команду
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
        # Запускаем процесс. capture_output=True скрывает вывод train2.py, 
        # чтобы он не ломал наш прогресс-бар, но мы ждем завершения.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout
        
        # Парсим результат
        if "__RESULT_JSON__" in stdout:
            json_str = stdout.split("__RESULT_JSON__")[1]
            metrics = json.loads(json_str)
            return metrics
        else:
            # Если JSON нет, пишем ошибку в отдельный лог, чтобы не ломать бар
            with open("experiment_errors.log", "a") as err_f:
                err_f.write(f"\nError in Exp {idx} ({cfg['note']}):\n")
                err_f.write(stdout[-500:])
            return None
            
    except subprocess.CalledProcessError as e:
        with open("experiment_errors.log", "a") as err_f:
            err_f.write(f"\nCRITICAL Error in Exp {idx}:\n")
            err_f.write(str(e.stderr))
        return None

def main():
    headers = [
        "Config ID", "Note", "LoRA R", "Quantization", "LR", "Warmup", "Weight Decay", 
        "JSON Validity", "Exact Match", "Slot-F1"
    ]
    
    if not os.path.exists(DATASET_FILE):
        print(f"❌ ВНИМАНИЕ: Файл {DATASET_FILE} не найден.")
        return

    # Создаем/очищаем CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print(f"🚀 Запуск серии из {len(EXPERIMENTS)} экспериментов...")
    print(f"📁 Результаты будут в: {OUTPUT_CSV}")
    print("-" * 60)

    # Используем TQDM для отображения общего прогресса и времени
    pbar = tqdm(enumerate(EXPERIMENTS), total=len(EXPERIMENTS), unit="exp", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for i, exp_cfg in pbar:
        # Обновляем описание бара (текущий эксперимент)
        pbar.set_description(f"Exp {i+1}: {exp_cfg['note']}")
        
        metrics = run_experiment(exp_cfg, i)
        
        if metrics:
            row = [
                i, exp_cfg["note"], exp_cfg["lora_r"], exp_cfg["quantization"],
                exp_cfg["learning_rate"], exp_cfg["warmup_ratio"], exp_cfg["weight_decay"],
                f"{metrics['json_validity']:.4f}",
                f"{metrics['exact_match']:.4f}",
                f"{metrics['slot_f1']:.4f}"
            ]
            
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f_append:
                writer_append = csv.writer(f_append)
                writer_append.writerow(row)
            
            # Можно вывести краткий результат рядом с баром, если нужно, 
            # но лучше не спамить, чтобы не ломать TQDM.
        else:
            # Если ошибка - просто идем дальше
            pass

    print("\n" + "="*60)
    print(f"✅ Все эксперименты завершены!")
    print(f"📊 Откройте файл {OUTPUT_CSV} для анализа.")

if __name__ == "__main__":
    main()