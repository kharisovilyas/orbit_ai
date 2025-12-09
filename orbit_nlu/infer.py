#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Инференс: загрузка базовой модели + LoRA-адаптера и генерация JSON из промта.
Оптимизировано для астрономических и баллистических данных малых и больших КА.
Совместимо с transformers>=4.44.0, peft>=0.12.0, bitsandbytes>=0.43.0, pyyaml>=6.0.1.
"""

import json
import yaml
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Проверка версий библиотек
try:
    import transformers
    import peft
    import bitsandbytes
    import yaml
    from importlib.metadata import version

    assert transformers.__version__ >= "4.44.0", "Требуется transformers>=4.44.0"
    assert peft.__version__ >= "0.12.0", "Требуется peft>=0.12.0"
    assert bitsandbytes.__version__ >= "0.43.0", "Требуется bitsandbytes>=0.43.0"
except ImportError as e:
    raise ImportError(f"Ошибка импорта библиотеки: {e}. Убедитесь, что установлены все зависимости.")
except AssertionError as e:
    raise AssertionError(f"Несовместимая версия библиотеки: {e}")


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_cfg():
    """Чтение конфигурационного файла YAML."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка при чтении config.yaml: {e}")
        raise

def build_prompt(system_prompt, user):
    """Формирование промта для модели."""
    return f"{system_prompt}\n\n**Запрос:** {user}\n\n**Ответ:**"

def fix_json(raw: str) -> dict:
    """Попытка исправить некорректный JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Некорректный JSON: {e}. Пытаемся исправить...")
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != 0:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                logger.error("Не удалось исправить JSON.")
                return {}
        return {}

def main(user_text="Подбери низкую группировку, видит Россию, масса до 10 кг"):
    try:
        # Загрузка конфигурации
        cfg = load_cfg()
        logger.info("Конфигурация успешно загружена.")

        base = cfg["model_name"]
        out = cfg["output_dir"]

        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Токенизатор загружен: {base}")

        # Загрузка модели с квантованием
        quant = BitsAndBytesConfig(
            load_in_8bit=cfg.get("load_8bit", True),
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base,
            device_map="auto",
            quantization_config=quant,
            torch_dtype=torch.bfloat16
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, out)
        model.eval()
        logger.info(f"Модель с LoRA-адаптером загружена: {out}")

        # Формирование промта
        prompt = build_prompt(cfg["system_prompt"], user_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Генерация ответа
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                top_k=50
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # Извлечение JSON после **Ответ:**
        result = text.split("**Ответ:**")[-1].strip()

        # Парсинг JSON
        filters = fix_json(result)
        logger.info(f"Сгенерированные фильтры: {filters}")

        print(json.dumps(filters, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"Ошибка при инференсе: {e}")
        raise

if __name__ == "__main__":
    main()