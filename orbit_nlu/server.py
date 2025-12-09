#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import yaml
import torch
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Orbit-NLU Parser")

# Pydantic-модели
class DtoFilters(BaseModel):
    coverage: str = ""
    altitude: str = ""
    orbitType: str = ""
    status: str = ""
    formFactor: str = ""
    mass: str = ""
    scale: str = ""
    tleDate: str = ""
    numberOfSatellites: str = ""


class ParseResponse(BaseModel):
    filters: DtoFilters
    valid: bool
    timestamp: int
    raw: str


class ParseRequest(BaseModel):
    text: str = Field(..., description="Входной запрос пользователя")


def load_cfg():
    """Чтение конфигурационного файла YAML."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка при чтении config.yaml: {e}")
        raise


def build_prompt(system_prompt: str, user: str) -> str:
    """Формирование промпта для модели."""
    template = cfg.get("prompt_template", "{system_prompt}\n\n**Запрос:** {user}\n\n**Ответ:**")
    return template.format(system_prompt=system_prompt, user=user)


def fix_json(raw_text: str) -> dict:
    """Извлекает и парсит первый найденный JSON-объект из строки."""
    start_index = raw_text.find('{')
    end_index = raw_text.rfind('}')
    if start_index == -1 or end_index == -1 or end_index < start_index:
        logger.warning(f"Не удалось найти JSON-объект в строке: {raw_text}")
        return {}
    json_str = raw_text[start_index: end_index + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON: {e}. Содержимое: {json_str}")
        return {}


def sanitize_filters(filters: dict) -> dict:
    """Очищает и нормализует значения фильтров."""
    if not isinstance(filters, dict):
        return {}
    sanitized = {}
    keys_to_clean = ["orbitType", "coverage", "altitude", "status", "scale", "tleDate", "formFactor"]
    for key, value in filters.items():
        if key in keys_to_clean and str(value) == "0":
            sanitized[key] = ""
        elif key == "status" and value == "актив":
            sanitized[key] = "активен"
        else:
            sanitized[key] = value
    return sanitized


# Загрузка конфигурации
cfg = load_cfg()
base = cfg.get("model_name")
out = cfg.get("output_dir")
if not base:
    raise RuntimeError("В config.yaml не указан ключ 'model_name'")
if not out:
    raise RuntimeError("В config.yaml не указан ключ 'output_dir'")

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
if tokenizer.pad_token is None:
    # Используем eos_token как pad_token, если pad не задан
    tokenizer.pad_token = tokenizer.eos_token
logger.info(f"Токенизатор загружен: {base}")

# Конфигурация квантизации (bitsandbytes)
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Загрузка модели
logger.info("Начинаем загрузку модели...")
model = AutoModelForCausalLM.from_pretrained(
    base,
    device_map="auto",
    quantization_config=quant,
    dtype=torch.bfloat16,  # вместо deprecated torch_dtype
)

# Подключаем LoRA-адаптер
model = PeftModel.from_pretrained(model, out)
model.eval()
model.config.pad_token_id = tokenizer.pad_token_id
logger.info(f"Модель с LoRA-адаптером загружена: {out}")


@app.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest):
    try:
        logger.info(f"Получен запрос: {req.text}")
        prompt = build_prompt(cfg.get("system_prompt", ""), req.text)

        # Токенизация и перенос входа на устройство модели
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        raw = text.split("**Ответ:**")[-1].strip()

        # Парсим и нормализуем
        raw_filters_dict = sanitize_filters(fix_json(raw))
        is_valid = bool(raw_filters_dict)

        if is_valid:
            final_filters = DtoFilters(**raw_filters_dict)
        else:
            final_filters = DtoFilters()

        current_timestamp = int(time.time())

        response_data = {
            "filters": final_filters,
            "valid": is_valid,
            "timestamp": current_timestamp,
            "raw": raw,
        }

        logger.info(f"Сгенерированный ответ: {response_data}")
        return response_data

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
