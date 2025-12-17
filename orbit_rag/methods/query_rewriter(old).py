import requests
import json

# ========================= НАСТРОЙКИ =========================
GEMINI_API_KEY = "AIzaSyDwP0zc9y8bazLSyzNxs2l9ZeiWk37dir0"  # <-- вставь свой ключ
# Актуальная модель на декабрь 2025 (бесплатная и быстрая)
GEMINI_MODEL = "gemini-2.5-flash"  # или "gemini-2.0-flash"

if not GEMINI_API_KEY:
    raise ValueError("Задай переменную окружения GEMINI_API_KEY с новым ключом!")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)
# ===========================================================

REWRITE_PROMPT = """
Перефразируй запрос так, чтобы он максимально точно передавал задачу выбора спутника. 
Уточни: тип орбиты, массу, форм-фактор, статус, покрытие.
Не добавляй новые данные — только переформулируй существующие.
Не придумывай значения, которых нет в исходном запросе.
Сохрани исходный смысл полностью.

Исходный запрос:
{query}

Перефразированный запрос:
"""

def rewrite_query(query: str) -> str:
    if not query.strip():
        return query

    full_prompt = REWRITE_PROMPT.format(query=query.strip())

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 150
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return query

        text_parts = candidates[0].get("content", {}).get("parts", [])
        rewritten = "".join(part.get("text", "") for part in text_parts).strip()

        if rewritten.lower().startswith("перефразированный запрос"):
            rewritten = rewritten.split(":", 1)[1].strip()

        return rewritten or query

    except requests.exceptions.HTTPError as e:
        print(f"Ошибка Gemini API ({e.response.status_code}): {e.response.text}")
        return query
    except Exception as e:
        print(f"Ошибка при обращении к Gemini API: {e}")
        return query


# ========================= ТЕСТ =========================
if __name__ == "__main__":

    test_queries = [
        "Выведи 4 спутник с массой 42 кг и статусом неработающие.",
        "Найди спутники на низкой орбите с массой до 100 кг",
        "Покажи активные CubeSat'ы на геостационарной орбите",
        "Спутник с глобальным покрытием и массой больше тонны",
        "Неработающий спутник форм-фактора 3U",
        "Выведи спутник с покрытием Европы и статусом активен"
    ]

    print("Тестирование перефразирования через Google Gemini\n")
    print("=" * 80)

    for i, q in enumerate(test_queries, 1):
        print(f"{i}. Исходный: {q}")
        rewritten = rewrite_query(q)
        print(f"   Перефразированный: {rewritten}")
        print("-" * 80)