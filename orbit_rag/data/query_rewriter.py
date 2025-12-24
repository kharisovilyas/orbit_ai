import requests
import json
import openai
import os

# ========================= НАСТРОЙКИ =========================


# 2. Выберите модель. Попробуйте начать с этой:
GROQ_MODEL = "llama-3.3-70b-versatile"  # Быстрая и мощная модель[citation:4]

# 3. базовый URL Groq (совместим с OpenAI)
import openai
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    #api_key=GROQ_API_KEY,
    api_key=os.getenv("GROQ_API_KEY", "")
)
# ===========================================================

def rewrite_query(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content":
                    "Ты помощник для перефразирования запросов о спутниках. Перефразируй запрос так, чтобы он максимально точно передавал задачу выбора спутника. Уточни: тип орбиты, массу, форм-фактор, статус, покрытие.Не добавляй новые данные — только переформулируй существующие.Не придумывай значения, которых нет в исходном запросе.Сохрани исходный смысл полностью."},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка Groq API: {e}")
        return query  # Возвращаем оригинал в случае ошибки
# ===========================================================

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

    print("Тестирование перефразирования через api llm\n")
    print("=" * 80)

    for i, q in enumerate(test_queries, 1):
        print(f"{i}. Исходный: {q}")
        rewritten = rewrite_query(q)
        print(f"   Перефразированный: {rewritten}")
        print("-" * 80)