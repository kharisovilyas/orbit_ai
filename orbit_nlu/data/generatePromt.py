import json
import random
import re
import sys
from typing import Dict, Any, List, Optional, Tuple

import pymorphy2

class DatasetGenerator:
    """
    Класс для генерации датасета промптов для поиска спутников.
    Генерирует осмысленные и грамматически корректные запросы на основе
    предопределенных шаблонов и данных, с гарантированным минимальным
    количеством фильтров в каждом промпте.
    """

    def __init__(self, min_filters: int = 3):
        self.morph = pymorphy2.MorphAnalyzer()
        # Гарантируем, что каждый промпт будет содержать не менее 3 фильтров
        self.MIN_FILTERS = min_filters
        self.ALL_FILTER_KEYS = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "number"]

        # --- Структурированные данные для генерации ---
        self.DATA = {
            "coverage": {
                "Россия": ["российский регион", "территория РФ"],
                "Арктика": ["арктический регион", "северный полюс"],
                "Африка": ["африканский континент"],
                "Китай": ["китайский регион", "территория КНР"],
                "Европа": ["европейский регион", "территория ЕС"],
                "Южная Америка": ["южноамериканский континент"],
            },
            "altitude": {
                "<1000 км": ["<1000 км", "менее 1000 километров", "ниже 1000 км"],
                "500-800 км": ["500-800 км", "от 500 до 800 километров", "высота 500-800 км"],
                "~800 км": ["~800 км", "около 800 километров", "приблизительно 800 км"],
                "~36000 км": ["~36000 км", "около 36000 километров", "высота ~36000 км"],
                "2000-20000 км": ["2000-20000 км", "в диапазоне 2000-20000 км"],
                "400-40000 км": ["400-40000 км", "с перигеем ~400 км и апогеем ~40000 км"],
            },
            "orbitType": {
                "LEO": {"nom": "низкая околоземная орбита", "acc": "низкую околоземную орбиту", "prep": "на низкой околоземной орбите"},
                "MEO": {"nom": "средняя околоземная орбита", "acc": "среднюю околоземную орбиту", "prep": "на средней околоземной орбите"},
                "GEO": {"nom": "геостационарная орбита", "acc": "геостационарную орбиту", "prep": "на геостационарной орбите"},
                "SSO": {"nom": "солнечно-синхронная орбита", "acc": "солнечно-синхронную орбиту", "prep": "на солнечно-синхронной орбите"},
                "Molniya": {"nom": "орбита Молния", "acc": "орбиту Молния", "prep": "на орбите Молния"},
                "HEO": {"nom": "высокая эллиптическая орбита", "acc": "высокую эллиптическую орбиту", "prep": "на высокой эллиптической орбите"},
            },
            "status": {
                "активен": ["активный", "работающий", "функционирующий", "в рабочем состоянии"],
                "неактивен": ["неактивный", "неработающий", "вышедший из строя"],
            },
            "formFactor": {
                "1U": ["1U", "кубсат 1U"], "3U": ["3U", "кубсат 3U"],
                "6U": ["6U", "кубсат 6U"], "12U": ["12U", "кубсат 12U"],
            },
            "request_verb": [
                "Подбери", "Найди", "Выведи", "Покажи", "Ищи", "Предоставь", "Дай", "Предложи", "Подскажи"
            ]
        }

        # --- Правила и ограничения ---
        self.VALIDITY_RULES = {
            "mass_for_formFactor": {"1U": (1, 2), "3U": (4, 7), "6U": (8, 12), "12U": (20, 30)},
            "altitude_for_orbit": {
                "LEO": ["<1000 км", "500-800 км", "~800 км"],
                "MEO": ["2000-20000 км"],
                "GEO": ["~36000 км"],
                "SSO": ["500-800 км", "~800 км"],
                "Molniya": ["400-40000 км"],
                "HEO": ["400-40000 км"],
            },
            "invalid_combinations": [
                {"orbitType": "GEO", "coverage": "Арктика"},
                {"orbitType": "Molniya", "coverage": "Африка"},
            ]
        }

        # --- Шаблоны ---
        # Формат: (текст_шаблона, [необходимые_фильтры])
        self.TEMPLATES = [
            # --- Шаблоны с 3 фильтрами ---
            ("{request_verb} {satellite} для мониторинга {coverage} с высоты {altitude}.", ["number", "coverage", "altitude"]),
            ("{request_verb} {satellite} с массой {mass} и статусом {status}.", ["number", "mass", "status"]),
            ("Нужен {satellite} на {orbitType} для покрытия {coverage}.", ["number", "orbitType", "coverage"]),
            ("Требуется {satellite} для {coverage} на {orbitType}.", ["number", "coverage", "orbitType"]),
            ("Существуют ли {satellite} с массой более {mass} и статусом {status}?", ["number", "mass", "status"]),
            ("Для высоты {altitude} подбери {satellite}, которые видят {coverage}.", ["number", "altitude", "coverage"]),
            ("Какие есть {satellite} с форм-фактором {formFactor} для {coverage}?", ["number", "formFactor", "coverage"]),

            # --- Шаблоны с 4 фильтрами ---
            ("{request_verb} {status} {satellite} на {orbitType} для {coverage}.", ["number", "status", "orbitType", "coverage"]),
            ("Мне нужен {satellite} с форм-фактором {formFactor}, массой {mass} для {coverage}.", ["number", "formFactor", "mass", "coverage"]),
            ("Покажи {satellite} на {orbitType} с высотой {altitude} и статусом {status}.", ["number", "orbitType", "altitude", "status"]),
            
            # --- Шаблон с 5 фильтрами ---
            ("Какой {status} {satellite} с форм-фактором {formFactor} можно использовать для {coverage} на {orbitType}?", ["number", "status", "formFactor", "coverage", "orbitType"]),
        ]

    def _get_random_key(self, data_key: str) -> str:
        return random.choice(list(self.DATA[data_key].keys()))

    def _get_random_value(self, data_key: str, key: str) -> str:
        return random.choice(self.DATA[data_key][key])

    def _get_random_satellite_number(self) -> int:
        return 1 if random.random() < 0.8 else random.randint(2, 10)

    def _incline_word(self, word: str, case: str) -> str:
        """Склоняет слово или фразу в нужный падеж."""
        p = self.morph.parse(word.split(' ')[-1])[0]
        inflected = p.inflect({case})
        if inflected:
            return word.rsplit(' ', 1)[0] + ' ' + inflected.word if ' ' in word else inflected.word
        return word

    def _incline_adjective(self, adj: str, number: int, case: str) -> str:
        """Согласует прилагательное с числом и падежом."""
        p = self.morph.parse(adj)[0]
        grammemes = {case}
        grammemes.add('plur' if number > 1 else 'sing')
        inflected = p.inflect(grammemes)
        return inflected.word if inflected else adj

    def _get_satellite_text(self, number: int, case: str) -> str:
        """Возвращает корректную форму 'N спутник' в нужном падеже."""
        word = "спутник"
        parsed_word = self.morph.parse(word)[0]
        return f"{number} {parsed_word.make_agree_with_number(number).inflect({case}).word}"

    def _is_combination_valid(self, filters: Dict[str, Any]) -> bool:
        """Проверяет, является ли комбинация фильтров физически и логически возможной."""
        for rule in self.VALIDITY_RULES["invalid_combinations"]:
            if all(filters.get(key) == val for key, val in rule.items()):
                return False
        return True

    def _postprocess_prompt(self, prompt: str, filters: Dict[str, Any]) -> str:
        """Финальная обработка промпта для улучшения читаемости и грамматики."""
        number = int(filters.get("number", 1))
        if "которые" in prompt and number == 1:
            prompt = prompt.replace("которые", "который")
            prompt = re.sub(r'(\w+ют)\b', lambda m: m.group(1)[:-2] + 'ет', prompt)
        
        prompt = prompt.strip().capitalize()
        prompt = re.sub(r'\s+', ' ', prompt)
        
        if not re.search(r'[.?!]$', prompt):
            prompt += "?" if prompt.lower().startswith(("какие", "существуют ли")) else "."

        return prompt

    def generate_one(self) -> Optional[Dict[str, Any]]:
        """Генерирует одну запись (промпт + фильтры)."""
        
        # 1. Выбираем только те шаблоны, которые удовлетворяют требованию по мин. числу фильтров
        valid_templates = [t for t in self.TEMPLATES if len(t[1]) >= self.MIN_FILTERS]
        if not valid_templates:
            raise ValueError("Нет шаблонов, удовлетворяющих требованию по минимальному количеству фильтров.")
        
        template, required_filters = random.choice(valid_templates)
        
        # 2. Генерируем значения для этих фильтров
        generated_filters = {}
        
        if "number" in required_filters:
            generated_filters["number"] = str(self._get_random_satellite_number())

        if "orbitType" in required_filters:
            orbit = self._get_random_key("orbitType")
            generated_filters["orbitType"] = orbit
            if "altitude" in required_filters:
                alt_key = random.choice(self.VALIDITY_RULES["altitude_for_orbit"][orbit])
                generated_filters["altitude"] = alt_key
        elif "altitude" in required_filters:
             generated_filters["altitude"] = self._get_random_key("altitude")

        if "formFactor" in required_filters:
            ff = self._get_random_key("formFactor")
            generated_filters["formFactor"] = ff
            if "mass" in required_filters:
                min_m, max_m = self.VALIDITY_RULES["mass_for_formFactor"][ff]
                generated_filters["mass"] = str(random.randint(min_m, max_m))
        elif "mass" in required_filters:
            generated_filters["mass"] = str(random.randint(5, 50))
        
        for key in ["coverage", "status"]:
            if key in required_filters:
                 generated_filters[key] = self._get_random_key(key)

        # 3. Проверяем комбинацию на валидность
        if not self._is_combination_valid(generated_filters):
            return None

        # 4. Собираем промпт
        prompt = template
        text_values = {}
        
        case_map = {'для': 'gent', 'с': 'ablt', 'на': 'loct', 'в': 'loct', 'по': 'datv'}
        
        for key, value in generated_filters.items():
            case = 'nomn'
            match = re.search(r'(\b\w+\b)\s+{\s*' + key + r'\s*}', prompt)
            if match and match.group(1).lower() in case_map:
                case = case_map[match.group(1).lower()]

            if key == 'number':
                text_values['satellite'] = self._get_satellite_text(int(value), 'accs')
            elif key == 'coverage':
                base_word = random.choice([value] + self.DATA['coverage'][value])
                text_values['coverage'] = self._incline_word(base_word, case)
            elif key == 'orbitType':
                text_values['orbitType'] = self.DATA['orbitType'][value]['prep']
            elif key == 'altitude':
                text_values['altitude'] = self._get_random_value('altitude', value)
            elif key == 'mass':
                text_values['mass'] = f"{value} кг"
            elif key == 'formFactor':
                text_values['formFactor'] = self._get_random_value('formFactor', value)
            elif key == 'status':
                base_status = self._get_random_value('status', value)
                num = int(generated_filters.get("number", 1))
                text_values['status'] = self._incline_adjective(base_status.split()[0], num, 'nomn')
        
        text_values['request_verb'] = random.choice(self.DATA['request_verb'])
        
        prompt = prompt.format(**text_values)
        
        # 5. Пост-обработка
        prompt = self._postprocess_prompt(prompt, generated_filters)

        # 6. Формирование финального объекта
        final_filters = {key: "" for key in self.ALL_FILTER_KEYS}
        final_filters.update(generated_filters)

        return {"prompt": prompt, "filters": final_filters}

def main():
    try:
        count = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    except (ValueError, IndexError):
        count = 500
        print(f"Неверный аргумент. Будет сгенерировано {count} записей.")

    # Создаем экземпляр генератора с требованием минимум 3 фильтра
    generator = DatasetGenerator(min_filters=3)
    seen_prompts = set()
    dataset = []
    
    max_attempts = count * 20
    attempts = 0

    print(f"Генерация {count} записей (минимум 3 фильтра в каждой)...")
    
    while len(dataset) < count and attempts < max_attempts:
        attempts += 1
        record = generator.generate_one()
        
        if record:
            prompt_key = record['prompt'].lower()
            if prompt_key not in seen_prompts:
                seen_prompts.add(prompt_key)
                dataset.append(record)
                
                progress = len(dataset) / count
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f"\rПрогресс: |{bar}| {len(dataset)}/{count} ({progress:.0%})")
                sys.stdout.flush()

    print("\nГенерация завершена.")

    output_filename = "prompts.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Сохранено {len(dataset)} уникальных записей в файл '{output_filename}'.")
    print(f"Всего попыток: {attempts}.")

if __name__ == "__main__":
    main()