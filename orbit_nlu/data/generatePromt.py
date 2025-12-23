import json
import random
import re
import sys
import pymorphy2
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

class DatasetGenerator:
    """
    Генератор датасета NLU (Balanced Edition).
    75% данных — чистый, грамотный русский язык.
    25% данных — реалистичные ошибки, сленг, небрежность.
    """

    def __init__(self, min_filters: int = 2):
        self.morph = pymorphy2.MorphAnalyzer()
        self.MIN_FILTERS = min_filters
        self.ALL_FILTER_KEYS = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "number"]
        
        # Настройка баланса
        self.CLEAN_RATIO = 0.75  # 75% чистых данных

        self.NUM_TO_TEXT = {
            1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
            6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
            12: "двенадцать"
        }

        # --- ДАННЫЕ ---
        self.DATA = {
            "coverage": [
                ("Россия", "Россия"), ("РФ", "Россия"), ("Российская Федерация", "Россия"), ("территория РФ", "Россия"),
                ("Москва", "Россия"), ("Сибирь", "Россия"), ("Дальний Восток", "Россия"), ("Крым", "Россия"),
                ("США", "США"), ("Америка", "США"), ("Северная Америка", "Северная Америка"),
                ("Китай", "Китай"), ("КНР", "Китай"), ("Поднебесная", "Китай"),
                ("Европа", "Европа"), ("ЕС", "Европа"), ("Евросоюз", "Европа"), ("страны ЕС", "Европа"),
                ("Африка", "Африка"), ("африканский континент", "Африка"),
                ("Южная Америка", "Южная Америка"), ("Бразилия", "Южная Америка"),
                ("Азия", "Азия"), ("Индия", "Индия"), ("Ближний Восток", "Ближний Восток"),
                ("Арктика", "Арктика"), ("Северный полюс", "Арктика"), ("Севморпуть", "Арктика"),
                ("Антарктида", "Антарктида"), ("Южный полюс", "Антарктида"),
                ("Тихий океан", "Тихий океан"), ("Атлантика", "Атлантический океан"),
                ("Экватор", "Экватор"), ("экваториальная зона", "Экватор"), ("Global", "Global"),
            ],
            
            "orbitType": [
                ("LEO", "LEO"), ("НОО", "LEO"), ("низкая околоземная орбита", "LEO"), ("низкая орбита", "LEO"),
                ("MEO", "MEO"), ("СОО", "MEO"), ("средняя околоземная орбита", "MEO"), ("средняя орбита", "MEO"),
                ("GEO", "GEO"), ("ГСО", "GEO"), ("геостационарная орбита", "GEO"), ("геостационар", "GEO"),
                ("SSO", "SSO"), ("ССО", "SSO"), ("солнечно-синхронная орбита", "SSO"),
                ("HEO", "HEO"), ("ВЭО", "HEO"), ("высокая эллиптическая орбита", "HEO"),
                ("Molniya", "Molniya"), ("орбита Молния", "Molniya"),
                ("Polar", "Polar"), ("полярная орбита", "Polar"),
                ("GTO", "GTO"), ("ГПО", "GTO"),
            ],
            
            "status": [
                ("активный", "активен"), ("рабочий", "активен"), ("функционирующий", "активен"), ("в строю", "активен"), ("работающий", "активен"),
                ("неактивный", "неактивен"), ("списанный", "неактивен"), ("вышедший из строя", "неактивен"), ("неработающий", "неактивен"), ("мусор", "неактивен"),
            ],
            
            "formFactor": [
                ("1U", "1U"), ("1 unit", "1U"), ("2U", "2U"), ("3U", "3U"), ("6U", "6U"), ("12U", "12U"),
                ("CubeSat", "CubeSat"), ("кубсат", "CubeSat"), ("наноспутник", "CubeSat"),
                ("SmallSat", "SmallSat"), ("малый спутник", "SmallSat"), ("микроспутник", "SmallSat"),
                ("Large", "Large"), ("большой спутник", "Large"), ("тяжелый аппарат", "Large"),
            ],
            
            "synonyms": [
                "спутник", "аппарат", "космический аппарат", "КА", "объект", "искусственный спутник", "сателлит", "система"
            ],
            
            "units": {
                "mass": ["кг", "килограмм", "кило"],
                "dist": ["км", "километров", "тыс. км"]
            }
        }

        # --- РАЗДЕЛЕНИЕ ЛЕКСИКИ ---
        
        # 1. ГЛАГОЛЫ
        self.VERBS_FORMAL = [
            "Найди", "Подбери", "Покажи", "Выведи", "Ищи", "Найти", "Отобрази", 
            "Предоставь", "Требуется", "Запроси", "Сформируй список", "Нужен", "Дай список"
        ]
        self.VERBS_SLANG = [
            "Чекни", "Глянь", "Пробей", "Посмотри", "Нарой", "Скинь", "Есть че", "Метни", "Поищи"
        ]

        # 2. ВВОДНЫЕ СЛОВА (FILLERS)
        self.FILLERS_FORMAL = [
            "пожалуйста,", "будь добр,", "если есть,", "подскажи,", "интересует,"
        ]
        self.FILLERS_SLANG = [
            "короче,", "слыш,", "типа", "эээ", "в натуре,", "чисто", "срочняк,", "брат,"
        ]

        # 3. ШАБЛОНЫ
        self.TEMPLATES_CLEAN = [
            ("{verb} {number} {synonym:accs}, которые покрывают {coverage:accs}.", ["number", "coverage"]),
            ("Нужен {synonym:nomn} для мониторинга {coverage:gent}.", ["coverage"]),
            ("Покажи {synonym:accs}, видящий {coverage:accs}, на {orbitType:loct}.", ["coverage", "orbitType"]),
            ("Ищи {synonym:accs} с массой {mass} и статусом {status:ablt}.", ["mass", "status"]),
            ("Есть ли {synonym:nomn} весом {mass} на {orbitType:loct}?", ["mass", "orbitType"]),
            ("Какие {synonym:nomn} летают над {coverage:ablt}?", ["coverage"]),
            ("Сколько {status:gent} {synonym:gent} находится на {orbitType:loct}?", ["status", "orbitType", "number"]),
            ("Запрос: {orbitType:nomn}, {coverage:nomn}, {status:nomn}.", ["orbitType", "coverage", "status"]),
            ("Фильтр: масса {mass}, регион {coverage:nomn}.", ["mass", "coverage"]),
            ("Для высоты {altitude} подбери {number} {synonym:accs}.", ["altitude", "number"]),
        ]

        self.TEMPLATES_NOISY = [
            ("{verb} {synonym:accs}... ну чтобы {coverage:accs} видел.", ["coverage"]),
            ("надо {synonym:accs} {formFactor} или типа того для {coverage:gent}", ["formFactor", "coverage"]),
            ("{status:nomn} {synonym:nomn} на {orbitType:loct} есть?", ["status", "orbitType"]),
            ("для {coverage:gent} {verb} что-нибудь на {altitude}", ["coverage", "altitude"]),
            ("интересно, а есть {synonym:nomn} {mass}?", ["mass"]),
            ("{coverage:nomn}... что там летает?", ["coverage"]),
            ("{verb} {number} {synonym:accs} {status:nomn}", ["number", "status"]),
        ]
        
        self.LOGIC_RULES = {
            "mass_limits": {"1U": (1, 2), "2U": (2, 3), "3U": (3, 6), "6U": (8, 12), "12U": (18, 25), "CubeSat": (1, 20), "Large": (500, 5000)},
            "orbit_alt": {"LEO": (160, 2000), "MEO": (2000, 35000), "GEO": (35700, 35800), "HEO": (500, 40000)}
        }

    def _add_typo(self, text: str) -> str:
        """Добавляет опечатку только для NOISY данных."""
        if random.random() > 0.3: return text # Даже в noisy не всегда опечатки
        
        chars = list(text)
        if len(chars) < 4: return text
        
        idx = random.randint(1, len(chars) - 2)
        typo_type = random.choice(['swap', 'drop', 'double'])
        
        if typo_type == 'swap':
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif typo_type == 'drop':
            chars.pop(idx)
        elif typo_type == 'double':
            chars.insert(idx, chars[idx])
            
        return "".join(chars)

    def _inflect_phrase(self, text: str, case: str, number: int = 1) -> str:
        words = text.split()
        res = []
        for word in words:
            if self._is_immutable(word):
                res.append(word)
                continue
            
            p = self.morph.parse(word)[0]
            grams = {case}
            if number > 1: grams.add('plur')
            inflected = p.inflect(grams)
            res.append(inflected.word if inflected else word)
        return " ".join(res)

    def _is_immutable(self, word: str) -> bool:
        if re.match(r'^[A-Za-z0-9\.\-]+$', word): return True
        if word.isupper() and len(word) > 1: return True
        return False

    def _number_to_text(self, num: int, case: str, is_clean: bool) -> str:
        # В чистых данных чаще пишем словами маленькие числа
        threshold = 0.4 if is_clean else 0.8
        
        if random.random() > threshold and num in self.NUM_TO_TEXT:
            word = self.NUM_TO_TEXT[num]
            p = self.morph.parse(word)[0]
            if case not in ['nomn', 'accs']:
                inflected = p.inflect({case})
                return inflected.word if inflected else str(num)
            return word
        return str(num)

    def _agree_with_number(self, word: str, number: int, case: str = 'nomn') -> str:
        p = self.morph.parse(word)[0]
        agreed = p.make_agree_with_number(number)
        if not agreed: return f"{number} {word}"
        
        if case == 'nomn':
            return agreed.word
        else:
            if number > 1:
                inflected = p.inflect({'plur', case})
            else:
                inflected = p.inflect({case})
            return inflected.word if inflected else word

    def _generate_mass_val(self, filters, is_clean: bool) -> Tuple[str, str]:
        min_m, max_m = 5, 5000
        if "formFactor" in filters:
            ff_raw = filters["formFactor"]
            for k, limits in self.LOGIC_RULES["mass_limits"].items():
                if k in ff_raw: 
                    min_m, max_m = limits
                    break
        val = random.randint(min_m, max_m)
        unit = random.choice(self.DATA["units"]["mass"])
        
        if random.random() > 0.7:
            val2 = val + random.randint(5, 50)
            text_phrase = f"от {val} до {val2} {unit}" if is_clean else f"{val}-{val2}"
            return f"{val}-{val2}", text_phrase
        else:
            return str(val), f"{val} {unit}"

    def _generate_alt_val(self, filters, is_clean: bool) -> Tuple[str, str]:
        min_a, max_a = 400, 36000
        orbit = filters.get("orbitType", "")
        if orbit in self.LOGIC_RULES["orbit_alt"]:
            min_a, max_a = self.LOGIC_RULES["orbit_alt"][orbit]
        val = random.randint(min_a, max_a)
        unit = random.choice(self.DATA["units"]["dist"])
        
        if random.random() > 0.7:
            val2 = val + random.randint(100, 2000)
            text_phrase = f"от {val} до {val2} {unit}" if is_clean else f"{val}-{val2}"
            return f"{val}-{val2} км", text_phrase
        else:
            return f"~{val} км", f"{val} {unit}"

    def generate_one(self) -> Optional[Dict[str, Any]]:
        # ОПРЕДЕЛЕНИЕ РЕЖИМА: ЧИСТЫЙ (75%) или ГРЯЗНЫЙ (25%)
        is_clean = random.random() < self.CLEAN_RATIO
        
        # Выбор шаблона и словарей в зависимости от режима
        if is_clean:
            template, req_fields = random.choice(self.TEMPLATES_CLEAN)
            verbs_pool = self.VERBS_FORMAL
            fillers_pool = self.FILLERS_FORMAL
        else:
            # В грязном режиме мешаем все подряд
            template, req_fields = random.choice(self.TEMPLATES_CLEAN + self.TEMPLATES_NOISY)
            verbs_pool = self.VERBS_FORMAL + self.VERBS_SLANG
            fillers_pool = self.FILLERS_FORMAL + self.FILLERS_SLANG

        filters = {}
        context = {}
        
        # 1. Генерация Числа
        if "number" in req_fields or "{number}" in template:
            num = random.randint(1, 12)
            filters["number"] = str(num)
        else:
            num = 1
            
        context["number"] = str(num)
        context["verb"] = random.choice(verbs_pool)
        synonym_base = random.choice(self.DATA["synonyms"])
        
        # 2. Заполнение полей
        shuffled_keys = self.ALL_FILTER_KEYS.copy()
        random.shuffle(shuffled_keys)
        target_fields = set(req_fields)
        while len(target_fields) < self.MIN_FILTERS and len(target_fields) < len(self.ALL_FILTER_KEYS):
            target_fields.add(random.choice(self.ALL_FILTER_KEYS))
            
        for field in target_fields:
            if field == "number": continue
            
            if field == "mass":
                f_val, t_val = self._generate_mass_val(filters, is_clean)
                filters["mass"] = f_val
                context["mass"] = t_val
            elif field == "altitude":
                f_val, t_val = self._generate_alt_val(filters, is_clean)
                filters["altitude"] = f_val
                context["altitude"] = t_val
            elif field in self.DATA:
                txt, val = random.choice(self.DATA[field])
                filters[field] = val
                context[field] = txt

        # 3. Подстановка в шаблон
        def replace_match(match):
            content = match.group(1)
            parts = content.split(":")
            key = parts[0]
            case = parts[1] if len(parts) > 1 else "nomn"
            
            if key == "number":
                return self._number_to_text(num, case, is_clean)
            
            if key == "synonym":
                word_agreed = self._agree_with_number(synonym_base, num, 'nomn')
                return self._inflect_phrase(word_agreed, case, num)
            
            if key in context:
                val_text = context[key]
                return self._inflect_phrase(val_text, case, num if key == 'status' else 1)
            return ""

        # Fallback
        for req in req_fields:
            if req not in context and req != "number":
                if req in self.DATA:
                    t, v = random.choice(self.DATA[req])
                    filters[req] = v
                    context[req] = t

        prompt = re.sub(r'{([\w:]+)}', replace_match, template)
        
        # 4. ПОСТ-ОБРАБОТКА ПО РЕЖИМАМ
        
        if is_clean:
            # ЧИСТЫЙ РЕЖИМ
            # Только вежливые вводные слова (редко)
            if random.random() > 0.85:
                filler = random.choice(self.FILLERS_FORMAL)
                prompt = f"{filler} {prompt}" if random.random() > 0.5 else f"{prompt} {filler}"
            
            prompt = re.sub(r'\s+', ' ', prompt).strip()
            prompt = prompt[0].upper() + prompt[1:] # Всегда с большой буквы
            
            # Всегда правильная пунктуация
            if not prompt.endswith(('.', '?', '!')):
                prompt += "?" if any(x in prompt.lower() for x in ["какие", "сколько", "есть ли"]) else "."
                
        else:
            # ГРЯЗНЫЙ РЕЖИМ
            # Добавляем любой мусор
            if random.random() > 0.6:
                filler = random.choice(fillers_pool)
                prompt = f"{filler} {prompt}" if random.random() > 0.5 else f"{prompt} {filler}"
            
            # Опечатки
            prompt = self._add_typo(prompt)
            
            prompt = re.sub(r'\s+', ' ', prompt).strip()
            
            # Случайный регистр
            if random.random() > 0.4:
                prompt = prompt.lower()
            else:
                prompt = prompt[0].upper() + prompt[1:]
                
            # Может не быть знака в конце
            if random.random() > 0.5 and not prompt.endswith(('.', '?', '!')):
                pass # Оставляем без точки

        final_filters = {k: filters.get(k, "") for k in self.ALL_FILTER_KEYS}
        return {"prompt": prompt, "filters": final_filters}

def main():
    try:
        count = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    except:
        count = 500

    gen = DatasetGenerator(min_filters=2)
    data = []
    unique_prompts = set()
    
    print(f"⚖️  Генерация {count} примеров (75% Clean / 25% Noisy)...")
    
    pbar = tqdm(total=count, unit="ex")
    attempts = 0
    
    while len(data) < count and attempts < count * 20:
        attempts += 1
        item = gen.generate_one()
        if item:
            p_hash = item['prompt'].lower()
            if p_hash not in unique_prompts:
                unique_prompts.add(p_hash)
                data.append(item)
                pbar.update(1)
    
    pbar.close()
    
    output_file = "prompts2.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\nГотово! Сохранено в {output_file}")

if __name__ == "__main__":
    main()