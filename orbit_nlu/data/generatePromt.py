import json
import random
import re
import sys
import pymorphy2
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

class DatasetGenerator:
    """
    Генератор датасета NLU (версия 'Heavy').
    База знаний увеличена в ~10 раз для максимальной вариативности.
    """

    def __init__(self, min_filters: int = 2):
        self.morph = pymorphy2.MorphAnalyzer()
        self.MIN_FILTERS = min_filters
        self.ALL_FILTER_KEYS = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "number"]

        # Словарь для конвертации чисел в текст
        self.NUM_TO_TEXT = {
            1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
            6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
            11: "одиннадцать", 12: "двенадцать", 15: "пятнадцать", 20: "двадцать"
        }

        # --- ГИГАНТСКАЯ БАЗА ЗНАНИЙ ---
        self.DATA = {
            "coverage": [
                # === РОССИЯ И СНГ ===
                ("Россия", "Россия"), ("РФ", "Россия"), ("Российская Федерация", "Россия"), ("RUSSIA", "Россия"),
                ("территория России", "Россия"), ("наша страна", "Россия"), ("Отечество", "Россия"),
                ("Москва", "Россия"), ("Питер", "Россия"), ("СПб", "Россия"), ("Московская область", "Россия"),
                ("Урал", "Россия"), ("Сибирь", "Россия"), ("Дальний Восток", "Россия"), ("Камчатка", "Россия"),
                ("Сахалин", "Россия"), ("Курилы", "Россия"), ("Байкал", "Россия"), ("Поволжье", "Россия"),
                ("Кавказ", "Россия"), ("Сочи", "Россия"), ("Крым", "Россия"), ("Калининград", "Россия"),
                ("Новосибирск", "Россия"), ("Владивосток", "Россия"), ("Екатеринбург", "Россия"),
                ("Казахстан", "Казахстан"), ("Байконур", "Казахстан"), ("Астана", "Казахстан"),
                ("Беларусь", "Беларусь"), ("Минск", "Беларусь"), ("Украина", "Европа"), ("Киев", "Европа"),

                # === СЕВЕРНАЯ АМЕРИКА ===
                ("США", "США"), ("USA", "США"), ("Соединенные Штаты", "США"), ("Штаты", "США"), ("Америка", "США"),
                ("Вашингтон", "США"), ("Нью-Йорк", "США"), ("Калифорния", "США"), ("Техас", "США"), ("Флорида", "США"),
                ("Мыс Канаверал", "США"), ("Аляска", "США"), ("Гавайи", "США"), ("Пентагон", "США"),
                ("Канада", "Канада"), ("Торонто", "Канада"), ("Оттава", "Канада"), ("Ванкувер", "Канада"),
                ("Мексика", "Северная Америка"),

                # === ЮЖНАЯ АМЕРИКА ===
                ("Южная Америка", "Южная Америка"), ("Латинская Америка", "Южная Америка"), ("Латам", "Южная Америка"),
                ("Бразилия", "Южная Америка"), ("Амазонка", "Южная Америка"), ("Рио", "Южная Америка"),
                ("Аргентина", "Южная Америка"), ("Чили", "Южная Америка"), ("Анды", "Южная Америка"),
                ("Куру", "Южная Америка"), ("космодром Куру", "Южная Америка"),

                # === АЗИЯ ===
                ("Китай", "Китай"), ("КНР", "Китай"), ("China", "Китай"), ("Поднебесная", "Китай"),
                ("Пекин", "Китай"), ("Шанхай", "Китай"), ("Тибет", "Китай"), ("Гонконг", "Китай"),
                ("Индия", "Индия"), ("Дели", "Индия"), ("Мумбаи", "Индия"), ("Бангалор", "Индия"),
                ("Япония", "Япония"), ("Токио", "Япония"), ("Острова", "Япония"),
                ("Корея", "Южная Корея"), ("Сеул", "Южная Корея"), ("КНДР", "Северная Корея"),
                ("Азия", "Азия"), ("Юго-Восточная Азия", "Азия"), ("Тайвань", "Азия"),
                ("Ближний Восток", "Ближний Восток"), ("Израиль", "Ближний Восток"), ("ОАЭ", "Ближний Восток"),
                ("Дубай", "Ближний Восток"), ("Иран", "Ближний Восток"), ("Турция", "Ближний Восток"),

                # === ЕВРОПА ===
                ("Европа", "Европа"), ("ЕС", "Европа"), ("Евросоюз", "Европа"), ("EU", "Европа"), ("Старый Свет", "Европа"),
                ("Западная Европа", "Европа"), ("Восточная Европа", "Европа"), ("Скандинавия", "Европа"),
                ("Германия", "Европа"), ("Берлин", "Европа"), ("Франция", "Европа"), ("Париж", "Европа"),
                ("Британия", "Европа"), ("Лондон", "Европа"), ("UK", "Европа"), ("Италия", "Европа"), ("Рим", "Европа"),
                ("Испания", "Европа"), ("Польша", "Европа"),

                # === АФРИКА И ОКЕАНИЯ ===
                ("Африка", "Африка"), ("африканский континент", "Африка"), ("Сахара", "Африка"),
                ("Египет", "Африка"), ("Каир", "Африка"), ("ЮАР", "Африка"), ("Нигерия", "Африка"),
                ("Австралия", "Австралия"), ("Сидней", "Австралия"), ("Мельбурн", "Австралия"),
                ("Океания", "Австралия"), ("Новая Зеландия", "Австралия"),

                # === ВОДОЕМЫ И ЗОНЫ ===
                ("Арктика", "Арктика"), ("Северный полюс", "Арктика"), ("Севморпуть", "Арктика"), ("Заполярье", "Арктика"),
                ("Антарктида", "Антарктида"), ("Южный полюс", "Антарктида"), ("Антарктика", "Антарктида"),
                ("Тихий океан", "Тихий океан"), ("Пацифика", "Тихий океан"),
                ("Атлантика", "Атлантический океан"), ("Атлантический океан", "Атлантический океан"),
                ("Индийский океан", "Индийский океан"),
                ("Черное море", "Россия"), ("Балтика", "Европа"), ("Средиземноморье", "Европа"),
                ("Экватор", "Экватор"), ("экваториальная зона", "Экватор"), ("тропики", "Экватор"),
                ("Весь мир", "Global"), ("Глобально", "Global"), ("Планета", "Global"), ("Земля", "Global"),
            ],
            
            "orbitType": [
                # LEO
                ("LEO", "LEO"), ("НОО", "LEO"), ("низкая околоземная орбита", "LEO"), ("низкая орбита", "LEO"),
                ("низкая высота", "LEO"), ("Low Earth Orbit", "LEO"), ("на низах", "LEO"), ("низколетящие", "LEO"),
                ("200-2000 км", "LEO"), ("околоземная", "LEO"),
                # MEO
                ("MEO", "MEO"), ("СОО", "MEO"), ("средняя околоземная орбита", "MEO"), ("средняя орбита", "MEO"),
                ("Medium Earth Orbit", "MEO"), ("навигационная орбита", "MEO"), ("GPS-орбита", "MEO"), ("ГЛОНАСС-орбита", "MEO"),
                ("полусуточная", "MEO"),
                # GEO
                ("GEO", "GEO"), ("ГСО", "GEO"), ("геостационарная орбита", "GEO"), ("геостационар", "GEO"),
                ("Geostationary", "GEO"), ("геостационарное кольцо", "GEO"), ("ГСО-точка", "GEO"), ("стационар", "GEO"),
                ("36000", "GEO"), ("висящие на месте", "GEO"), ("синхронная с Землей", "GEO"),
                # SSO
                ("SSO", "SSO"), ("ССО", "SSO"), ("солнечно-синхронная орбита", "SSO"), ("солнечно-синхронная", "SSO"),
                ("Sun-Synchronous", "SSO"), ("солнечная синхронизация", "SSO"), ("по солнцу", "SSO"),
                # HEO / Molniya
                ("HEO", "HEO"), ("ВЭО", "HEO"), ("высокая эллиптическая орбита", "HEO"), ("высокая орбита", "HEO"),
                ("эллиптическая", "HEO"), ("вытянутая орбита", "HEO"), ("High Earth Orbit", "HEO"),
                ("Molniya", "Molniya"), ("орбита Молния", "Molniya"), ("молния", "Molniya"), ("высокоэллиптическая", "Molniya"),
                ("Tundra", "HEO"), ("Тундра", "HEO"),
                # Other / Slang
                ("Polar", "Polar"), ("полярная орбита", "Polar"), ("через полюса", "Polar"), ("полярник", "Polar"),
                ("GTO", "GTO"), ("ГПО", "GTO"), ("геопереходная орбита", "GTO"), ("переходная", "GTO"),
                ("Graveyard", "Graveyard"), ("орбита захоронения", "Graveyard"), ("кладбище", "Graveyard"), ("мусорная орбита", "Graveyard"),
                ("LEO/SSO", "SSO"), ("экваториальная", "Equatorial"),
            ],
            
            "status": [
                # Active
                ("активный", "активен"), ("рабочий", "активен"), ("живой", "активен"), ("функционирующий", "активен"), 
                ("в строю", "активен"), ("работающий", "активен"), ("действующий", "активен"), ("operational", "активен"),
                ("on-duty", "активен"), ("в эксплуатации", "активен"), ("исправный", "активен"), ("на связи", "активен"),
                ("передающий", "активен"), ("включенный", "активен"), ("OK", "активен"), ("nominal", "активен"),
                
                # Inactive / Dead
                ("неактивный", "неактивен"), ("мертвый", "неактивен"), ("списанный", "неактивен"), 
                ("вышедший из строя", "неактивен"), ("сломанный", "неактивен"), ("мусор", "неактивен"), 
                ("космический мусор", "неактивен"), ("обломок", "неактивен"), ("retired", "неактивен"),
                ("дохлый", "неактивен"), ("не отвечающий", "неактивен"), ("broken", "неактивен"), ("dead", "неактивен"),
                ("потерянный", "неактивен"), ("отключенный", "неактивен"), ("умолкнувший", "неактивен"),
                ("деорбитированный", "неактивен"), ("сгоревший", "неактивен"), ("аварийный", "неактивен"),
                
                # Special (mapped to broad categories usually, but here as text variation)
                ("тестовый", "активен"), ("резервный", "неактивен"), ("в консервации", "неактивен"),
            ],
            
            "formFactor": [
                # CubeSats
                ("1U", "1U"), ("один юнит", "1U"), ("1 unit", "1U"), ("одноюнитовый", "1U"), ("1-U", "1U"),
                ("2U", "2U"), ("два юнита", "2U"), ("двухюнитовый", "2U"),
                ("3U", "3U"), ("три юнита", "3U"), ("трехюнитовый", "3U"), ("3-U", "3U"),
                ("6U", "6U"), ("шесть юнитов", "6U"), ("6-unit", "6U"), ("шестиюнитовый", "6U"),
                ("12U", "12U"), ("12 юнитов", "12U"), ("12-unit", "12U"),
                ("16U", "16U"), ("27U", "27U"),
                ("CubeSat", "CubeSat"), ("кубсат", "CubeSat"), ("наноспутник", "CubeSat"), ("cube", "CubeSat"),
                ("кубик", "CubeSat"), ("NanoSat", "CubeSat"), ("PicoSat", "CubeSat"),
                
                # SmallSats
                ("SmallSat", "SmallSat"), ("малый спутник", "SmallSat"), ("малый КА", "SmallSat"), ("микроспутник", "SmallSat"),
                ("MicroSat", "SmallSat"), ("миниспутник", "SmallSat"), ("платформа", "SmallSat"),
                
                # Big
                ("Full-scale", "Large"), ("большой спутник", "Large"), ("тяжелый аппарат", "Large"), ("тяжеловес", "Large"),
                ("стандартный", "Large"), ("геостационарный борт", "Large"), ("тонник", "Large"),
                ("телеком-спутник", "Large"), ("обсерватория", "Large"), ("станция", "Large"),
            ],
            
            "synonyms": [
                # Formal
                "спутник", "аппарат", "космический аппарат", "КА", "объект", "искусственный спутник",
                "сателлит", "ИСЗ", "автоматическая станция", "платформа", "система", "устройство",
                
                # Slang / Jargon
                "борт", "птичка", "изделие", "железяка", "штука", "единица", "точка", "цель",
                "банка", "коробка", "посудина", "разведчик", "ретранслятор", "зонд",
                "bird", "sat", "unit", "target", "космолет"
            ],
            
            "verbs": [
                # Direct
                "Найди", "Подбери", "Покажи", "Выведи", "Ищи", "Найти", "Отобрази", "Дай", "Выдай",
                
                # Polite / Soft
                "Нужен", "Требуется", "Есть ли", "Хочу найти", "Нужен список", "Подскажи", 
                "Интересует", "Интересуют", "Хотелось бы узнать", "Помоги найти", "Можешь показать",
                
                # Imperative / Slang
                "Запроси", "Сформируй список", "Сгенерируй", "Перечисли", "Назови",
                "Чекни", "Глянь", "Пробей", "Посмотри", "Нарой", "Отфильтруй", "Выбери",
                "Листани", "Скинь", "Поищи", "Отыщи", "Вычисли", "Детектируй"
            ],
            
            "units": {
                "mass": ["кг", "килограмм", "кило", "kg", "килограммов", "кг."],
                "dist": ["км", "километров", "тыс. км", "km", "километра", "тысяч километров"]
            },
            
            "filler_words": [
                "пожалуйста", "плз", "срочно", "кстати,", "эээ", "может быть", "примерно", 
                "типа", "вообще", "если есть,", "короче,", "слушай,", "подскажи,",
                "братан,", "в общем,", "как бы,", "ну,", "эмм,", "срочняк,", "для диплома,",
                "по-братски,", "чисто,", "реально,", "в натуре,", "допустим,"
            ]
        }

        self.TEMPLATES = [
            # Стандартные
            ("{verb} {number} {synonym:accs}, которые покрывают {coverage:accs}.", ["number", "coverage"]),
            ("Нужен {synonym:nomn} для мониторинга {coverage:gent}.", ["coverage"]),
            ("Покажи {synonym:accs}, видящий {coverage:accs}, на {orbitType:loct}.", ["coverage", "orbitType"]),
            ("Ищи {synonym:accs} с массой {mass} и статусом {status:ablt}.", ["mass", "status"]),
            
            # Вопросы
            ("Есть ли {synonym:nomn} весом {mass} на {orbitType:loct}?", ["mass", "orbitType"]),
            ("Какие {synonym:nomn} летают над {coverage:ablt}?", ["coverage"]),
            ("Сколько {status:gent} {synonym:gent} находится на {orbitType:loct}?", ["status", "orbitType", "number"]),
            ("Можешь найти {synonym:accs} ({formFactor})?", ["formFactor"]),
            
            # Технические / Телеграфные
            ("{orbitType:nomn}, {coverage:nomn}, {status:nomn}.", ["orbitType", "coverage", "status"]),
            ("Фильтр: масса {mass}, регион {coverage:nomn}.", ["mass", "coverage"]),
            ("Запрос на {synonym:accs}: {formFactor}, {altitude}.", ["formFactor", "altitude"]),
            ("Параметры: {orbitType:nomn}, {mass}.", ["orbitType", "mass"]),
            
            # Разговорные / Несвязные
            ("{verb} {synonym:accs}... ну чтобы {coverage:accs} видел.", ["coverage"]),
            ("надо {synonym:accs} {formFactor} или типа того для {coverage:gent}", ["formFactor", "coverage"]),
            ("{status:nomn} {synonym:nomn} на {orbitType:loct} есть?", ["status", "orbitType"]),
            ("для {coverage:gent} {verb} что-нибудь на {altitude}", ["coverage", "altitude"]),
            ("слыш, {verb} {synonym:accs} {status:accs}", ["status"]),
            ("интересно, а есть {synonym:nomn} {mass}?", ["mass"]),
            ("{coverage:nomn}... что там летает?", ["coverage"]),
        ]
        
        self.LOGIC_RULES = {
            "mass_limits": {"1U": (1, 2), "2U": (2, 3), "3U": (3, 6), "6U": (8, 12), "12U": (18, 25), "CubeSat": (1, 20), "Large": (500, 5000)},
            "orbit_alt": {"LEO": (160, 2000), "MEO": (2000, 35000), "GEO": (35700, 35800), "HEO": (500, 40000)}
        }

    def _add_typo(self, text: str) -> str:
        """Добавляет случайную опечатку в текст с вероятностью 15%."""
        if random.random() > 0.15: return text
        
        chars = list(text)
        if len(chars) < 4: return text
        
        idx = random.randint(1, len(chars) - 2)
        typo_type = random.choice(['swap', 'drop', 'double', 'wrong_key'])
        
        if typo_type == 'swap':
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif typo_type == 'drop':
            chars.pop(idx)
        elif typo_type == 'double':
            chars.insert(idx, chars[idx])
        elif typo_type == 'wrong_key':
            # Симуляция промаха по клавише (очень грубая)
            # Заменим гласную на гласную или согласную на соседнюю (условно)
            if chars[idx] in 'аеёиоуыэюя':
                chars[idx] = random.choice('аеёиоуыэюя'.replace(chars[idx], ''))
            
        return "".join(chars)

    def _add_filler(self, text: str) -> str:
        """Добавляет мусорные слова в начало или конец."""
        if random.random() > 0.6: # Повысили вероятность мусора
            filler = random.choice(self.DATA["filler_words"])
            if random.random() > 0.5:
                text = f"{filler} {text}"
            else:
                text = f"{text} {filler}"
        return text

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
        # Регулярка теперь разрешает точки, дефисы и цифры внутри
        if re.match(r'^[A-Za-z0-9\.\-]+$', word): return True
        # Если слово капсом (аббревиатура) и длиннее 1 буквы
        if word.isupper() and len(word) > 1: return True
        return False

    def _number_to_text(self, num: int, case: str) -> str:
        """Конвертирует число в текст (пять) или оставляет цифрой (5)."""
        # 70% цифра, 30% текст (только для малых чисел)
        if random.random() > 0.7 and num in self.NUM_TO_TEXT:
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

    def _generate_mass_val(self, filters) -> Tuple[str, str]:
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
            return f"{val}-{val2}", f"от {val} до {val2} {unit}"
        else:
            return str(val), f"{val} {unit}"

    def _generate_alt_val(self, filters) -> Tuple[str, str]:
        min_a, max_a = 400, 36000
        orbit = filters.get("orbitType", "")
        if orbit in self.LOGIC_RULES["orbit_alt"]:
            min_a, max_a = self.LOGIC_RULES["orbit_alt"][orbit]
        val = random.randint(min_a, max_a)
        unit = random.choice(self.DATA["units"]["dist"])
        
        if random.random() > 0.7:
            val2 = val + random.randint(100, 2000)
            return f"{val}-{val2} км", f"{val}-{val2} {unit}"
        else:
            # Варианты: просто число, "~число", "высота число"
            r = random.random()
            if r < 0.3:
                return f"~{val} км", f"~{val} {unit}"
            elif r < 0.6:
                return f"~{val} км", f"высота {val} {unit}"
            else:
                return f"~{val} км", f"{val} {unit}"

    def generate_one(self) -> Optional[Dict[str, Any]]:
        template, req_fields = random.choice(self.TEMPLATES)
        filters = {}
        context = {}
        
        # 1. Генерация Числа
        if "number" in req_fields or "{number}" in template:
            num = random.randint(1, 12)
            filters["number"] = str(num)
        else:
            num = 1
            
        context["number"] = str(num)
        context["verb"] = random.choice(self.DATA["verbs"])
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
                f_val, t_val = self._generate_mass_val(filters)
                filters["mass"] = f_val
                context["mass"] = t_val
            elif field == "altitude":
                f_val, t_val = self._generate_alt_val(filters)
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
                return self._number_to_text(num, case)
            
            if key == "synonym":
                word_agreed = self._agree_with_number(synonym_base, num, 'nomn')
                return self._inflect_phrase(word_agreed, case, num)
            
            if key in context:
                val_text = context[key]
                return self._inflect_phrase(val_text, case, num if key == 'status' else 1)
            return ""

        for req in req_fields:
            if req not in context and req != "number":
                if req in self.DATA:
                    t, v = random.choice(self.DATA[req])
                    filters[req] = v
                    context[req] = t

        prompt = re.sub(r'{([\w:]+)}', replace_match, template)
        
        # 4. Пост-обработка: Шум, Опечатки, Форматирование
        prompt = self._add_filler(prompt)
        prompt = self._add_typo(prompt)
        
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        if random.random() > 0.5:
            prompt = prompt.lower()
        else:
            prompt = prompt[0].upper() + prompt[1:]
        
        if random.random() > 0.3:
            if not prompt.endswith(('.', '?', '!')):
                prompt += "?" if any(x in prompt.lower() for x in ["какие", "сколько", "есть ли"]) else "."

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
    
    print(f"🚀 Генерация {count} уникальных примеров (База знаний X10)...")
    
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
    
    output_file = "prompts_generated.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Готово! Сохранено в {output_file}")

if __name__ == "__main__":
    main()