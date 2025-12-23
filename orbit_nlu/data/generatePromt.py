import json
import random
import re
import sys
import pymorphy2
from typing import Dict, Any, List, Optional, Tuple

class DatasetGenerator:
    """
    Продвинутый генератор датасета для NLU.
    Покрывает широкий спектр грамматических конструкций, синонимов и физических параметров.
    """

    def __init__(self, min_filters: int = 2):
        self.morph = pymorphy2.MorphAnalyzer()
        self.MIN_FILTERS = min_filters
        self.ALL_FILTER_KEYS = ["orbitType", "coverage", "altitude", "mass", "status", "formFactor", "number"]

        # --- БАЗА ЗНАНИЙ ---
        self.DATA = {
            "coverage": [
                # Страны и регионы
                ("Россия", "Россия"), ("РФ", "Россия"), ("Российская Федерация", "Россия"), ("территория России", "Россия"),
                ("США", "США"), ("Соединенные Штаты", "США"), ("Северная Америка", "Северная Америка"),
                ("Китай", "Китай"), ("КНР", "Китай"), ("Поднебесная", "Китай"),
                ("Европа", "Европа"), ("ЕС", "Европа"), ("Евросоюз", "Европа"), ("европейский континент", "Европа"),
                ("Африка", "Африка"), ("африканские страны", "Африка"),
                ("Южная Америка", "Южная Америка"), ("Латинская Америка", "Южная Америка"), ("Бразилия", "Южная Америка"),
                ("Австралия", "Австралия"), ("Океания", "Австралия"),
                ("Азия", "Азия"), ("Индия", "Индия"), ("Ближний Восток", "Ближний Восток"),
                # Специфические зоны
                ("Арктика", "Арктика"), ("Северный полюс", "Арктика"), ("Севморпуть", "Арктика"), ("заполярье", "Арктика"),
                ("Антарктида", "Антарктида"), ("Южный полюс", "Антарктида"),
                ("Тихий океан", "Тихий океан"), ("Атлантика", "Атлантический океан"),
                ("Экватор", "Экватор"), ("экваториальная зона", "Экватор"),
            ],
            "orbitType": [
                # (Текст, Значение) - предлоги добавляются динамически или в шаблоне
                ("LEO", "LEO"), ("НОО", "LEO"), ("низкая околоземная орбита", "LEO"), ("низкая орбита", "LEO"),
                ("MEO", "MEO"), ("СОО", "MEO"), ("средняя околоземная орбита", "MEO"), ("средняя орбита", "MEO"),
                ("GEO", "GEO"), ("ГСО", "GEO"), ("геостационарная орбита", "GEO"), ("геостационар", "GEO"),
                ("SSO", "SSO"), ("ССО", "SSO"), ("солнечно-синхронная орбита", "SSO"),
                ("HEO", "HEO"), ("ВЭО", "HEO"), ("высокая эллиптическая орбита", "HEO"), ("высокая орбита", "HEO"),
                ("Molniya", "Molniya"), ("орбита Молния", "Molniya"), ("молния", "Molniya"),
                ("Polar", "Polar"), ("полярная орбита", "Polar"),
                ("GTO", "GTO"), ("ГПО", "GTO"), ("геопереходная орбита", "GTO"),
            ],
            "status": [
                ("активный", "активен"), ("рабочий", "активен"), ("живой", "активен"), ("функционирующий", "активен"), ("в строю", "активен"), ("работающий", "активен"),
                ("неактивный", "неактивен"), ("мертвый", "неактивен"), ("списанный", "неактивен"), ("вышедший из строя", "неактивен"), ("сломанный", "неактивен"), ("мусор", "неактивен"),
            ],
            "formFactor": [
                ("1U", "1U"), ("один юнит", "1U"), ("1 unit", "1U"),
                ("2U", "2U"), ("два юнита", "2U"),
                ("3U", "3U"), ("три юнита", "3U"), ("трехюнитовый", "3U"),
                ("6U", "6U"), ("шесть юнитов", "6U"),
                ("12U", "12U"), ("12 юнитов", "12U"),
                ("16U", "16U"),
                ("CubeSat", "CubeSat"), ("кубсат", "CubeSat"), ("наноспутник", "CubeSat"),
                ("SmallSat", "SmallSat"), ("малый спутник", "SmallSat"), ("малый КА", "SmallSat"),
            ],
            "synonyms": [
                "спутник", "аппарат", "космический аппарат", "КА", "борт", "объект", "искусственный спутник"
            ],
            "verbs": [
                "Найди", "Подбери", "Покажи", "Выведи", "Ищи", "Нужен", "Требуется", "Дай список", "Есть ли", 
                "Предоставь", "Отобрази", "Найти", "Запроси", "Сформируй список"
            ],
            "units": {
                "mass": ["кг", "килограмм", "кило"],
                "dist": ["км", "километров", "тыс. км"]
            }
        }

        # --- ШАБЛОНЫ ---
        # {ключ:падеж}
        # nomn=Им, gent=Род, datv=Дат, accs=Вин, ablt=Твор, loct=Предл
        self.TEMPLATES = [
            # Простые запросы
            ("{verb} {number} {synonym:accs}, которые покрывают {coverage:accs}.", ["number", "coverage"]),
            ("Нужен {synonym:nomn} для мониторинга {coverage:gent}.", ["coverage"]),
            ("Покажи {synonym:accs}, видящий {coverage:accs}, на {orbitType:loct}.", ["coverage", "orbitType"]),
            
            # С параметрами физики
            ("Ищи {synonym:accs} с массой {mass} и статусом {status:ablt}.", ["mass", "status"]),
            ("Есть ли {synonym:nomn} весом {mass} на {orbitType:loct}?", ["mass", "orbitType"]),
            ("Требуются {status:nomn} {synonym:nomn} {formFactor}.", ["status", "formFactor"]),
            
            # Сложные конструкции
            ("Для высоты {altitude} подбери {number} {synonym:accs} типа {formFactor}.", ["altitude", "number", "formFactor"]),
            ("Выведи список: {synonym:nomn}, регион {coverage:nomn}, орбита {orbitType:nomn}.", ["coverage", "orbitType"]),
            ("Интересуют {synonym:nomn} на {orbitType:loct} высотой {altitude}.", ["orbitType", "altitude"]),
            
            # Разговорный / Краткий стиль
            ("{orbitType:nomn}, {coverage:nomn}, {status:nomn}.", ["orbitType", "coverage", "status"]),
            ("{synonym:nomn} {formFactor}, масса {mass}, {status:nomn}.", ["formFactor", "mass", "status"]),
            
            # Вопросы
            ("Какие {synonym:nomn} летают над {coverage:ablt}?", ["coverage"]),
            ("Сколько {status:gent} {synonym:gent} находится на {orbitType:loct}?", ["status", "orbitType", "number"]), # Тут хитро с "Сколько", number в ответе мб
            
            # Специфичные
            ("{verb} {synonym:accs} с перигеем/апогеем {altitude}.", ["altitude"]),
            ("Нужны {synonym:nomn} ({formFactor}) для работы по {coverage:datv}.", ["formFactor", "coverage"]),
        ]
        
        # Правила для генерации логичных данных
        self.LOGIC_RULES = {
            "mass_limits": {"1U": (1, 2), "2U": (2, 3), "3U": (3, 6), "6U": (8, 12), "12U": (18, 25), "CubeSat": (1, 20)},
            "orbit_alt": {
                "LEO": (160, 2000),
                "MEO": (2000, 35000),
                "GEO": (35700, 35800),
                "HEO": (500, 40000) # эллипс
            }
        }

    def _inflect_phrase(self, text: str, case: str, number: int = 1) -> str:
        """
        Умное склонение фразы. 
        Учитывает, что некоторые слова (аббревиатуры) не склоняются.
        Согласует прилагательные с существительными.
        """
        words = text.split()
        res = []
        
        # Пытаемся найти главное существительное для согласования рода (эвристика)
        # Обычно это последнее слово, если оно не цифра/аббревиатура
        main_noun_tag = None
        for w in reversed(words):
            if not self._is_immutable(w):
                p = self.morph.parse(w)[0]
                if 'NOUN' in p.tag:
                    main_noun_tag = p.tag
                    break
        
        for word in words:
            if self._is_immutable(word):
                res.append(word)
                continue
            
            p = self.morph.parse(word)[0]
            
            # Если число > 1, форсируем множественное число
            # Но есть нюанс: "21 спутник" (ед.ч), "22 спутника" (ед.ч, род.п? нет, тут сложно).
            # Для простоты в NLU датасетах, если number > 1, используем plural форму для всех,
            # КРОМЕ случаев, когда мы явно согласуем с числительным через _agree_with_number.
            # Здесь метод просто ставит фразу в падеж.
            
            grams = {case}
            if number > 1:
                grams.add('plur')
                
            inflected = p.inflect(grams)
            
            # Если не удалось просклонять (например, неизменяемое слово, не попавшее в фильтр)
            res.append(inflected.word if inflected else word)
            
        return " ".join(res)

    def _is_immutable(self, word: str) -> bool:
        """Проверка на неизменяемые слова (аббревиатуры, латиница, цифры)."""
        if re.match(r'^[A-Za-z0-9]+$', word): return True
        if word.isupper() and len(word) > 1: return True # ГСО, КНР
        return False

    def _agree_with_number(self, word: str, number: int, case: str = 'nomn') -> str:
        """
        Согласует слово с числом и падежом всей фразы.
        Пример: 
        (спутник, 5, nomn) -> 5 спутников
        (спутник, 2, gent) -> (нет) 2 спутников
        """
        p = self.morph.parse(word)[0]
        
        # Сначала согласуем с числом в именительном падеже (1 спутник, 2 спутника, 5 спутников)
        agreed = p.make_agree_with_number(number)
        
        if not agreed: 
            return f"{number} {word}"

        # Если падеж фразы не именительный (например "вижу 5 спутников"), 
        # то pymorphy make_agree_with_number возвращает форму, зависящую от числа.
        # Но если нам нужно склонять ВСЮ группу (например "о 5 спутниках"), нужно доп. действие.
        
        if case == 'nomn':
            return agreed.word
        else:
            # Грубая эвристика: для >1 во всех косвенных падежах используется plural
            # кроме винительного (вижу 2 спутника vs вижу 5 спутников).
            # Для генерации промптов достаточно поставить слово в Plural + Case, если num > 1
            if number > 1:
                inflected = p.inflect({'plur', case})
            else:
                inflected = p.inflect({case})
            return inflected.word if inflected else word

    def _generate_mass_val(self, filters) -> Tuple[str, str]:
        """Генерирует значение фильтра массы и текст для промпта."""
        # Определяем границы на основе форм-фактора
        min_m, max_m = 5, 5000
        if "formFactor" in filters:
            ff_raw = filters["formFactor"]
            # Ищем ключ в RULES
            for k, limits in self.LOGIC_RULES["mass_limits"].items():
                if k in ff_raw: # например "1U" in "1U"
                    min_m, max_m = limits
                    break
        
        val = random.randint(min_m, max_m)
        
        # 30% вероятность генерации диапазона
        if random.random() > 0.7:
            val2 = val + random.randint(5, 50)
            filter_val = f"{val}-{val2}"
            
            phrases = [
                f"от {val} до {val2}",
                f"{val}-{val2}",
                f"в диапазоне {val}...{val2}"
            ]
            text_val = random.choice(phrases)
        else:
            filter_val = str(val)
            # Варианты текста
            phrases = [
                str(val),
                f"около {val}",
                f"порядка {val}",
                f"> {val-1}", # Хитрый вариант
            ]
            text_val = random.choice(phrases)
            
        # Добавляем единицы измерения
        unit = random.choice(self.DATA["units"]["mass"])
        return filter_val, f"{text_val} {unit}"

    def _generate_alt_val(self, filters) -> Tuple[str, str]:
        """Генерирует высоту."""
        min_a, max_a = 400, 36000
        orbit = filters.get("orbitType", "")
        if orbit in self.LOGIC_RULES["orbit_alt"]:
            min_a, max_a = self.LOGIC_RULES["orbit_alt"][orbit]
            
        val = random.randint(min_a, max_a)
        unit = random.choice(self.DATA["units"]["dist"])
        
        if random.random() > 0.6:
            # Диапазон
            val2 = val + random.randint(100, 2000)
            filter_val = f"{val}-{val2} км"
            text_val = f"от {val} до {val2} {unit}"
        else:
            # Точное (или "около")
            filter_val = f"~{val} км"
            prefix = random.choice(["", "около ", "примерно ", "на высоте "])
            text_val = f"{prefix}{val} {unit}"
            
        return filter_val, text_val

    def generate_one(self) -> Optional[Dict[str, Any]]:
        template, req_fields = random.choice(self.TEMPLATES)
        filters = {}
        context = {}
        
        # 1. Генерация Числа (number)
        # Если в шаблоне есть {number} или {synonym} (который зависит от number)
        if "number" in req_fields or "{number}" in template:
            num = random.randint(1, 15)
            filters["number"] = str(num)
        else:
            # 20% шанс, что пользователь укажет число, даже если шаблон не требует явно (implicit)
            # Но для простоты, если шаблона нет под число, считаем 1.
            num = 1
            
        context["number"] = str(num)
        context["verb"] = random.choice(self.DATA["verbs"])
        synonym_base = random.choice(self.DATA["synonyms"])
        
        # 2. Заполнение полей
        # Сначала заполняем обязательные поля из шаблона
        shuffled_keys = self.ALL_FILTER_KEYS.copy()
        random.shuffle(shuffled_keys) # Случайный порядок заполнения для вариативности
        
        # Объединяем обязательные поля и случайные дополнительные (чтобы было >= MIN_FILTERS)
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
            content = match.group(1) # например "synonym:accs"
            parts = content.split(":")
            key = parts[0]
            case = parts[1] if len(parts) > 1 else "nomn"
            
            if key == "number":
                return str(num)
            
            if key == "synonym":
                # Согласуем "спутник" с числом, потом склоняем
                # Если падеж nomn - make_agree делает всё само.
                # Если косвенный - нужно склонять.
                word_agreed = self._agree_with_number(synonym_base, num, 'nomn')
                # Если слово уже во мн.ч (5 спутников), inflect_phrase просто просклоняет его по падежу
                return self._inflect_phrase(word_agreed, case, num)
            
            if key in context:
                val_text = context[key]
                # Особая обработка для статуса ("активные")
                return self._inflect_phrase(val_text, case, num if key == 'status' else 1)
            
            return ""

        # Если в шаблоне есть поле, которого нет в фильтрах (например, шаблон требует mass, а мы её не сгенерировали по ошибке),
        # скрипт упадет. Но мы выше заполнили target_fields на основе req_fields.
        # Однако, target_fields - это множество. А req_fields - список из шаблона.
        # Проверим, все ли поля из шаблона есть в context.
        for req in req_fields:
            if req not in context and req != "number":
                # Fallback: генерируем на лету
                if req in self.DATA:
                    t, v = random.choice(self.DATA[req])
                    filters[req] = v
                    context[req] = t

        prompt = re.sub(r'{([\w:]+)}', replace_match, template)
        
        # Очистка
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        # Капитализация первого слова
        prompt = prompt[0].upper() + prompt[1:]
        
        # Исправление пунктуации (если шаблон заканчивается некрасиво)
        if not prompt.endswith(('.', '?', '!')):
            prompt += "?" if any(x in prompt.lower() for x in ["какие", "сколько", "есть ли"]) else "."

        # Формируем итоговый JSON (заполняем пропуски)
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
    
    print(f"Генерация {count} уникальных примеров...")
    
    attempts = 0
    while len(data) < count and attempts < count * 5:
        attempts += 1
        item = gen.generate_one()
        if item:
            # Дедупликация по тексту промпта
            p_hash = item['prompt'].lower()
            if p_hash not in unique_prompts:
                unique_prompts.add(p_hash)
                data.append(item)
    
    output_file = "prompts_generated.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Готово! Сгенерировано {len(data)} примеров. Сохранено в {output_file}")

if __name__ == "__main__":
    main()