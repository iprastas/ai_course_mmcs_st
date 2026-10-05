"""Пять инструментов над базой мест Ростова и их JSON-схемы.

Инструменты исполняются нашим циклом (без фреймворка): имя + JSON-аргументы
приходят от модели, результат уходит ей наблюдением. Ошибки в данных
(неизвестный вид места) возвращаются строкой — модель получает шанс
исправиться, а не роняет прогон.

Описания есть в двух вариантах для A/B-эксперимента:
  short — одна фраза на инструмент;
  full  — что делает, когда звать, когда не звать, единицы, пример и
          описания полей.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Callable

DB = Path(__file__).resolve().parents[1] / "practice" / "poi.sqlite"

LAT_M = 111_320
LON_M = LAT_M * math.cos(math.radians(47.23))

KINDS: tuple[str, ...] = ()


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(sql, params)]


def _one(sql: str, params: tuple = ()) -> dict | None:
    got = _rows(sql, params)
    return got[0] if got else None


def _table(items: list[dict], columns: tuple[str, ...]) -> str:
    """Табличка вместо JSON: те же данные, вдвое меньше токенов."""
    if not items:
        return "ничего не нашлось"
    head = " | ".join(columns)
    body = "\n".join(
        " | ".join(str(it.get(c)) if it.get(c) is not None else "—" for c in columns)
        for it in items
    )
    return f"{head}\n{body}"


def _kind_error(kind: Any) -> str | None:
    if not isinstance(kind, str) or kind not in KINDS:
        return f"ошибка: вида места {kind!r} нет в базе; есть: {', '.join(KINDS)}"
    return None


# ══════════════════════════════════════════════════════════ инструменты

def count_places(kind: str, only_24_7: bool = False) -> str:
    bad = _kind_error(kind)
    if bad:
        return bad
    sql = "SELECT COUNT(*) AS n FROM poi WHERE kind = ?" + (" AND is_24_7 = 1" if only_24_7 else "")
    n = _one(sql, (kind,))["n"]
    return f"{kind}: {n} шт."


def find_places(kind: str, street: str | None = None, limit: int = 10) -> str:
    bad = _kind_error(kind)
    if bad:
        return bad
    sql = "SELECT id, name, street, housenumber, opening_hours FROM poi WHERE kind = ?"
    params: list[Any] = [kind]
    if street:
        sql += " AND street LIKE ?"
        params.append(f"%{street}%")
    try:
        limit = max(1, min(int(limit), 25))
    except (TypeError, ValueError):
        limit = 10
    got = _rows(sql + " LIMIT ?", (*params, limit))
    return _table(got, ("id", "name", "street", "housenumber", "opening_hours"))


def top_brands(kind: str, limit: int = 5) -> str:
    bad = _kind_error(kind)
    if bad:
        return bad
    try:
        limit = max(1, min(int(limit), 25))
    except (TypeError, ValueError):
        limit = 5
    got = _rows(
        "SELECT brand, COUNT(*) AS n FROM poi WHERE kind = ? AND brand IS NOT NULL "
        "GROUP BY brand ORDER BY n DESC LIMIT ?",
        (kind, limit),
    )
    return _table(got, ("brand", "n"))


def nearest(lat: float, lon: float, kind: str, limit: int = 1) -> str:
    bad = _kind_error(kind)
    if bad:
        return bad
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return "ошибка: lat и lon должны быть числами в градусах, например 47.2685"
    try:
        limit = max(1, min(int(limit), 5))
    except (TypeError, ValueError):
        limit = 1
    got = _rows("SELECT id, name, lat, lon FROM poi WHERE kind = ?", (kind,))
    for r in got:
        r["distance_m"] = int(math.hypot((r["lat"] - lat) * LAT_M, (r["lon"] - lon) * LON_M))
    got.sort(key=lambda r: r["distance_m"])
    if not got:
        return f"мест вида {kind} в базе нет"
    return "\n".join(
        f"id={r['id']} {r['name'] or 'без названия'} — {r['distance_m']} м"
        for r in got[:limit]
    )


def details(place_id: int) -> str:
    try:
        place_id = int(place_id)
    except (TypeError, ValueError):
        return "ошибка: place_id должен быть целым числом из ответа nearest, например 43"
    card = _one(
        "SELECT name, street, housenumber, opening_hours, phone FROM poi WHERE id = ?",
        (place_id,),
    )
    if card is None:
        return f"ошибка: места с id={place_id} в базе нет; id берутся из ответа nearest"
    address = ", ".join(p for p in (card["street"], card["housenumber"]) if p) or "адрес не указан"
    return (
        f"{card['name'] or 'без названия'} · часы: {card['opening_hours'] or '—'} · "
        f"адрес: {address} · телефон: {card['phone'] or '—'}"
    )


TOOLS: dict[str, Callable[..., str]] = {
    "count_places": count_places,
    "find_places": find_places,
    "top_brands": top_brands,
    "nearest": nearest,
    "details": details,
}


# ═══════════════════════════════════════════════════ описания: short / full

SHORT: dict[str, str] = {
    "count_places": "Сколько в городе мест такого вида.",
    "find_places": "Список мест города такого вида, по желанию на одной улице.",
    "top_brands": "Самые частые бренды мест такого вида.",
    "nearest": "Ближайшие к точке с координатами места такого вида.",
    "details": "Карточка места по id: адрес, часы, телефон.",
}

SHORT_PARAMS: dict[str, dict[str, str]] = {
    "count_places": {"kind": "вид места", "only_24_7": "только круглосуточные"},
    "find_places": {"kind": "вид места", "street": "улица", "limit": "лимит строк"},
    "top_brands": {"kind": "вид места", "limit": "сколько брендов"},
    "nearest": {"lat": "широта", "lon": "долгота", "kind": "вид места", "limit": "сколько мест"},
    "details": {"place_id": "id места"},
}

FULL: dict[str, str] = {
    "count_places": (
        "Считает места города заданного вида и возвращает одно целое число.\n\n"
        "Зови на вопросы со словами «сколько», «какое количество»: «Сколько в Ростове аптек?», "
        "«Сколько круглосуточных заправок?».\n\n"
        "Не зови, когда нужен список мест с адресами и часами — это find_places; "
        "когда просят бренды или сети — это top_brands; "
        "когда в вопросе «ближайший» или координаты — это nearest.\n\n"
        "Результат: строка вида «pharmacy: 330 шт.», единица — штук (мест в базе Ростова).\n\n"
        "Пример: «Сколько круглосуточных аптек?» → count_places(kind=\"pharmacy\", only_24_7=True)."
    ),
    "find_places": (
        "Возвращает список мест города заданного вида таблицей: id, название, адрес, часы работы.\n\n"
        "Зови, когда просят перечень или показать: «какие аптеки есть», «покажи кафе на Пушкинской», "
        "«какие заправки в городе».\n\n"
        "Не зови на вопросы «сколько» — для них count_places; "
        "про сети и бренды — top_brands; "
        "про близость к точке с координатами («ближе всего», «сколько метров до») — nearest.\n\n"
        "Результат: строки «id | название | улица | дом | часы», не больше limit строк; "
        "если подходящих мест нет — «ничего не нашлось».\n\n"
        "Пример: «Какие кафе есть на Пушкинской улице?» → find_places(kind=\"cafe\", street=\"Пушкин\")."
    ),
    "top_brands": (
        "Считает бренды мест заданного вида и возвращает таблицу «бренд | количество» "
        "по убыванию количества.\n\n"
        "Зови на вопросы про сети и бренды: «какие сети представлены в городе», "
        "«самый частый бренд», «какая сеть преобладает».\n\n"
        "Не зови для списков конкретных мест с адресами — это find_places; "
        "для простого подсчёта «сколько всего» — count_places.\n\n"
        "Результат: строки «brand | n», n — число мест этого бренда в базе; "
        "места без бренда в результат не попадают.\n\n"
        "Пример: «Какие сети пунктов выдачи чаще встречаются?» → top_brands(kind=\"outpost\") → "
        "первая строка «Ozon | 183»."
    ),
    "nearest": (
        "Ищет места заданного вида, ближайшие к точке с координатами, "
        "и возвращает их с расстоянием до точки.\n\n"
        "Зови, когда в вопросе есть точка или привязка к месту: «ближе всего к 47.2357, 39.7015», "
        "«ближайшая аптека отсюда», «сколько метров до пункта выдачи».\n\n"
        "Не зови для перечней мест по городу или улице — у тебя нет координат, это find_places; "
        "для подсчёта — count_places.\n\n"
        "Результат: строки «id=… название — N м», первая строка — самое ближнее место; "
        "N — расстояние по прямой, целые метры.\n\n"
        "Пример: «Какая ближайшая школа к 47.2685, 39.7875?» → "
        "nearest(lat=47.2685, lon=39.7875, kind=\"school\")."
    ),
    "details": (
        "Возвращает карточку места по его id: название, часы работы, адрес, телефон.\n\n"
        "Зови после nearest, когда про конкретное место спрашивают «до скольки работает», "
        "«где находится», «какой телефон».\n\n"
        "Не зови без id из ответа nearest — придуманный id бесполезен; "
        "для списков мест есть find_places.\n\n"
        "Результат: строка «Название · часы: … · адрес: … · телефон: …».\n\n"
        "Пример: nearest вернул «id=43 Радуга — 79 м» → details(place_id=43) → "
        "«Радуга · часы: Mo-Su 08:00-21:00 · адрес: …»."
    ),
}

FULL_PARAMS: dict[str, dict[str, str]] = {
    "count_places": {
        "kind": (
            "вид места латиницей, ровно значение из базы: pharmacy, cafe, school, fuel, "
            "outpost, supermarket, bank, restaurant, fast_food, convenience, kindergarten, "
            "playground, park; не переводи на русский"
        ),
        "only_24_7": (
            "True только когда прямо спрашивают про круглосуточные "
            "(«круглосуточно», «24/7», «работает ночью»); иначе False"
        ),
    },
    "find_places": {
        "kind": "вид места латиницей, значение из базы: pharmacy, cafe, fuel, outpost и т.д.",
        "street": (
            "часть названия улицы кириллицей, как в вопросе: «Пушкин» для Пушкинской улицы; "
            "None — если улицы в вопросе нет"
        ),
        "limit": "сколько строк вернуть максимум, по умолчанию 10",
    },
    "top_brands": {
        "kind": "вид места латиницей, значение из базы",
        "limit": "сколько первых брендов вернуть, по умолчанию 5",
    },
    "nearest": {
        "lat": "широта точки в градусах, десятичная, например 47.2685 — ровно число из вопроса",
        "lon": "долгота точки в градусах, например 39.7875 — ровно число из вопроса",
        "kind": "вид места латиницей, значение из базы",
        "limit": "сколько ближайших мест вернуть, по умолчанию 1",
    },
    "details": {
        "place_id": (
            "числовой id места — ровно то, что вернул nearest в начале строки «id=…»"
        ),
    },
}

REQUIRED: dict[str, list[str]] = {
    "count_places": ["kind"],
    "find_places": ["kind"],
    "top_brands": ["kind"],
    "nearest": ["lat", "lon", "kind"],
    "details": ["place_id"],
}

PARAM_TYPES: dict[str, dict[str, str]] = {
    "count_places": {"kind": "string", "only_24_7": "boolean"},
    "find_places": {"kind": "string", "street": ["string", "null"], "limit": "integer"},
    "top_brands": {"kind": "string", "limit": "integer"},
    "nearest": {"lat": "number", "lon": "number", "kind": "string", "limit": "integer"},
    "details": {"place_id": "integer"},
}


def schemas(variant: str) -> list[dict]:
    """JSON-схемы инструментов для OpenAI-совместимого chat/completions.

    variant: "short" | "full" — меняются только описания, структура одна.
    """
    if variant not in ("short", "full"):
        raise ValueError(f"неизвестный вариант описаний: {variant}")
    texts = SHORT if variant == "short" else FULL
    params_texts = SHORT_PARAMS if variant == "short" else FULL_PARAMS
    out = []
    for name, func in TOOLS.items():
        properties = {}
        for pname, ptype in PARAM_TYPES[name].items():
            prop: dict[str, Any] = {"type": ptype, "description": params_texts[name][pname]}
            properties[pname] = prop
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": texts[name],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": REQUIRED[name],
                },
            },
        })
    return out


def _load_kinds() -> tuple[str, ...]:
    return tuple(r["kind"] for r in _rows("SELECT DISTINCT kind FROM poi ORDER BY kind"))


KINDS = _load_kinds()
