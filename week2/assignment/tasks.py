"""Десять задач для check.py: вопрос, верный инструмент, верные аргументы и
судья финального ответа.

Ground truth считается из poi.sqlite без модели — судья сравнивает финальный
ответ модели с базой, а не с эталонным текстом модели. expected_tool и
expected_args нужны для A/B-метрики («выбран верный инструмент и аргументы»)
и для классификации ошибок в отчёте.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DB = Path(__file__).resolve().parents[1] / "practice" / "poi.sqlite"

LAT_M = 111_320
LON_M = LAT_M * math.cos(math.radians(47.23))

LON_SPOT_SCHOOL = (47.2685, 39.7875)    # ближайшая школа «Музыкальная», 79 м
LON_SPOT_CHAIN = (47.2076, 39.6258)     # ближайшая аптека «Радуга», 79 м, до 21:00


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(sql, params)]


def _one(sql: str, params: tuple = ()) -> dict:
    return _rows(sql, params)[0]


def norm(text: Any) -> str:
    """Нормализация для сравнения ответа с базой: регистр, ё, кавычки, пробелы."""
    s = str(text or "").lower().replace("ё", "е")
    for ch in '"“”«»\'':
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip()


def _count(kind: str, only_24_7: bool = False) -> int:
    sql = "SELECT COUNT(*) AS n FROM poi WHERE kind = ?" + (" AND is_24_7 = 1" if only_24_7 else "")
    return _one(sql, (kind,))["n"]


def _nearest(lat: float, lon: float, kind: str) -> dict:
    got = _rows("SELECT id, name, lat, lon, opening_hours FROM poi WHERE kind = ?", (kind,))
    for r in got:
        r["distance_m"] = int(math.hypot((r["lat"] - lat) * LAT_M, (r["lon"] - lon) * LON_M))
    return sorted(got, key=lambda r: r["distance_m"])[0]


@dataclass
class Task:
    id: str
    question: str
    tool: str                 # ожидаемый первый вызов
    args: dict[str, Any]      # ожидаемые аргументы первого вызова
    truth: str                # эталон из базы (для отчёта и отладки)
    judge: Callable[[str], bool]  # финальный ответ модели верный?


# ═══════════════════════════════════════════════════════ судьи ответов

def _judge_count(n: int) -> Callable[[str], bool]:
    def judge(answer: str) -> bool:
        return str(n) in (answer or "")
    return judge


def _judge_names(answer_norm_fn: Callable[[], list[str]], *need: str) -> Callable[[str], bool]:
    """Ответ верен, если в нём есть хотя бы одно имя из базы (и все `need`)."""
    def judge(answer: str) -> bool:
        a = norm(answer)
        if any(w not in a for w in (norm(x) for x in need)):
            return False
        names = [norm(x) for x in answer_norm_fn() if x]
        return any(name in a for name in names)
    return judge


def _judge_school(name: str, distance_m: int) -> Callable[[str], bool]:
    def judge(answer: str) -> bool:
        a = norm(answer)
        return norm(name) in a and str(distance_m) in a
    return judge


def _judge_hours(name: str, hours: str) -> Callable[[str], bool]:
    """«до скольки работает»: верным считаем конец времени работы из базы."""
    times = re.findall(r"\d{1,2}:\d{2}", hours or "")
    end = times[-1] if times else None

    def judge(answer: str) -> bool:
        a = norm(answer)
        if end and end in a:
            return True
        # «Радуга работает до 21» — часы без минут, но только вместе с названием
        hour_only = end.split(":")[0].lstrip("0") if end else None
        return bool(hour_only) and norm(name) in a and re.search(rf"\b{hour_only}\b", a) is not None
    return judge


# ═══════════════════════════════════════════════════════ десять задач

def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    n_pharmacy = _count("pharmacy")
    tasks.append(Task(
        id="count_pharmacy",
        question="Сколько в Ростове-на-Дону аптек?",
        tool="count_places",
        args={"kind": "pharmacy", "only_24_7": False},
        truth=str(n_pharmacy),
        judge=_judge_count(n_pharmacy),
    ))

    n_24_7 = _count("pharmacy", only_24_7=True)
    tasks.append(Task(
        id="count_24_7",
        question="Сколько в Ростове-на-Дону круглосуточных аптек?",
        tool="count_places",
        args={"kind": "pharmacy", "only_24_7": True},
        truth=str(n_24_7),
        judge=_judge_count(n_24_7),
    ))

    n_school = _count("school")
    tasks.append(Task(
        id="count_school",
        question="Сколько в Ростове-на-Дону школ?",
        tool="count_places",
        args={"kind": "school", "only_24_7": False},
        truth=str(n_school),
        judge=_judge_count(n_school),
    ))

    cafes = _rows(
        "SELECT name FROM poi WHERE kind = 'cafe' AND street LIKE '%Пушкин%' LIMIT 10"
    )
    tasks.append(Task(
        id="find_cafe_pushinskaya",
        question="Какие кафе есть на Пушкинской улице в Ростове?",
        tool="find_places",
        args={"kind": "cafe", "street": "Пушкин"},
        truth=", ".join(r["name"] for r in cafes),
        judge=_judge_names(lambda: [r["name"] for r in cafes]),
    ))

    pharmacies = _rows(
        "SELECT name FROM poi WHERE kind = 'pharmacy' AND street = 'Ростовская улица' LIMIT 10"
    )
    tasks.append(Task(
        id="find_pharmacy_rostovskaya",
        question="Какие аптеки есть на Ростовской улице в Ростове?",
        tool="find_places",
        args={"kind": "pharmacy", "street": "Ростовск"},
        truth=", ".join(r["name"] for r in pharmacies),
        judge=_judge_names(lambda: [r["name"] for r in pharmacies]),
    ))

    outposts = _rows("SELECT name FROM poi WHERE kind = 'outpost' LIMIT 10")
    tasks.append(Task(
        id="find_outposts",
        question="Покажи пункты выдачи заказов в Ростове.",
        tool="find_places",
        args={"kind": "outpost"},
        truth=", ".join(dict.fromkeys(r["name"] for r in outposts if r["name"])),
        judge=_judge_names(lambda: [r["name"] for r in outposts]),
    ))

    brand_outpost = _one(
        "SELECT brand, COUNT(*) AS n FROM poi WHERE kind = 'outpost' AND brand IS NOT NULL "
        "GROUP BY brand ORDER BY n DESC LIMIT 1"
    )
    tasks.append(Task(
        id="brands_outpost",
        question="Какие сети пунктов выдачи чаще всего встречаются в Ростове?",
        tool="top_brands",
        args={"kind": "outpost"},
        truth=f"{brand_outpost['brand']} ({brand_outpost['n']})",
        judge=lambda a, b=norm(brand_outpost["brand"]): b in norm(a),
    ))

    brand_fuel = _one(
        "SELECT brand, COUNT(*) AS n FROM poi WHERE kind = 'fuel' AND brand IS NOT NULL "
        "GROUP BY brand ORDER BY n DESC LIMIT 1"
    )
    tasks.append(Task(
        id="brands_fuel",
        question="Какие сети заправок чаще всего встречаются в Ростове?",
        tool="top_brands",
        args={"kind": "fuel"},
        truth=f"{brand_fuel['brand']} ({brand_fuel['n']})",
        judge=lambda a, b=norm(brand_fuel["brand"]): b in norm(a),
    ))

    school = _nearest(*LON_SPOT_SCHOOL, "school")
    tasks.append(Task(
        id="nearest_school",
        question=(f"Какая ближайшая школа к точке {LON_SPOT_SCHOOL[0]}, {LON_SPOT_SCHOOL[1]} "
                  f"и сколько до неё метров?"),
        tool="nearest",
        args={"kind": "school", "lat": LON_SPOT_SCHOOL[0], "lon": LON_SPOT_SCHOOL[1]},
        truth=f"{school['name']} — {school['distance_m']} м",
        judge=_judge_school(school["name"], school["distance_m"]),
    ))

    pharmacy = _nearest(*LON_SPOT_CHAIN, "pharmacy")
    hours = pharmacy["opening_hours"] or ""
    tasks.append(Task(
        id="chain_hours",
        question=(f"Где ближайшая аптека к точке {LON_SPOT_CHAIN[0]}, {LON_SPOT_CHAIN[1]} "
                  f"и до скольки она работает?"),
        tool="nearest",
        args={"kind": "pharmacy", "lat": LON_SPOT_CHAIN[0], "lon": LON_SPOT_CHAIN[1]},
        truth=f"{pharmacy['name']} — {hours}",
        judge=_judge_hours(pharmacy["name"], hours),
    ))

    return tasks


TASKS: list[Task] = build_tasks()


# ═══════════════════════════════════════════════ сверка вызова с ожиданием

def args_match(expected: dict[str, Any], called: dict[str, Any]) -> bool:
    """Совпадают ли аргументы вызова с ожидаемыми.

    street — по подстроке (вопрос «на Пушкинской», вызов street="Пушкинская улица");
    lat/lon — с допуском 1e-3 градуса (~100 м), модель округляет;
    only_24_7 — строго по булеву значению (отсутствие = False).
    """
    for key, want in expected.items():
        got = called.get(key)
        if key == "street":
            if norm(want) not in norm(got or ""):
                return False
        elif key in ("lat", "lon"):
            try:
                if abs(float(got) - float(want)) > 1e-3:
                    return False
            except (TypeError, ValueError):
                return False
        elif key == "only_24_7":
            if bool(got) != bool(want):
                return False
        elif key == "kind":
            if norm(got) != norm(want):
                return False
        else:
            if got != want:
                return False
    return True


def first_call_ok(task: Task, calls: list[dict]) -> bool:
    """Метрика A/B: первый вызов — верный инструмент и верные аргументы."""
    if not calls:
        return False
    first = calls[0]
    return first.get("name") == task.tool and args_match(task.args, first.get("args") or {})
