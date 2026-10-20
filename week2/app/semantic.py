"""Семантический поиск по базе мест: sqlite-vec + эмбеддинги AITunnel.

Индекс строится один раз (≈70 запросов embeddings, копеечно):

    uv run python week2/app/semantic.py --build

Проверка юзкейса (слова «пельмени» в базе нет ни разу — найдет только семантика):

    uv run python week2/app/semantic.py --test пельмени

Инструмент semantic_search(query) отдаёт модели короткую табличку —
в стиле остальных инструментов задания.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
POI_DB = HERE.parent / "practice" / "poi.sqlite"
VEC_DB = HERE / "vectors.sqlite"
EMBED_MODEL_DEFAULT = "text-embedding-3-small"
BATCH = 128


def _embed_model() -> str:
    return os.getenv("EMBED_MODEL", EMBED_MODEL_DEFAULT)


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=120.0,
    )


def _normalize(vec: list[float]) -> list[float]:
    """L2-нормализация: на единичных векторах косинус = обычному L2-расстоянию."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _embed(client, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=_embed_model(), input=texts)
    return [_normalize(d.embedding) for d in resp.data]


def _doc(row: sqlite3.Row) -> str:
    """Текст документа места: то, что мы ищем по смыслу."""
    parts = [
        row["name"] or "",
        row["kind"] or "",
        f"cuisine: {row['cuisine']}" if row["cuisine"] else "",
        row["brand"] or "",
        row["street"] or "",
    ]
    return "; ".join(p for p in parts if p)


def _poi_rows() -> list[sqlite3.Row]:
    with sqlite3.connect(POI_DB) as con:
        con.row_factory = sqlite3.Row
        return con.execute(
            "SELECT id, name, kind, cuisine, brand, street, housenumber "
            "FROM poi WHERE name IS NOT NULL"
        ).fetchall()


def build(force: bool = False) -> int:
    """Строит vectors.sqlite. Идемпотентен: готовый индекс пропускается."""
    load_dotenv()
    if VEC_DB.exists() and not force:
        with sqlite3.connect(VEC_DB) as con:
            n = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if n:
            print(f"индекс уже построен: {n} мест ({VEC_DB.name})")
            return n

    import sqlite_vec

    rows = _poi_rows()
    print(f"строк в базе: {len(rows)}; модель: {_embed_model()}")
    client = _client()

    VEC_DB.unlink(missing_ok=True)
    con = sqlite3.connect(VEC_DB)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.execute("CREATE TABLE docs(rowid INTEGER PRIMARY KEY, id INTEGER, name TEXT, "
                "kind TEXT, cuisine TEXT, addr TEXT)")
    # векторы L2-нормализованы, поэтому обычное расстояние = косинусу
    con.execute("CREATE VIRTUAL TABLE vecs USING vec0(embedding float[1536])")
    con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")

    docs = [(r["id"], r["name"], r["kind"], r["cuisine"],
             ", ".join(p for p in (r["street"], r["housenumber"]) if p)) for r in rows]
    texts = [_doc(r) for r in rows]

    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        vecs = _embed(client, [texts[i] for i in range(start, start + len(chunk))])
        with con:
            for offset, (vec, doc) in enumerate(zip(vecs, docs[start : start + len(chunk)])):
                cur = con.execute(
                    "INSERT INTO docs(id, name, kind, cuisine, addr) VALUES (?,?,?,?,?)", doc)
                con.execute(
                    "INSERT INTO vecs(rowid, embedding) VALUES (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(vec)),
                )
        done = min(start + BATCH, len(rows))
        if done % (BATCH * 10) == 0 or done == len(rows):
            print(f"  заэмбеджено {done}/{len(rows)}")

    with con:
        con.execute("INSERT INTO meta VALUES ('model', ?)", (_embed_model(),))
        con.execute("INSERT INTO meta VALUES ('dim', '1536')")
        con.execute("INSERT INTO meta VALUES ('count', ?)", (str(len(rows)),))
    con.close()
    print(f"готово: {len(rows)} мест → {VEC_DB}")
    return len(rows)


def top_k(query: str, limit: int = 10) -> list[dict]:
    """Косинусный top-K по индексу. Чистая функция — тестируется без API."""
    if not VEC_DB.exists():
        raise FileNotFoundError(f"нет индекса: {VEC_DB}")
    import sqlite_vec

    load_dotenv()
    vec = _embed(_client(), [query])[0]
    blob = sqlite_vec.serialize_float32(vec)
    con = sqlite3.connect(VEC_DB)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    found = con.execute(
        "SELECT rowid, distance FROM vecs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (blob, limit),
    ).fetchall()
    out = []
    for rowid, distance in found:
        doc = con.execute(
            "SELECT id, name, kind, cuisine, addr FROM docs WHERE rowid = ?", (rowid,)
        ).fetchone()
        out.append({
            "id": doc[0], "name": doc[1], "kind": doc[2],
            "cuisine": doc[3], "addr": doc[4],
            # единичные векторы: cos = 1 - d²/2
            "score": round(1.0 - distance * distance / 2.0, 3),
        })
    con.close()
    return out


def semantic_search(query: str, limit: int = 10) -> str:
    """Инструмент для агента: места города, похожие по смыслу на запрос.

    Зови, когда человек ищет место по описанию, а не по точному названию:
    забыл название и говорит «пельмени», «где выпить кофе с завтраком»,
    «дорогой ресторан». Не зови для точных фильтров по виду и улице —
    для этого есть find_places, count_places и nearest.

    В ответе есть id места — для адреса, часов и телефона зови
    details(place_id=id) по самому интересному.

    Args:
        query: что ищем своими словами.
        limit: сколько мест вернуть, по умолчанию 10.
    """
    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = 10
    try:
        found = top_k(query, limit=limit)
    except FileNotFoundError:
        return ("ошибка: векторный индекс не построен; выполни "
                "`uv run python week2/app/semantic.py --build`")
    except Exception as exc:  # noqa: BLE001 — наблюдение вместо падения цикла
        return f"ошибка семантического поиска: {type(exc).__name__}: {exc}"
    if not found:
        return "ничего не нашлось"
    lines = ["id | название | вид | кухня | адрес | сходство"]
    for r in found:
        lines.append(" | ".join([
            str(r["id"]),
            str(r["name"] or "без названия"),
            str(r["kind"] or "—"),
            str(r["cuisine"] or "—"),
            str(r["addr"] or "адрес не указан"),
            str(r["score"]),
        ]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="Векторный индекс мест Ростова")
    parser.add_argument("--build", action="store_true", help="построить индекс")
    parser.add_argument("--force", action="store_true", help="перестроить принудительно")
    parser.add_argument("--test", metavar="ФРАЗА", help="проверить поиск фразой")
    args = parser.parse_args(argv)

    if args.build:
        build(force=args.force)
        return 0
    if args.test:
        for r in top_k(args.test, limit=10):
            print(f"  {r['score']:.3f}  {r['name']} ({r['kind']}"
                  + (f", {r['cuisine']}" if r["cuisine"] else "") + ")")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
