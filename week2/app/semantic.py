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
import asyncio
import math
import os
import sqlite3
import sys
import threading
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
POI_DB = HERE.parent / "practice" / "poi.sqlite"
VEC_DB = HERE / "vectors.sqlite"
EMBED_MODEL_DEFAULT = "text-embedding-3-small"
BATCH = 128
MIN_SCORE = 0.25          # ниже этого совпадения не показываем (шум)
INDEX_DIM = 1536          # размерность vec0-таблицы и meta['dim']

# LRU-кеш эмбеддингов: повторный запрос (дебаунс, второй юзер, restart
# поиска) не идёт в API. ~512 × 1536 float ≈ 3 МБ. Лок — т.к. пишут и
# build-поток, и event loop.
_EMBED_CACHE_MAX = 512
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_embed_lock = threading.Lock()
_embed_stats = {"hits": 0, "misses": 0}


def embed_cache_stats() -> dict[str, int]:
    with _embed_lock:
        return {"size": len(_embed_cache), **_embed_stats}


def _cache_get(text: str) -> list[float] | None:
    with _embed_lock:
        vec = _embed_cache.get(text)
        if vec is not None:
            _embed_cache.move_to_end(text)
            _embed_stats["hits"] += 1
        else:
            _embed_stats["misses"] += 1
        return vec


def _cache_put(text: str, vec: list[float]) -> list[float]:
    with _embed_lock:
        _embed_cache[text] = vec
        _embed_cache.move_to_end(text)
        while len(_embed_cache) > _EMBED_CACHE_MAX:
            _embed_cache.popitem(last=False)
    return vec


def _embed_model() -> str:
    return os.getenv("EMBED_MODEL", EMBED_MODEL_DEFAULT)


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=120.0,
    )


def _aclient():
    """Async-клиент для эмбеддингов: не блокирует event loop FastAPI."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=120.0,
    )


def _normalize(vec: list[float]) -> list[float]:
    """L2-нормализация: на единичных векторах косинус = обычному L2-расстоянию."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _embed(client, texts: list[str]) -> list[list[float]]:
    """Эмбеддинги с LRU-кешем: в API уезжают только промахи."""
    vecs: list[list[float] | None] = [_cache_get(t) for t in texts]
    missing = [i for i, v in enumerate(vecs) if v is None]
    if missing:
        resp = client.embeddings.create(model=_embed_model(),
                                        input=[texts[i] for i in missing])
        for i, d in zip(missing, resp.data):
            vecs[i] = _cache_put(texts[i], _normalize(d.embedding))
    return _require_all(vecs, len(texts))


async def _aembed(client, texts: list[str]) -> list[list[float]]:
    """То же async: промахи одним пакетом в AsyncOpenAI, кеш — общий."""
    vecs: list[list[float] | None] = [_cache_get(t) for t in texts]
    missing = [i for i, v in enumerate(vecs) if v is None]
    if missing:
        resp = await client.embeddings.create(model=_embed_model(),
                                              input=[texts[i] for i in missing])
        for i, d in zip(missing, resp.data):
            vecs[i] = _cache_put(texts[i], _normalize(d.embedding))
    return _require_all(vecs, len(texts))


def _require_all(vecs: list[list[float] | None], expected: int) -> list[list[float]]:
    if len(vecs) != expected or any(v is None for v in vecs):
        raise RuntimeError("провайдер эмбеддингов вернул меньше векторов, чем запросов")
    return [v for v in vecs if v is not None]


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
    """Строит индекс в VEC_DB (атомарно, через временный файл). Идемпотентен."""
    load_dotenv()
    if VEC_DB.exists() and not force and not is_stale():
        with sqlite3.connect(VEC_DB) as con:
            n = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if n:
            print(f"индекс актуален: {n} мест ({VEC_DB.name})")
            return n

    import sqlite_vec

    rows = _poi_rows()
    print(f"строк в базе: {len(rows)}; модель: {_embed_model()}")
    client = _client()

    tmp_db = VEC_DB.with_suffix(".sqlite.tmp")
    tmp_db.unlink(missing_ok=True)
    con = sqlite3.connect(tmp_db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.execute("CREATE TABLE docs(rowid INTEGER PRIMARY KEY, id INTEGER, name TEXT, "
                "kind TEXT, cuisine TEXT, addr TEXT)")
    # векторы L2-нормализованы, поэтому обычное расстояние = косинусу
    con.execute(f"CREATE VIRTUAL TABLE vecs USING vec0(embedding float[{INDEX_DIM}])")
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
        con.execute("INSERT INTO meta VALUES ('dim', ?)", (str(INDEX_DIM),))
        con.execute("INSERT INTO meta VALUES ('count', ?)", (str(len(rows)),))
        con.execute("INSERT INTO meta VALUES ('poi_mtime', ?)",
                    (str(int(POI_DB.stat().st_mtime)),))
    con.close()
    os.replace(tmp_db, VEC_DB)
    print(f"готово: {len(rows)} мест → {VEC_DB}")
    return len(rows)


def _meta() -> dict[str, str]:
    if not VEC_DB.exists():
        return {}
    try:
        with sqlite3.connect(VEC_DB) as con:
            return dict(con.execute("SELECT key, value FROM meta").fetchall())
    except Exception:  # noqa: BLE001 — битый файл → пересчитаем как stale
        return {}


def index_model_mismatch() -> str | None:
    """Причина, по которой текущий индекс непригоден для запросов, или None.

    Ловим два класса тихих ошибок: сменили EMBED_MODEL после сборки
    (эмбеддинги разных моделей несовместимы) и сменили размерность.
    """
    meta = _meta()
    if not meta:
        return None if not VEC_DB.exists() else "в meta нет данных"
    model = meta.get("model")
    if model and model != _embed_model():
        return f"индекс собран моделью {model}, сейчас EMBED_MODEL={_embed_model()}"
    dim = meta.get("dim")
    if dim and dim != str(INDEX_DIM):
        return f"индекс dim={dim}, ожидалось {INDEX_DIM}"
    return None


def is_stale() -> bool:
    """Индекс устарел, если менялся poi.sqlite или несовместима модель эмбеддингов."""
    if not VEC_DB.exists():
        return True
    if index_model_mismatch() is not None:
        return True
    try:
        with sqlite3.connect(VEC_DB) as con:
            row = con.execute("SELECT value FROM meta WHERE key='poi_mtime'").fetchone()
    except Exception:  # noqa: BLE001 — битый файл → пересобрать
        return True
    if row is None:
        return True
    return row[0] != str(int(POI_DB.stat().st_mtime))


def ensure_index() -> None:
    """Вызывается при старте сервера: нет индекса — строим; устарел — пересобираем.

    Пересборка идёт в фоновом потоке: старый индекс работает, пока новый
    не готов (файл меняется атомарно через os.replace).
    """
    if not VEC_DB.exists():
        print("[index] индекс не найден — строю…")
        build()
        return
    if is_stale():
        import threading

        print("[index] poi.sqlite изменился — пересобираю индекс в фоне…")
        threading.Thread(target=build, kwargs={"force": True}, daemon=True).start()


def _lookup(vec: list[float], limit: int) -> list[dict]:
    """KNN по уже готовому вектору запроса (общая часть sync/async top_k)."""
    import sqlite_vec

    blob = sqlite_vec.serialize_float32(vec)
    con = sqlite3.connect(VEC_DB)
    try:
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
        return out
    finally:
        con.close()


def _ensure_queryable() -> None:
    if not VEC_DB.exists():
        raise FileNotFoundError(f"нет индекса: {VEC_DB}")
    mismatch = index_model_mismatch()
    if mismatch:
        raise RuntimeError(mismatch)


def top_k(query: str, limit: int = 10) -> list[dict]:
    """Косинусный top-K по индексу. Чистая функция — тестируется без API."""
    _ensure_queryable()
    load_dotenv()
    vec = _embed(_client(), [query])[0]
    return _lookup(vec, limit)


async def atop_k(query: str, limit: int = 10) -> list[dict]:
    """То же, что top_k, но async: embedding-запрос не блокирует event loop."""
    _ensure_queryable()
    load_dotenv()
    vec = (await _aembed(_aclient(), [query]))[0]
    return await asyncio.to_thread(_lookup, vec, limit)


async def semantic_search(query: str, limit: int = 10) -> str:
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
        found = await atop_k(query, limit=limit)
    except FileNotFoundError:
        return ("ошибка: векторный индекс не построен; выполни "
                "`uv run python week2/app/semantic.py --build`")
    except RuntimeError as exc:
        return f"ошибка: {exc}; выполни `uv run python week2/app/semantic.py --force`"
    except Exception as exc:  # noqa: BLE001 — наблюдение вместо падения цикла
        return f"ошибка семантического поиска: {type(exc).__name__}: {exc}"
    # порог сходства: ниже MIN_SCORE считаём совпадением шумом
    found = [r for r in found if r["score"] >= MIN_SCORE]
    if not found:
        return "ничего похожего не нашлось (схожесть ниже порога)"
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
