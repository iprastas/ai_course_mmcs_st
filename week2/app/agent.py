"""ADK-агент «Городская справочная»: иерархия coordinator → researcher + LiteLlm.

Здесь же — маппинг событий ADK в события для SSE-фида фронтенда:

    start | wait | thinking | delegate | tool_call | tool_result |
    map_points | answer_delta | final | usage | error

События агентов несут поле agent (coordinator / researcher) — UI показывает
бейдж, кто думает; delegate — момент transfer между агентами.

Контекст диалога ведёт ADK-сессия в SQLite (DatabaseSessionService) —
переживает рестарт сервера; у каждого пользователя своя сессия.
Каждый прогон дополнительно пишется строкой в runs/web_ГГГГММДД.jsonl
(в том числе при отмене клиентом — журнал в finally).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
import warnings
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv

load_dotenv()

# тишина в консоли: OTel-трассировку ADK выключаем, экспериментальные warnings прячем
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
warnings.filterwarnings("ignore", message=r".*JSON_SCHEMA_FOR_FUNC_DECL.*")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.agents.run_config import RunConfig, StreamingMode, ToolThreadPoolConfig  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import DatabaseSessionService  # noqa: E402
from google.adk.sessions.base_session_service import GetSessionConfig  # noqa: E402
from google.genai import types  # noqa: E402

from week2.assignment import tools as city  # noqa: E402
from week2.app.semantic import POI_DB, semantic_search  # noqa: E402

APP_NAME = "rostov"
COORDINATOR_NAME = "coordinator"   # корень дерева: делегирует и пишет финал
RESEARCHER_NAME = "researcher"     # sub-agent: работает только инструментами
_OUR_AGENTS = frozenset({COORDINATOR_NAME, RESEARCHER_NAME})
AGENT_NAME = COORDINATOR_NAME      # обратная совместимость (тесты, старые ссылки)
MAX_LLM_CALLS = 16          # предохранитель (transfer между агентами ест вызовы)
TEXT_LIMIT = 4000           # обрезка наблюдений в ленте
THINK_FLUSH = 160           # копим мысли-дельты, прежде чем отправлять в SSE
ANSWER_FLUSH = 64           # финальный ответ стримим чаще — он виден юзеру
MAX_POINTS = 60             # не больше столько точек на один прогон карты
MAX_CONTEXT_EVENTS = 40     # хвост истории, который видит модель (см. _run_config)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RUNS_DIR = HERE / "runs"

# ── иерархия агентов: coordinator (без тулов) → researcher (все 6 тулов) ──
# Разделение ответственности: исследователь не сочиняет ответы, координатор
# не лезет в инструменты. В SSE видно, кто думает, и момент делегирования.

COORDINATOR_INSTRUCTION = (
    "Ты — координатор городской справочной Ростова-на-Дону. "
    "Своих инструментов у тебя нет: факты собирает sub-agent researcher. "
    "Порядок работы: 1) сразу передавай задачу researcher (transfer), "
    "чётко сформулировав, что найти (вид места, улица, координаты, «по смыслу» — "
    "если пользователь забыл название, прямо скажи researcher звать semantic_search); "
    "2) получив его факты, дай пользователю краткий итог по-русски: "
    "цифры, адреса и часы — только из фактов researcher, ничего не выдумывай. "
    "Если researcher вернул ошибку — поправь задачу и передай повторно "
    "(не больше двух попыток на инструмент). "
    "Не пиши «вызову инструмент» текстом — у тебя их есть только через transfer. "
    "Отвечай коротко и по делу."
)

RESEARCHER_INSTRUCTION = (
    "Ты — исследователь справочной по базе мест Ростова-на-Дону "
    "(8860 мест из OpenStreetMap). Работаешь только инструментами: "
    "сначала вызови подходящий, затем верни координатору краткие факты по его "
    "результату. Числа, адреса и часы не выдумывай — бери из ответов инструментов. "
    "Если пользователь описывает место по смыслу, а не называет его точно "
    "(забыл название, вспоминает блюдо — «пельмени», «где выпить кофе») — "
    "зови semantic_search, а для адреса и часов найденного места — details по его id. "
    "Из кандидатов semantic_search выбирай только подходящие по смыслу и отметь "
    "их на карте пользователя: pin_places(place_ids=\"…\") — мимо запроса "
    "(парфюмерия, цветы, алкоголь и т.п.) не отмечай. "
    "Если инструмент вернул ошибку, поправь аргументы и вызови его снова. "
    "Не перебирай вызовы зря: маловато данных — честно перечисли, чего не хватает. "
    "Позвать инструмент можно только через механизм tool_calls: не пиши "
    "«count_places(...)» текстом — этого недостаточно, вызов не произойдёт. "
    "Возвращай по-русски, коротко и по делу."
)


def _docstring(name: str) -> str:
    """Описание инструмента из варианта full (как в A/B задания) + Args."""
    args = "\n".join(f"    {p}: {d}" for p, d in city.FULL_PARAMS[name].items())
    return f"{city.FULL[name]}\n\nArgs:\n{args}"


def _install_docstrings() -> None:
    # У инструментов задания описания жили в JSON-схемах, а ADK читает
    # докстроки — подставляем тот же текст full-варианта.
    for name in city.TOOLS:
        setattr(city.TOOLS[name], "__doc__", _docstring(name))


_install_docstrings()


def _model() -> LiteLlm:
    return LiteLlm(
        model=f"openai/{os.environ['MODEL']}",
        api_base=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        num_retries=2,           # лёгкий ретрай на уровне клиента модели
        temperature=0.2,         # ниже — реже «ленивые» ответы без вызова
        timeout=60.0,            # один запрос к модели — не дольше минуты
    )


def _enable_wal(db_file: Path) -> None:
    """WAL для ADK-сессий: параллельные пользователи без «database is locked».

    journal_mode=WAL — постоянный флаг файла; busy_timeout задаётся
    connect_args в get_runner (на каждое соединение пула SQLAlchemy).
    """
    try:
        con = sqlite3.connect(db_file, timeout=5.0)
        try:
            con.execute("PRAGMA journal_mode=WAL")
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — не критично: просто останемся в delete
        pass


def _build_agent() -> LlmAgent:
    """Иерархия: coordinator (корень, без тулов) → researcher (все 6 тулов).

    ADK сам выдаёт coordinator'у инструмент transfer_to_agent; событие
    передачи ловится в event_to_sse по ev.actions.transfer_to_agent.
    """
    researcher = LlmAgent(
        name=RESEARCHER_NAME,
        model=_model(),
        instruction=RESEARCHER_INSTRUCTION,
        description="Собирает факты о местах Ростова инструментами",
        tools=[city.count_places, city.find_places, city.top_brands,
               city.nearest, city.details, semantic_search, pin_places],
    )
    return LlmAgent(
        name=COORDINATOR_NAME,
        model=_model(),
        instruction=COORDINATOR_INSTRUCTION,
        description="Координатор: делегирует исследование и формулирует ответ",
        sub_agents=[researcher],
    )


def get_runner() -> Runner:
    """Раннер-синглтон: агент + сессии ADK в SQLite (переживают рестарт)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_file = DATA_DIR / "sessions.db"
    _enable_wal(db_file)
    db_url = f"sqlite+aiosqlite:///{db_file.as_posix()}"
    return Runner(
        agent=_build_agent(),
        app_name=APP_NAME,
        session_service=DatabaseSessionService(
            db_url=db_url,
            connect_args={"timeout": 15.0},   # busy_timeout на соединение пула
        ),
    )


def _run_config() -> RunConfig:
    """Конфиг одного прогона: лимит контекста модели + offload sync-тулов."""
    return RunConfig(
        streaming_mode=StreamingMode.SSE,
        max_llm_calls=MAX_LLM_CALLS,
        # «учёт контекста»: Runner грузит из БД только хвост истории —
        # последние MAX_CONTEXT_EVENTS событий (DBSessionService сам
        # делает ORDER BY timestamp DESC LIMIT n). UI читает сессию
        # без лимита — после F5 виден весь диалог.
        get_session_config=GetSessionConfig(num_recent_events=MAX_CONTEXT_EVENTS),
        # пять sync sqlite-тулов идут в thread pool, а не в event loop —
        # параллельные SSE-потоки не ждут чужие запросы к базе
        tool_thread_pool_config=ToolThreadPoolConfig(max_workers=4),
    )


_RUNNER = None


def runner() -> Runner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = get_runner()
    return _RUNNER


# ───────────────────────────────────── события → SSE

def _usage_counts(ev: Any) -> tuple[int, int]:
    um = getattr(ev, "usage_metadata", None)
    if um is None:
        return 0, 0
    inp = getattr(um, "prompt_token_count", None) or getattr(um, "input_token_count", 0) or 0
    out = getattr(um, "candidates_token_count", None) or getattr(um, "output_token_count", 0) or 0
    return int(inp), int(out)


def _event_text(ev: Any) -> str:
    """Весь текст события — и рассуждения (thought=True), и сам ответ."""
    if not (ev.content and ev.content.parts):
        return ""
    return "".join(p.text for p in ev.content.parts if getattr(p, "text", None))


def _answer_text(ev: Any) -> str:
    """Только часть ответа модели, без блоков рассуждений (thought)."""
    if not (ev.content and ev.content.parts):
        return ""
    parts = [p.text for p in ev.content.parts
             if getattr(p, "text", None) and not getattr(p, "thought", False)]
    if parts:
        return "".join(parts)
    # если провайдер не разделил thought — берём весь текст
    return _event_text(ev)


def _response_text(response: Any) -> str:
    """ADK оборачивает результат инструмента в {'result': …}; для ленты и карты
    нужен голый текст (иначе \n в repr становятся литеральными и таблицы ломаются)."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and isinstance(response.get("result"), str):
        return response["result"]
    return str(response)


def _looks_like_thought(ev: Any) -> bool:
    """True, если весь текст события — только рассуждения (thought=True)."""
    if not (ev.content and ev.content.parts):
        return False
    texts = [p.text for p in ev.content.parts if getattr(p, "text", None)]
    if not texts:
        return False
    thoughts = [p for p in ev.content.parts
                if getattr(p, "text", None) and getattr(p, "thought", False)]
    return len(thoughts) == len(texts)


def _transfer_target(ev: Any) -> str | None:
    """Куда это событие передаёт управление (multi-agent transfer), или None."""
    actions = getattr(ev, "actions", None)
    return getattr(actions, "transfer_to_agent", None) if actions else None


def event_to_sse(ev: Any) -> dict | None:
    """Событие ADK → событие для браузера (или None, если не показываем)."""
    # transfer — раньше responses: передача тоже приезжает с function_response
    # с именем transfer_to_agent, а нам нужен красивый event «делегирование»
    target = _transfer_target(ev)
    if target:
        return {"type": "delegate", "from": ev.author, "to": target}

    calls = ev.get_function_calls()
    responses = ev.get_function_responses()

    if responses:
        r = responses[0]
        if r.name == "transfer_to_agent":
            # transfer без actions (страховка) — всё равно не показываем как тул
            return None
        return {"type": "tool_result", "name": r.name,
                "result": _response_text(r.response)[:TEXT_LIMIT], "ok": True}

    if calls:
        fc = calls[0]
        # во время стриминга вызов приходит кусками (partial_args) —
        # показываем только собранный
        if getattr(fc, "will_continue", False) and getattr(fc, "partial_args", None):
            return None
        if not fc.args:
            return None
        return {"type": "tool_call", "id": fc.id, "name": fc.name,
                "args": {k: v for k, v in (fc.args or {}).items()}}

    text = _event_text(ev)
    if not text:
        return None
    if getattr(ev, "partial", False):
        # partial-реплика: это либо «мысль» (thought), либо уже ответ —
        # thought появляется отдельными part'ами; если провайдер не пометил
        # thought, отдаём как thinking (фронт склеивает), а финал соберём
        # из session.events ниже
        return {"type": "thinking", "text": text}
    # полный текст модели без вызовов — промежуточная или финальная реплика
    # не дублируем: фронт уже получил partial-куски, final отправит ask()
    return None


def _final_answer(session: Any) -> str:
    """Последняя реплика агента без вызовов — финальный ответ.

    Ищем по любому не-user автору: так же читаются старые сессии, где агент
    назывался иначе (rostov_guide), — иначе его ответы превращались в «вопросы».
    В иерархии финал пишет coordinator, но если transfer не случился и
    ответил researcher — его текст тоже валиден (ищем с конца по обоим).
    """
    for ev in reversed(session.events):
        if (ev.author != "user" and not ev.get_function_calls()
                and not ev.get_function_responses()):
            text = _answer_text(ev)
            if text:
                return text
    return ""


# ───────────────────────────────────── карта: координаты найденных мест

_ID_TABLE = re.compile(r"(?m)^\s*(\d+)\s*\|")     # первая колонка табличек: «1385 | Пельмень…»
_ID_EQ = re.compile(r"\bid=(\d+)")                # «id=43 Радуга — 79 м»


def _ids_from_sse(ev: dict) -> list[int]:
    """Какие id мест упомянуты в событии — только из «релевантных» источников.

    На карту попадают места, которые агент реально открыл или рекомендовал:
    аргументы details / pin_places и таблицы ответных тулов (find_places,
    nearest). Результат semantic_search — сырые кандидаты векторного поиска
    (среди них парфюмерия, цветы и прочий мимо запроса шум): агент отбрасывает
    их текстом, поэтому на карту они НЕ пинятся — иначе карта противоречит
    ответу («остальные совпадения нерелевантны», а точки стоят).
    """
    ids: list[int] = []
    name = ev.get("name") or ""
    if ev.get("type") == "tool_call":
        args = ev.get("args") or {}
        if name == "pin_places":
            raw = args.get("place_ids")
            if isinstance(raw, (list, tuple)):
                ids.extend(int(x) for x in raw
                           if str(x).isdigit())
            else:
                ids.extend(int(x) for x in re.findall(r"\d+", str(raw or "")))
        else:
            pid = args.get("place_id")
            if isinstance(pid, bool):
                pass
            elif isinstance(pid, int):
                ids.append(pid)
            elif isinstance(pid, str) and pid.isdigit():
                ids.append(int(pid))
    elif ev.get("type") == "tool_result" and name != "semantic_search":
        text = str(ev.get("result") or "")
        ids.extend(int(x) for x in _ID_TABLE.findall(text))
        ids.extend(int(x) for x in _ID_EQ.findall(text))
    return ids


def _coords(ids: list[int]) -> list[dict]:
    """Координаты мест из poi.sqlite для точек на карте."""
    ids = list(dict.fromkeys(int(i) for i in ids))[:MAX_POINTS]
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    with sqlite3.connect(POI_DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            f"SELECT id, name, kind, lat, lon FROM poi WHERE id IN ({marks})", ids
        ).fetchall()
    order = {i: n for n, i in enumerate(ids)}
    points = [{"id": r["id"], "name": r["name"] or "без названия", "kind": r["kind"],
               "lat": round(r["lat"], 5), "lon": round(r["lon"], 5)} for r in rows]
    points.sort(key=lambda p: order.get(p["id"], 0))
    return points


def collect_points(events: list[dict]) -> list[dict]:
    """Все точки из списка SSE-событий (для восстановления карты из истории)."""
    ids: list[int] = []
    for ev in events:
        ids.extend(_ids_from_sse(ev))
        if len(ids) >= MAX_POINTS:
            break
    return _coords(ids)


def pin_places(place_ids: str) -> str:
    """Отметить места на карте в интерфейсе пользователя.

    Зови с id мест, которые ты реально рекомендуешь по запросу (например,
    отобранных из кандидатов semantic_search): на карте окажутся только они
    и места, открытые тобой через details/nearest/find_places — сырые
    кандидаты поиска сами по себе не пинятся. id берутся из ответов
    инструментов (колонка «id» в таблице или строка «id=…»).

    Args:
        place_ids: id мест через запятую, например "1385, 750".
    """
    ids = list(dict.fromkeys(int(x) for x in re.findall(r"\d+", str(place_ids or ""))))[:MAX_POINTS]
    if not ids:
        return 'ошибка: укажи id мест через запятую, например "1385, 750"'
    marks = ",".join("?" * len(ids))
    with sqlite3.connect(POI_DB) as con:
        rows = con.execute(
            f"SELECT id, name FROM poi WHERE id IN ({marks})", ids
        ).fetchall()
    found = {r[0] for r in rows}
    names = ", ".join(f"{r[1] or 'без названия'} (id={r[0]})" for r in rows)
    out = f"отмечено на карте: {names}" if names else "ошибка: ни один id не найден в базе"
    missing = [i for i in ids if i not in found]
    if missing:
        out += f"; нет в базе: {missing}"
    return out


# ───────────────────────────────────── журнал прогонов (JSONL)

def _append_run(record: dict) -> None:
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNS_DIR / f"web_{time.strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — журнал не должен ронять ответ
        pass


# ───────────────────────────────────── публичный API

async def create_session(user_id: str) -> str:
    session = await runner().session_service.create_session(
        app_name=APP_NAME, user_id=user_id)
    return session.id


async def session_exists(user_id: str, session_id: str) -> bool:
    session = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    return session is not None


async def delete_session(user_id: str, session_id: str) -> None:
    """Удаляет сессию из ADK. Ошибки НЕ глотаем: вызывающий обязан узнать,
    что удаление не прошло (иначе чат «воскресал» в сайдбаре)."""
    await runner().session_service.delete_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)


async def list_user_sessions(user_id: str) -> list[dict]:
    """Все сессии пользователя для сайдбара: id + время последнего обновления."""
    try:
        resp = await runner().session_service.list_sessions(
            app_name=APP_NAME, user_id=user_id)
    except Exception:  # noqa: BLE001 — сайдбар не должен ронять запрос
        return []
    out: list[dict] = []
    for s in resp.sessions:
        t = s.last_update_time
        if hasattr(t, "timestamp"):          # datetime → unix
            t = t.timestamp()
        out.append({"id": s.id, "updated_at": float(t or 0)})
    out.sort(key=lambda x: -x["updated_at"])
    return out


async def session_is_empty(user_id: str, session_id: str) -> bool:
    """True, если в сессии нет ни одного события (пустой «призрачный» чат)."""
    session = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    return session is None or not session.events


async def first_user_text(user_id: str, session_id: str) -> str | None:
    """Первый вопрос пользователя в сессии — источник авто-названия чата."""
    session = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if session is None:
        return None
    for ev in session.events:
        if ev.author == "user":
            text = _event_text(ev)
            if text:
                return text
    return None


async def purge_empty_sessions() -> int:
    """Удаляет сессии без единого события (пустые призраки) и возвращает их число.

    Пустая сессия не несёт истории, поэтому удаление безопасно: хранить
    обязательно нужно именно диалоги. Вызывается при старте сервера.
    """
    resp = await runner().session_service.list_sessions(app_name=APP_NAME)
    removed = 0
    for s in resp.sessions:
        session = await runner().session_service.get_session(
            app_name=APP_NAME, user_id=s.user_id, session_id=s.id)
        if session is not None and not session.events:
            await delete_session(s.user_id, s.id)
            removed += 1
    return removed


# ───────────────────────────────────── follow-up вопросы после ответа

def _parse_suggestions(text: str) -> list[str]:
    """JSON-массив из ответа модели → до 3 коротких строк (fences допустимы)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for it in items if isinstance(items, list) else []:
        s = str(it).strip().strip('"').strip()
        if 3 <= len(s) <= 100:
            out.append(s)
        if len(out) == 3:
            break
    return out


async def suggest_followups(question: str, answer: str) -> list[str]:
    """3 вопроса-продолжения одним дешёвым вызовом (мимо агентского цикла)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=20.0,
    )
    prompt = (
        "Пользователь спросил про места Ростова-на-Дону, агент ответил. "
        "Придумай 3 коротких вопроса-продолжения по теме ответа: "
        "каждый до 60 символов, по-русски, конкретный (уточнение, "
        "альтернатива, детали). Ответь ТОЛЬКО JSON-массивом строк, "
        "без пояснений и markdown.\n\n"
        f"Вопрос пользователя: {question[:400]}\n"
        f"Ответ агента: {answer[:600]}"
    )
    resp = await client.chat.completions.create(
        model=os.environ["MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,   # у reasoning-моделей ход мыслей тоже ест max_tokens:
        temperature=0.7,  # при 200 пустой content → JSON не распарсится
    )
    return _parse_suggestions(resp.choices[0].message.content or "")


async def generate_title(question: str) -> str | None:
    """Короткое название чата по первому вопросу (один дешёвый LLM-вызов).

    Возвращает None при любой ошибке: в этом случае остаётся запасное
    название — обрезанный первый вопрос (его ставит main.touch_chat).
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=15.0,
    )
    prompt = (
        "Придумай короткое название чата (2–5 слов, до 40 символов) по первому "
        "вопросу пользователя городской справочной Ростова-на-Дону. "
        "Только само название, без кавычек, точки и пояснений.\n\n"
        f"Вопрос: {question[:300]}"
    )
    resp = await client.chat.completions.create(
        model=os.environ["MODEL"],
        messages=[{"role": "user", "content": prompt}],
        # reasoning-модель тратит часть max_tokens на ход мыслей — с маленьким
        # лимитом content остаётся пустым (finish_reason=length), поэтому
        # запас щедрый; сам title всё равно обрезается до 48 символов
        max_tokens=400,
        temperature=0.3,
    )
    title = " ".join((resp.choices[0].message.content or "").split())
    title = title.strip(" .\"'«»“”")
    return title[:48] if 3 <= len(title) else None


async def ask(user_id: str, session_id: str, question: str,
              queue_wait: float = 0.0) -> AsyncIterator[dict]:
    """Один прогон: поток событий для SSE (start → … → final → usage).

    Журнал пишется в finally — при отмене клиентом (Стоп / закрытие вкладки)
    генератор закрывается через GeneratorExit/CancelledError, и статус
    «cancelled» всё равно попадает в runs/*.jsonl. queue_wait — секунды
    ожидания свободного слота глобального семафора (0, если слот был сразу).
    """
    started = time.time()

    def pack(event: dict) -> dict:
        event["t"] = round(time.time() - started, 1)
        return event

    steps = 0
    in_tokens = out_tokens = 0
    status = "ok"
    calls: list[dict] = []
    point_ids: set[int] = set()
    final = ""
    journaled = False

    def journal() -> None:
        nonlocal journaled
        if journaled:
            return
        journaled = True
        price = in_tokens / 1e6 * float(os.getenv("PRICE_IN", "30")) \
            + out_tokens / 1e6 * float(os.getenv("PRICE_OUT", "120"))
        _append_run({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "web", "user": user_id, "session": session_id,
            "question": question, "status": status, "steps": steps,
            "input_tokens": in_tokens, "output_tokens": out_tokens,
            "cost_rub": round(price, 4), "seconds": round(time.time() - started, 1),
            "queue_seconds": round(queue_wait, 2),
            "calls": calls, "points": sorted(point_ids), "final": final,
        })

    msg = types.Content(role="user", parts=[types.Part(text=question)])
    run_config = _run_config()
    # авто-название чата по ПЕРВОМУ вопросу: LLM-вызов идёт фоном,
    # параллельно прогону, и готовое название уходит событием title
    title_task: asyncio.Task | None = None
    session0 = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if session0 is None or not any(ev.author == "user" for ev in session0.events):
        title_task = asyncio.ensure_future(generate_title(question))
    think_buf = ""
    think_agent = None           # кто думает в текущем think_buf (для бейджа)
    # дельты финального ответа (не-thought partial-текст после инструментов)
    answer_buf = ""
    in_answer_phase = False
    try:
        # start — тоже внутри try: иначе aclose() после первого yield
        # не дойдёт до finally и прогон не джурналится
        yield pack({"type": "start", "question": question})
        async for ev in runner().run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=msg,
            run_config=run_config,
        ):
            if ev.author in _OUR_AGENTS and not getattr(ev, "partial", False):
                steps += 1
                # собранный не-partial текст без вызовов — конец фазы ответа
                if not ev.get_function_calls() and not ev.get_function_responses():
                    full = _answer_text(ev)
                    if full and full != answer_buf:
                        # досылаем хвост, если дельты отставали от полного текста
                        tail = full[len(answer_buf):] if full.startswith(answer_buf) else full
                        if tail:
                            yield pack({"type": "answer_delta", "text": tail,
                                        "agent": ev.author})
                        answer_buf = full
            inp, out = _usage_counts(ev)
            in_tokens += inp
            out_tokens += out
            sse = event_to_sse(ev)
            if sse and sse["type"] == "thinking":
                # дельты стриминга копим, чтобы не сыпать в UI по слову
                chunk = sse["text"]
                think_buf += chunk
                think_agent = ev.author
                # после последнего tool_result начинается «финальная» фаза:
                # partial-текст без thought — это уже ответ модели
                if in_answer_phase and not _looks_like_thought(ev):
                    answer_buf += chunk
                    if len(answer_buf) >= ANSWER_FLUSH:
                        yield pack({"type": "answer_delta", "text": answer_buf,
                                    "agent": ev.author})
                        answer_buf = ""
                    continue
                if len(think_buf) >= THINK_FLUSH:
                    yield pack({"type": "thinking", "text": think_buf,
                                "agent": think_agent})
                    think_buf = ""
                continue
            if think_buf:
                yield pack({"type": "thinking", "text": think_buf,
                            "agent": think_agent})
                think_buf = ""
            if not sse:
                continue
            sse["agent"] = ev.author
            yield pack(sse)
            if sse["type"] == "tool_call":
                in_answer_phase = False
                answer_buf = ""
                calls.append({"name": sse["name"], "args": sse.get("args")})
            elif sse["type"] == "delegate":
                # передача управления: следующий текст — мысли нового агента
                in_answer_phase = False
                answer_buf = ""
                calls.append({"name": f"transfer→{sse['to']}",
                              "args": {"from": sse.get("from")}})
            elif sse["type"] == "tool_result":
                # после результата инструмента следующий текст — обычно ответ
                in_answer_phase = True
            if sse["type"] in ("tool_call", "tool_result"):
                fresh = [i for i in _ids_from_sse(sse) if i not in point_ids]
                if fresh:
                    point_ids.update(fresh)
                    points = _coords(fresh)
                    if points:
                        yield pack({"type": "map_points", "points": points})
    except (asyncio.CancelledError, GeneratorExit):
        status = "cancelled"
        journal()
        raise
    except Exception as exc:  # noqa: BLE001 — ошибка прогона не роняет соединение
        # важно: ловим ДО BaseException — иначе обычные ошибки уходили
        # в ветку cancelled, а событие error ниже было мёртвым кодом
        status = "error"
        journal()
        yield pack({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return
    except BaseException:  # noqa: BLE001 — KeyboardInterrupt и прочее
        status = "cancelled"
        journal()
        raise
    finally:
        if title_task is not None and not title_task.done():
            title_task.cancel()   # прогон прерван — фоновое название не ждём
        journal()   # no-op, если уже записали; иначе — последний шанс

    if think_buf:
        yield pack({"type": "thinking", "text": think_buf, "agent": think_agent})
    if answer_buf:
        yield pack({"type": "answer_delta", "text": answer_buf})
        answer_buf = ""

    session = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    final = _final_answer(session) if session else ""
    yield pack({"type": "final", "answer": final, "agent": COORDINATOR_NAME})
    if title_task is not None:
        # готовое название — приятный бонус, его сбой не должен валить прогон
        title = None
        try:
            await asyncio.wait({title_task}, timeout=8.0)
            if title_task.done() and not title_task.cancelled():
                exc = title_task.exception()
                if exc is None:
                    title = title_task.result()
        except asyncio.CancelledError:
            raise                      # гасят сам прогон — пробрасываем честно
        except Exception:              # noqa: BLE001
            title = None
        finally:
            if not title_task.done():
                title_task.cancel()
            title_task = None
        if title:
            yield pack({"type": "title", "title": title})
    price = in_tokens / 1e6 * float(os.getenv("PRICE_IN", "30")) \
        + out_tokens / 1e6 * float(os.getenv("PRICE_OUT", "120"))
    yield pack({"type": "usage", "steps": steps, "input_tokens": in_tokens,
                "output_tokens": out_tokens, "cost_rub": round(price, 4),
                "seconds": round(time.time() - started, 1)})
    status = "ok"
    journal()


async def history(user_id: str, session_id: str) -> list[dict]:
    """Контекст диалога для UI (последние события сессии, без partial-дублей)."""
    session = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if session is None:
        return []
    out: list[dict] = []
    answer = _final_answer(session)
    # читаем сессию без лимита get_session_config — в UI виден весь диалог,
    # хотя в контекст модели уходят только последние MAX_CONTEXT_EVENTS событий
    for ev in session.events:
        # вопрос — только реплика пользователя; всё остальное пишет агент
        # (включая старые сессии, где он назывался иначе — напр. rostov_guide)
        if ev.author == "user":
            text = _event_text(ev)
            if text:
                out.append({"type": "question", "text": text})
            continue
        target = _transfer_target(ev)
        if target:
            out.append({"type": "delegate", "from": ev.author, "to": target,
                        "agent": ev.author})
            continue
        calls, responses = ev.get_function_calls(), ev.get_function_responses()
        if responses:
            r = responses[0]
            if r.name == "transfer_to_agent":
                continue      # уже показали как delegate (или пропускаем дубль)
            out.append({"type": "tool_result", "name": r.name,
                        "result": _response_text(r.response)[:TEXT_LIMIT],
                        "ok": True, "agent": ev.author})
        elif calls and calls[0].args:
            out.append({"type": "tool_call", "id": calls[0].id,
                        "name": calls[0].name, "args": dict(calls[0].args),
                        "agent": ev.author})
        else:
            text = _event_text(ev)
            # финальный ответ не дублируем в ленте «как думает агент» —
            # в чат он попадает событием final ниже
            if text and text != answer:
                out.append({"type": "thinking", "text": text, "agent": ev.author})
    if answer:
        out.append({"type": "final", "answer": answer})
    points = collect_points(out)
    if points:
        out.append({"type": "map_points", "points": points})
    return out
