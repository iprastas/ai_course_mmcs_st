"""ADK-агент «Городская справочная»: LlmAgent + LiteLlm (наш провайдер).

Здесь же — маппинг событий ADK в события для SSE-фида фронтенда:

    start | thinking | tool_call | tool_result | final | usage | error

Контекст диалога ведёт ADK-сессия (InMemorySessionService): у каждого
пользователя своя, многопользовательность — из коробки.
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator

from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from week2.assignment import tools as city
from week2.app.semantic import semantic_search

APP_NAME = "rostov"
AGENT_NAME = "rostov_guide"
MAX_LLM_CALLS = 12          # предохранитель: максимум вызовов модели на прогон
TEXT_LIMIT = 4000           # обрезка наблюдений в ленте
THINK_FLUSH = 160           # копим мысли-дельты, прежде чем отправлять в SSE

INSTRUCTION = (
    "Ты — справочная по базе мест Ростова-на-Дону (8860 мест из OpenStreetMap). "
    "Отвечай на вопросы пользователя только с помощью инструментов: сначала вызови "
    "подходящий инструмент, затем дай краткий ответ по его результату. "
    "Числа, адреса и часы не выдумывай — бери только из ответов инструментов. "
    "Если пользователь описывает место по смыслу, а не называет его точно "
    "(забыл название, вспоминает блюдо или впечатление — «пельмени», "
    "«где выпить кофе», «дорогой ресторан») — зови semantic_search, а для "
    "адреса и часов найденного места — details по его id. "
    "Если инструмент вернул ошибку, поправь аргументы и вызови его снова. "
    "Не перебирай вызовы зря: получив результат, отвечай по нему; если данных "
    "мало — честно скажи, чего не хватает, вместо десятков попыток. "
    "Отвечай по-русски, коротко и по делу."
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
    )


def get_runner() -> Runner:
    """Синглтон-раннер: агент + сессионный сервис на всё приложение."""
    agent = LlmAgent(
        name=AGENT_NAME,
        model=_model(),
        instruction=INSTRUCTION,
        description="Справочная по базе мест Ростова",
        tools=[city.count_places, city.find_places, city.top_brands,
               city.nearest, city.details, semantic_search],
    )
    return Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=InMemorySessionService(),
    )


_RUNNER = None


def runner() -> Runner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = get_runner()
    return _RUNNER


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
    """ADK оборачивает результат инструмента в {'result': …}; для ленты
    нужен голый текст (иначе repr ломает таблицы)."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and isinstance(response.get("result"), str):
        return response["result"]
    return str(response)


def event_to_sse(ev: Any) -> dict | None:
    """Событие ADK → событие для браузера (или None, если не показываем)."""
    calls = ev.get_function_calls()
    responses = ev.get_function_responses()

    if responses:
        r = responses[0]
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
        return {"type": "thinking", "text": text}
    # полный текст модели без вызовов — промежуточная или финальная реплика;
    # в ленту потоком идут partial-куски, поэтому здесь не дублируем
    return None


def _final_answer(session: Any) -> str:
    """Последняя реплика модели без вызовов — финальный ответ (без мыслей)."""
    for ev in reversed(session.events):
        if ev.author == AGENT_NAME and not ev.get_function_calls():
            text = _answer_text(ev)
            if text:
                return text
    return ""


async def create_session(user_id: str) -> str:
    session = await runner().session_service.create_session(
        app_name=APP_NAME, user_id=user_id)
    return session.id


async def ask(user_id: str, session_id: str, question: str) -> AsyncIterator[dict]:
    """Один прогон: поток событий для SSE (start → … → final → usage)."""
    started = time.time()
    steps = 0
    in_tokens = out_tokens = 0
    yield {"type": "start", "question": question}

    msg = types.Content(role="user", parts=[types.Part(text=question)])
    run_config = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=MAX_LLM_CALLS)
    think_buf = ""
    try:
        async for ev in runner().run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=msg,
            run_config=run_config,
        ):
            if ev.author == AGENT_NAME and not getattr(ev, "partial", False):
                steps += 1
            inp, out = _usage_counts(ev)
            in_tokens += inp
            out_tokens += out
            sse = event_to_sse(ev)
            if sse and sse["type"] == "thinking":
                # дельты стриминга копим, чтобы не сыпать в UI по слову
                think_buf += sse["text"]
                if len(think_buf) >= THINK_FLUSH:
                    yield {"type": "thinking", "text": think_buf}
                    think_buf = ""
                continue
            if think_buf:
                yield {"type": "thinking", "text": think_buf}
                think_buf = ""
            if sse:
                yield sse
    except Exception as exc:  # noqa: BLE001 — ошибка прогона не роняет соединение
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return

    if think_buf:
        yield {"type": "thinking", "text": think_buf}

    session = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    yield {"type": "final", "answer": _final_answer(session)}
    price = in_tokens / 1e6 * float(os.getenv("PRICE_IN", "30")) \
        + out_tokens / 1e6 * float(os.getenv("PRICE_OUT", "120"))
    yield {"type": "usage", "steps": steps, "input_tokens": in_tokens,
           "output_tokens": out_tokens, "cost_rub": round(price, 4),
           "seconds": round(time.time() - started, 1)}


async def history(user_id: str, session_id: str) -> list[dict]:
    """Контекст диалога для UI (последние события сессии, без partial-дублей)."""
    session = await runner().session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if session is None:
        return []
    out: list[dict] = []
    for ev in session.events:
        if ev.author != AGENT_NAME:
            text = _event_text(ev)
            if text:
                out.append({"type": "question", "text": text})
            continue
        calls, responses = ev.get_function_calls(), ev.get_function_responses()
        if responses:
            r = responses[0]
            out.append({"type": "tool_result", "name": r.name,
                        "result": _response_text(r.response)[:TEXT_LIMIT], "ok": True})
        elif calls and calls[0].args:
            out.append({"type": "tool_call", "id": calls[0].id,
                        "name": calls[0].name, "args": dict(calls[0].args)})
        else:
            text = _event_text(ev)
            if text:
                out.append({"type": "thinking", "text": text})
    if session.events:
        answer = _final_answer(session)
        if answer:
            out.append({"type": "final", "answer": answer})
    return out
