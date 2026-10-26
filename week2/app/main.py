"""FastAPI-приложение «Городская справочная Ростова».

    uv run python -m uvicorn week2.app.main:app --port 8000

Роуты:
    GET  /                страница (форма вопросов)
    GET  /stats           страница метрик задания (pass@1, A/B)
    GET  /favicon.ico     иконка
    POST /login, /logout  вход/выход (кука HttpOnly, токен в SQLite, rate-limit)
    POST /api/register    регистрация (пароль хранится PBKDF2-хэшем в SQLite)
    POST /api/ask         SSE-поток: как агент думает и что вызывает (+ heartbeat)
    GET  /api/history     контекст диалога (восстановление после F5)
    GET  /api/search      быстрый семантический поиск (мимо агента, async)
    GET  /api/metrics     метрики из журналов задания
    GET  /api/health      живость сервиса, индекс, семафор, кеш эмбеддингов
    GET/POST/PATCH/DELETE /api/chats*  мультичаты: список, создание,
                           переименование, удаление, переключение
    POST /api/followups   3 вопроса-продолжения после ответа (дешёвый LLM-вызов)
    POST /login/demo      демо-вход одним кликом (DEMO_USER в .env)
    POST /api/reset       новый диалог = новая сессия контекста
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from week2.app import agent as agent_mod
from week2.app import auth
from week2.app import semantic
from week2.app import stats as stats_mod

APP_TITLE = "Городская справочная Ростова"
MAX_QUESTION = 2000
SSE_PING_SEC = 15.0              # heartbeat для прокси/таймаутов


def _fallback_title(question: str, limit: int = 48) -> str:
    """Запасное название чата: сам первый вопрос, без переносов и хвоста.

    LLM-название (короткий смысл вопроса) прилетит событием title и
    заменит этот текст — если пользователь ещё не переименовал чат сам."""
    text = " ".join(question.split())
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:!?")
    return (head or text[:limit]) + "…"


# глобальный лимит параллельных прогонов: слот = живой LLM-запрос.
# Лишние пользователи не получают отказ, а встают в очередь (событие wait)
MAX_RUNS = 4
_run_slots = asyncio.Semaphore(MAX_RUNS)
_runs_active = 0

# per-user лок: один активный прогон на сессию, без гонок session-маппинга
_user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# простой in-memory rate-limit на /login: попытки по ключу «ip:login»
LOGIN_WINDOW = 60.0
LOGIN_MAX_TRY = 10
_login_hits: dict[str, deque[float]] = defaultdict(deque)


def _login_throttled(key: str) -> bool:
    now = time.monotonic()
    hits = _login_hits[key]
    while hits and now - hits[0] > LOGIN_WINDOW:
        hits.popleft()
    if len(hits) >= LOGIN_MAX_TRY:
        return True
    hits.append(now)
    return False


async def _cleanup_orphans() -> None:
    """Уборка при старте: пустые сессии-призраки, осиротевшие чаты и маппинги.

    Пустая сессия (ноль событий) не несёт истории — только засоряет сайдбар
    «Новыми чатами». Настоящие диалоги не трогаются."""
    try:
        await agent_mod.purge_empty_sessions()
        for login, sid in auth.all_chat_meta():
            if not await agent_mod.session_exists(login, sid):
                auth.delete_chat_meta(login, sid)
        for login, sid in auth.all_stored_sessions():
            if not await agent_mod.session_exists(login, sid):
                auth.clear_stored_session(login)
    except Exception:  # noqa: BLE001 — уборка не должна ронять старт сервера
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    # индекс: нет — строим; устарел после правки poi.sqlite — пересобираем в фоне
    threading.Thread(target=semantic.ensure_index, daemon=True).start()
    await _cleanup_orphans()
    yield

app = FastAPI(title=APP_TITLE, lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _current_user(request: Request) -> str | None:
    return auth.user_of(request.cookies.get(auth.COOKIE_NAME))


def _unauthorized() -> JSONResponse:
    return JSONResponse({"error": "не залогинен"}, status_code=401)


class LoginIn(BaseModel):
    login: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class RegisterIn(BaseModel):
    login: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)


class FollowupsIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)
    answer: str = Field(min_length=1, max_length=4000)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/stats")
async def stats_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "stats.html")


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.post("/login")
async def do_login(body: LoginIn, request: Request) -> JSONResponse:
    ip = request.client.host if request.client else "?"
    key = f"{ip}:{body.login.strip()}"
    if _login_throttled(key):
        return JSONResponse({"error": "слишком много попыток, подождите минуту"},
                            status_code=429)
    token = auth.login(body.login.strip(), body.password)
    if token is None:
        return JSONResponse({"error": "неверный логин или пароль"}, status_code=401)
    _login_hits.pop(key, None)          # успешный вход сбрасывает счётчик
    response = JSONResponse({"ok": True, "login": body.login.strip()})
    response.set_cookie(auth.COOKIE_NAME, token, httponly=True,
                        samesite="lax", max_age=auth.TOKEN_TTL)
    return response


@app.post("/api/register")
async def do_register(body: RegisterIn, request: Request) -> JSONResponse:
    """Регистрация нового пользователя: пароль хранится только хэшем (PBKDF2).
    Успешная регистрация сразу авторизует (кука как при /login)."""
    ip = request.client.host if request.client else "?"
    login = body.login.strip()
    key = f"{ip}:{login}"
    if _login_throttled(key):
        return JSONResponse({"error": "слишком много попыток, подождите минуту"},
                            status_code=429)
    if not auth.valid_login(login):
        return JSONResponse(
            {"error": "логин: 3–32 символа, буквы, цифры, точка, дефис, подчёркивание"},
            status_code=422)
    if not auth.register(login, body.password):
        return JSONResponse({"error": "такой логин уже занят"}, status_code=409)
    token = auth.login(login, body.password)
    if token is None:                       # невозможная ветка: пароль только что вписан
        return JSONResponse({"error": "не удалось создать учётную запись"},
                            status_code=500)
    response = JSONResponse({"ok": True, "login": login})
    response.set_cookie(auth.COOKIE_NAME, token, httponly=True,
                        samesite="lax", max_age=auth.TOKEN_TTL)
    return response


@app.post("/login/demo")
async def do_demo_login(request: Request) -> JSONResponse:
    """Один клик для демо преподавателю: DEMO_USER/DEMO_PASSWORD в .env
    (DEMO_USER должен входить в APP_USERS). Без настроек — 404."""
    user = os.getenv("DEMO_USER", "").strip()
    password = os.getenv("DEMO_PASSWORD", "").strip()
    if not user or not password:
        return JSONResponse({"error": "демо-вход не настроен"}, status_code=404)
    ip = request.client.host if request.client else "?"
    if _login_throttled(f"{ip}:demo"):
        return JSONResponse({"error": "слишком много попыток, подождите минуту"},
                            status_code=429)
    token = auth.login(user, password)
    if token is None:
        return JSONResponse(
            {"error": "демо-вход настроен неверно: DEMO_USER должен быть в APP_USERS"},
            status_code=500)
    response = JSONResponse({"ok": True, "login": user})
    response.set_cookie(auth.COOKIE_NAME, token, httponly=True,
                        samesite="lax", max_age=auth.TOKEN_TTL)
    return response


@app.post("/logout")
async def do_logout(request: Request) -> JSONResponse:
    auth.logout(request.cookies.get(auth.COOKIE_NAME))
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE_NAME)
    return response


async def _session_for(login: str) -> str:
    """Сессия контекста на пользователя: лениво, в SQLite, переживает рестарт.

    Вызывается из-под user-lock в /api/ask — параллельные прогоны не создают
    вторую сессию. Для read-only эндпоинтов (history) используй _session_ro.
    """
    stored = auth.get_stored_session(login)
    if stored and await agent_mod.session_exists(login, stored):
        return stored
    session_id = await agent_mod.create_session(login)
    auth.set_stored_session(login, session_id)
    auth.ensure_chat_meta(login, session_id)
    return session_id


async def _session_ro(login: str) -> str | None:
    """Активная сессия без создания новой: None — диалог ещё не начат
    (черновик «Нового чата»). Раньше здесь лениво создавалась сессия —
    так накапливались пустые «призрачные» чаты в сайдбаре.
    Подвисший маппинг на удалённую сессию чистится."""
    stored = auth.get_stored_session(login)
    if stored is None:
        return None
    if await agent_mod.session_exists(login, stored):
        return stored
    auth.clear_stored_session(login)
    return None


async def _ask_sse(login: str, question: str,
                   ping_after: float = SSE_PING_SEC) -> AsyncIterator[str]:
    """SSE-строки одного прогона: события ask() + heartbeat-комментарии.

    ask() итерируется в ОДНОЙ фоновой Task (_pump) и кладёт события в
    очередь; потребитель ждёт очередь с таймаутом:

    * пауза дольше ping_after → «: ping», но прогон НЕ отменяется
      (наивный wait_for(iterator.__anext__) бросал CancelledError внутрь
      ask на первом же таймауте — поток умирал после первого ping);
    * одна Task на прогон, а не одна на событие — contextvars/OTel-токены
      создаются и сбрасываются в одном контексте (смена Task на каждом
      событии давала «Token was created in a different Context»).

    В finally гасим pump и закрываем ask — статус cancelled дописывается
    в журнал до того, как снаружи снимется user-lock. Глобальный семафор
    MAX_RUNS: при заполненных слотах первым событием уходит «wait» —
    клиент видит очередь, а не молчание; время ожидания уходит в журнал.
    """
    global _runs_active
    run_gen = None
    pump: asyncio.Task | None = None
    queue: asyncio.Queue = asyncio.Queue()
    slot_acquired = False
    wait_started = time.monotonic()
    try:
        if _runs_active >= MAX_RUNS:
            yield f"data: {json.dumps({'type': 'wait', 'active': _runs_active, 'limit': MAX_RUNS}, ensure_ascii=False)}\n\n"
        await _run_slots.acquire()
        slot_acquired = True
        _runs_active += 1
        queue_wait = time.monotonic() - wait_started
        session_id = await _session_for(login)
        # название чата — из ПЕРВОГО вопроса: запасной вариант (сам вопрос),
        # LLM-вариант «с кратким смыслом» прилетит событием title ниже
        is_first = await agent_mod.first_user_text(login, session_id) is None
        auth.touch_chat(login, session_id,
                        question_title=_fallback_title(question) if is_first else None)
        run_gen = agent_mod.ask(login, session_id, question,
                                queue_wait=queue_wait)

        async def _pump() -> None:
            try:
                async for ev in run_gen:
                    queue.put_nowait(("data", ev))
            except Exception as exc:  # noqa: BLE001 — отдаем потребителю
                queue.put_nowait(("error", exc))
            else:
                queue.put_nowait(("end", None))

        pump = asyncio.ensure_future(_pump())
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(),
                                                       timeout=ping_after)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if kind == "end":
                break
            if kind == "error":
                raise payload
            if payload.get("type") == "title":
                # LLM-название чата: только если пользователь не переименовал
                # его вручную (title_auto=1); клиенту событие тоже отдаём
                auth.set_auto_title(login, session_id, payload["title"])
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    finally:
        if pump is not None:
            pump.cancel()                       # CancelledError → ask (журнал)
            try:
                await pump
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if run_gen is not None:
            try:
                await run_gen.aclose()           # no-op, если уже закрыт
            except Exception:                    # noqa: BLE001
                pass
        if slot_acquired:
            _runs_active -= 1
            _run_slots.release()


@app.post("/api/reset")
async def do_reset(request: Request) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    # «Новый диалог»: отвязываем активную сессию — следующий вопрос создаст
    # свежий контекст. История чата остаётся в сайдбаре (хранить её обязательно)
    auth.clear_stored_session(login)
    return JSONResponse({"ok": True})


# ───────────────────────────────── мультичаты (сайдбар диалогов)

class ChatRenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=80)


@app.get("/api/chats")
async def do_list_chats(request: Request) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    active = await _session_ro(login)      # заодно чистит подвисший маппинг
    meta = auth.get_chat_meta(login)
    sessions = await agent_mod.list_user_sessions(login)
    chats = []
    for s in sessions:
        m = meta.get(s["id"])
        if m is None:
            # сессии старше чатовой таблицы — дозаписываем мету лениво
            auth.ensure_chat_meta(login, s["id"])
            m = {"title": auth.DEFAULT_CHAT_TITLE,
                 "updated_at": s.get("updated_at", 0.0)}
        title = m["title"]
        if title == auth.DEFAULT_CHAT_TITLE:
            # старый чат без название — подставляем первый вопрос пользователя
            first = await agent_mod.first_user_text(login, s["id"])
            if first:
                title = _fallback_title(first)
                auth.touch_chat(login, s["id"], question_title=title)
        chats.append({
            "id": s["id"],
            "title": title,
            "updated_at": m.get("updated_at") or s.get("updated_at") or 0.0,
            "active": s["id"] == active,
        })
    return JSONResponse({"chats": chats, "active": active})


@app.post("/api/chats")
async def do_create_chat(request: Request) -> JSONResponse:
    """Новый чат = черновик без сессии: контекст создаётся лениво, при первом
    вопросе (/api/ask). Так в сайдбаре не копятся пустые «Новые чаты» —
    чат появляется в списке, как только в нём что-то спросили."""
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    auth.clear_stored_session(login)
    return JSONResponse({"ok": True, "id": None, "draft": True})


@app.post("/api/chats/{session_id}/activate")
async def do_activate_chat(session_id: str, request: Request) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    if not await agent_mod.session_exists(login, session_id):
        return JSONResponse({"error": "чат не найден"}, status_code=404)
    auth.ensure_chat_meta(login, session_id)
    auth.set_stored_session(login, session_id)
    return JSONResponse({"ok": True, "id": session_id})


@app.patch("/api/chats/{session_id}")
async def do_rename_chat(session_id: str, body: ChatRenameIn,
                         request: Request) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    title = body.title.strip()
    if not auth.set_chat_title(login, session_id, title):
        return JSONResponse({"error": "чат не найден"}, status_code=404)
    return JSONResponse({"ok": True, "title": title})


@app.delete("/api/chats/{session_id}")
async def do_delete_chat(session_id: str, request: Request) -> JSONResponse:
    """Удаление чата: сначала сессия (с проверкой), потом мета.

    Порядок важен: если удалять мету первой, а сессия осталась бы живой,
    список чатов «воскрешал» чат с названием «Новый чат». Теперь ошибка
    удаления честно возвращается клиенту, и сайдбар не врёт."""
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    active = auth.get_stored_session(login)
    if session_id == active and _user_locks[login].locked():
        # по активному чату идёт прогон — удаление сломало бы сессию
        return JSONResponse({"error": "по этому чату идёт прогон"},
                            status_code=409)
    owned = (session_id == active
             or session_id in auth.get_chat_meta(login)
             or await agent_mod.session_exists(login, session_id))
    if not owned:
        return JSONResponse({"error": "чат не найден"}, status_code=404)
    try:
        await agent_mod.delete_session(login, session_id)
    except Exception as exc:  # noqa: BLE001 — сообщаем, не прячем
        return JSONResponse({"error": f"не удалось удалить чат: {exc}"},
                            status_code=500)
    if await agent_mod.session_exists(login, session_id):
        return JSONResponse({"error": "не удалось удалить чат"}, status_code=500)
    auth.delete_chat_meta(login, session_id)
    if session_id == active:
        auth.clear_stored_session(login)
    return JSONResponse({"ok": True})


@app.post("/api/ask")
async def do_ask(body: AskIn, request: Request):
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    question = body.question.strip()
    if not question:
        return JSONResponse({"error": "пустой вопрос"}, status_code=422)

    user_lock = _user_locks[login]
    if user_lock.locked():
        # второй параллельный прогон того же пользователя — не пускаем
        return JSONResponse({"error": "предыдущий ответ ещё готовится"},
                            status_code=409)

    # лок берём ДО возврата StreamingResponse: иначе конкурентный запрос
    # успеет пройти проверку locked() выше и уйдёт в параллельный прогон.
    # Снимается только в _GuardedResponse.__call__: сначала aclose(event_stream)
    # → aclose(_ask_sse) → ask дожурналит cancelled, и лишь потом unlock.
    await user_lock.acquire()
    released = False

    async def event_stream():
        stream = _ask_sse(login, question)
        try:
            async for line in stream:
                yield line
        except asyncio.CancelledError:
            # клиент нажал «Стоп» / закрыл вкладку — агент дожурналит в finally
            raise
        except Exception as exc:  # noqa: BLE001 — обрыв прогона не должен ломать поток
            err = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001
                pass

    class _GuardedResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):  # noqa: D102
            nonlocal released   # без nonlocal присваение ниже делало бы
            # released локальной → UnboundLocalError → лок утекает → 409 навсегда
            try:
                return await super().__call__(scope, receive, send)
            finally:
                # обрыв соединения оставляет event_stream зависшим на yield:
                # закрываем его явно — иначе лок снимется раньше, чем агент
                # допишет журнал, и следующий прогон того же пользователя
                # наложится на ещё живую сессию
                try:
                    await event_stream.aclose()
                except BaseException:  # noqa: BLE001 — гарантируем снятие лока ниже
                    pass
                if not released:
                    released = True
                    user_lock.release()

    return _GuardedResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/history")
async def do_history(request: Request) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    # читаем без user-lock: F5 во время стрима не должен ждать ответа агента;
    # без активной сессии (черновик нового чата) — пустая история, ничего не создаём
    session_id = await _session_ro(login)
    events = await agent_mod.history(login, session_id) if session_id else []
    return JSONResponse({"login": login, "events": events})


@app.get("/api/search")
async def do_search(request: Request, q: str) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    try:
        results = await semantic.atop_k(q, limit=10)
        results = [r for r in results if r["score"] >= semantic.MIN_SCORE]
    except FileNotFoundError:
        return JSONResponse({"error": "индекс не построен"}, status_code=503)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)
    return JSONResponse({"results": results, "min_score": semantic.MIN_SCORE})


@app.get("/api/metrics")
async def do_metrics(request: Request) -> JSONResponse:
    if _current_user(request) is None:
        return _unauthorized()
    try:
        return JSONResponse(stats_mod.all_metrics())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


@app.post("/api/followups")
async def do_followups(body: FollowupsIn, request: Request) -> JSONResponse:
    """3 вопроса-продолжения после ответа: один дешёвый LLM-вызов,
    мимо агентского цикла (не трогает журнал, лимиты и семафор)."""
    if _current_user(request) is None:
        return _unauthorized()
    try:
        suggestions = await agent_mod.suggest_followups(body.question, body.answer)
    except Exception as exc:  # noqa: BLE001 — фронт просто не покажет чипы
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                            status_code=502)
    return JSONResponse({"suggestions": suggestions})


@app.get("/api/health")
async def do_health() -> JSONResponse:
    """Для смоука и демо: индекс построен, совместим с моделью, жив ли семафор."""
    mismatch = semantic.index_model_mismatch()
    ok = semantic.VEC_DB.exists() and mismatch is None
    return JSONResponse({
        "ok": ok,
        "index": {
            "exists": semantic.VEC_DB.exists(),
            "model": semantic._meta().get("model"),
            "expected_model": semantic._embed_model(),
            "mismatch": mismatch,
            "stale": semantic.is_stale() if semantic.VEC_DB.exists() else True,
            "embed_cache": semantic.embed_cache_stats(),
        },
        "runs": {"active": _runs_active, "limit": MAX_RUNS},
    })

