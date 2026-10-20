"""FastAPI-приложение «Городская справочная Ростова».

    uv run python -m uvicorn week2.app.main:app --port 8000

Роуты:
    GET  /                страница (форма вопросов)
    POST /login, /logout  вход/выход (кука HttpOnly)
    POST /api/ask         SSE-поток: как агент думает и что вызывает
    GET  /api/history     контекст диалога (для восстановления после F5)
    GET  /api/search      быстрый семантический поиск (мимо агента)
    POST /api/reset       новый диалог = новая сессия контекста
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from week2.app import agent as agent_mod
from week2.app import auth
from week2.app.semantic import top_k

app = FastAPI(title="Городская справочная Ростова")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SESSIONS: dict[str, str] = {}        # логин → id ADK-сессии (контекст диалога)


def _current_user(request: Request) -> str | None:
    return auth.user_of(request.cookies.get(auth.COOKIE_NAME))


def _unauthorized() -> JSONResponse:
    return JSONResponse({"error": "не залогинен"}, status_code=401)


class LoginIn(BaseModel):
    login: str
    password: str


class AskIn(BaseModel):
    question: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/login")
async def do_login(body: LoginIn, request: Request) -> JSONResponse:
    token = auth.login(body.login, body.password)
    if token is None:
        return JSONResponse({"error": "неверный логин или пароль"}, status_code=401)
    response = JSONResponse({"ok": True, "login": body.login})
    response.set_cookie(auth.COOKIE_NAME, token, httponly=True,
                        samesite="lax", max_age=60 * 60 * 12)
    return response


@app.post("/logout")
async def do_logout(request: Request) -> JSONResponse:
    auth.logout(request.cookies.get(auth.COOKIE_NAME))
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE_NAME)
    return response


async def _session_for(login: str) -> str:
    """Сессия контекста на пользователя: создаётся лениво, живёт до рестарта."""
    if login not in SESSIONS:
        SESSIONS[login] = await agent_mod.create_session(login)
    return SESSIONS[login]


@app.post("/api/reset")
async def do_reset(request: Request) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    SESSIONS.pop(login, None)
    return JSONResponse({"ok": True})


@app.post("/api/ask")
async def do_ask(body: AskIn, request: Request):
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    session_id = await _session_for(login)

    async def event_stream():
        try:
            async for event in agent_mod.ask(login, session_id, body.question):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 — обрыв прогона не должен ломать поток
            err = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history")
async def do_history(request: Request) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    session_id = await _session_for(login)
    events = await agent_mod.history(login, session_id)
    return JSONResponse({"login": login, "events": events})


@app.get("/api/search")
async def do_search(request: Request, q: str) -> JSONResponse:
    login = _current_user(request)
    if login is None:
        return _unauthorized()
    try:
        results = top_k(q, limit=10)
    except FileNotFoundError:
        return JSONResponse({"error": "индекс не построен"}, status_code=503)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)
    return JSONResponse({"results": results})
