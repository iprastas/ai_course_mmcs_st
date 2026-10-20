"""Тесты веб-приложения (week2/app): без обращений к API модели."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from week2.app import agent as app_agent  # noqa: E402
from week2.app import auth, semantic  # noqa: E402


def _part(text: str, thought: bool = False) -> SimpleNamespace:
    return SimpleNamespace(text=text, thought=thought)


def _event(parts: list, partial: bool = False, calls=None, responses=None,
           author: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        author=author or app_agent.AGENT_NAME,
        content=SimpleNamespace(parts=parts),
        partial=partial,
        get_function_calls=lambda: calls or [],
        get_function_responses=lambda: responses or [],
    )


# ─────────────── авторизация ───────────────

def test_auth_login_ok_and_logout(monkeypatch) -> None:
    monkeypatch.setenv("APP_USERS", "alice:pw1,bob:pw2")
    auth.TOKENS.clear()
    token = auth.login("alice", "pw1")
    assert token and auth.user_of(token) == "alice"
    auth.logout(token)
    assert auth.user_of(token) is None


def test_auth_rejects_bad_password_and_unknown_login(monkeypatch) -> None:
    monkeypatch.setenv("APP_USERS", "alice:pw1")
    auth.TOKENS.clear()
    assert auth.login("alice", "wrong") is None
    assert auth.login("nobody", "pw1") is None


# ─────────────── маппинг событий ADK ───────────────

def test_answer_text_excludes_thoughts() -> None:
    ev = _event([_part("сначала подумаю. ", thought=True), _part("Ответ: 42")])
    assert app_agent._answer_text(ev) == "Ответ: 42"
    # если провайдер не разделил thought — берём весь текст
    ev2 = _event([_part("просто ответ")])
    assert app_agent._answer_text(ev2) == "просто ответ"
    assert app_agent._answer_text(_event([])) == ""


def test_final_answer_skips_tool_rounds() -> None:
    tool_round = _event([_part("вызываю", thought=True), _part("")])
    tool_round.get_function_calls = lambda: ["call"]
    middle = _event([_part("результат инструмента")])
    final = _event([_part("сначала думаю. ", thought=True), _part("Финальный ответ")])
    session = SimpleNamespace(events=[tool_round, middle, final])
    assert app_agent._final_answer(session) == "Финальный ответ"


def test_event_to_sse_partial_thinking() -> None:
    sse = app_agent.event_to_sse(_event([_part("Ковыряю")], partial=True))
    assert sse == {"type": "thinking", "text": "Ковыряю"}
    # полный текст без вызовов не дублируем в поток (он уйдёт в final из истории)
    assert app_agent.event_to_sse(_event([_part("Готово")])) is None


def test_event_to_sse_skips_partial_tool_call() -> None:
    fc_partial = SimpleNamespace(id="c1", name="nearest", args={"lat": 47.2},
                                 will_continue=True, partial_args=[1])
    sse = app_agent.event_to_sse(_event([], calls=[fc_partial]))
    assert sse is None
    fc_done = SimpleNamespace(id="c1", name="nearest", args={"lat": 47.2, "lon": 39.7},
                              will_continue=False, partial_args=None)
    sse = app_agent.event_to_sse(_event([], calls=[fc_done]))
    assert sse["type"] == "tool_call" and sse["name"] == "nearest" and sse["id"] == "c1"


# ─────────────── семантический поиск ───────────────

def test_normalize_is_unit_vector() -> None:
    v = semantic._normalize([3.0, 4.0])
    assert abs((v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9


def test_doc_composes_text() -> None:
    doc = semantic._doc({"name": "Пельмень", "kind": "cafe", "cuisine": None,
                         "brand": None, "street": "Ростовская"})
    assert "Пельмень" in doc and "cafe" in doc and "Ростовская" in doc
    assert "None" not in doc


def test_semantic_search_without_index(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(semantic, "VEC_DB", tmp_path / "nope.sqlite")
    out = semantic.semantic_search("пельмени")
    assert out.startswith("ошибка") and "build" in out


@pytest.mark.skipif(not semantic.VEC_DB.exists(), reason="индекс не построен")
def test_index_finds_pelmeni() -> None:
    found = semantic.top_k("пельмени", limit=5)
    assert found and found[0]["name"] in ("Пельмень", "Пельмешка")
    assert 0.0 <= found[0]["score"] <= 1.0


# ─────────────── HTTP-слой ───────────────

@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import week2.app.main as main_mod

    monkeypatch.setenv("APP_USERS", "tester:secret")
    monkeypatch.setattr(main_mod, "top_k", lambda q, limit=10: [])
    main_mod.SESSIONS.clear()
    auth.TOKENS.clear()
    return TestClient(main_mod.app)


def test_index_page(client) -> None:
    assert client.get("/").status_code == 200


def test_api_requires_auth(client) -> None:
    assert client.get("/api/history").status_code == 401
    assert client.post("/api/ask", json={"question": "привет"}).status_code == 401
    assert client.get("/api/search", params={"q": "пельмени"}).status_code == 401


def test_login_logout_flow(client) -> None:
    assert client.post("/login", json={"login": "tester", "password": "bad"}).status_code == 401
    assert client.post("/login", json={"login": "tester", "password": "secret"}).status_code == 200
    r = client.get("/api/history")
    assert r.status_code == 200 and r.json()["login"] == "tester"
    assert client.post("/api/reset").status_code == 200
    assert client.post("/logout").status_code == 200
    assert client.get("/api/history").status_code == 401
