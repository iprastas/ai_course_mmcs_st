"""Тесты веб-приложения (week2/app): без обращений к API модели."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from week2.app import agent as app_agent  # noqa: E402
from week2.app import auth, semantic, stats  # noqa: E402


@pytest.fixture()
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_DB", tmp_path / "auth.sqlite")
    return tmp_path / "auth.sqlite"


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

def test_auth_login_ok_and_logout(auth_db, monkeypatch) -> None:
    monkeypatch.setenv("APP_USERS", "alice:pw1,bob:pw2")
    token = auth.login("alice", "pw1")
    assert token and auth.user_of(token) == "alice"
    auth.logout(token)
    assert auth.user_of(token) is None


def test_auth_rejects_bad_password_and_unknown_login(auth_db, monkeypatch) -> None:
    monkeypatch.setenv("APP_USERS", "alice:pw1")
    assert auth.login("alice", "wrong") is None
    assert auth.login("nobody", "pw1") is None


def test_auth_tokens_and_sessions_in_sqlite(auth_db, monkeypatch) -> None:
    """Токен и «логин → сессия» лежат в SQLite: перечитали файл — всё на месте."""
    monkeypatch.setenv("APP_USERS", "alice:pw1")
    token = auth.login("alice", "pw1")
    assert auth.user_of(token) == "alice"
    assert auth.get_stored_session("alice") is None
    auth.set_stored_session("alice", "sess-123")
    assert auth.get_stored_session("alice") == "sess-123"
    auth.clear_stored_session("alice")
    assert auth.get_stored_session("alice") is None


def test_auth_expired_token_rejected(auth_db, monkeypatch) -> None:
    monkeypatch.setenv("APP_USERS", "alice:pw1")
    token = auth.login("alice", "pw1")
    with auth._connect() as con:          # протухаем вручную
        con.execute("UPDATE tokens SET expires = ? WHERE token = ?", (time.time() - 1, token))
    assert auth.user_of(token) is None


def test_register_then_login_and_hashed_storage(auth_db) -> None:
    """Регистрация создаёт учётку; пароль лежит в SQLite только хэшем."""
    assert auth.register("alice", "secret1")
    token = auth.login("alice", "secret1")
    assert token and auth.user_of(token) == "alice"
    assert auth.login("alice", "wrong") is None
    with auth._connect() as con:
        row = con.execute(
            "SELECT pw_hash, pw_salt, iterations FROM users WHERE login = ?",
            ("alice",)).fetchone()
    assert row, "пользователь должен сохраниться в users"
    pw_hash, pw_salt, iterations = row
    assert "secret1" not in pw_hash and "secret1" not in pw_salt
    assert iterations >= 100_000
    assert pw_hash == auth._hash_password("secret1", pw_salt, iterations)


def test_register_rejects_duplicate_and_bad_input(auth_db) -> None:
    assert auth.register("alice", "secret1")
    assert not auth.register("alice", "other12")   # логин занят
    assert not auth.register("al ice", "secret1")  # пробел в логине
    assert not auth.register("ab", "secret1")      # логин короче 3
    assert not auth.register("bob", "12345")       # пароль короче 6
    assert auth.login("bob", "12345") is None      # и не завёлся вовсе


# ─────────────── multi-agent: иерархия, delegate, финал ───────────────

def test_build_agent_hierarchy(monkeypatch) -> None:
    """coordinator (корень, без тулов) → researcher (все 6 инструментов)."""
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    root = app_agent._build_agent()
    assert root.name == app_agent.COORDINATOR_NAME
    assert not root.tools, "координатор не должен иметь своих тулов"
    assert len(root.sub_agents) == 1
    researcher = root.sub_agents[0]
    assert researcher.name == app_agent.RESEARCHER_NAME
    assert len(researcher.tools) == 7      # 5 тулов задания + semantic_search + pin_places
    assert app_agent._OUR_AGENTS == {app_agent.COORDINATOR_NAME,
                                     app_agent.RESEARCHER_NAME}


def test_event_to_sse_delegate_and_transfer_response() -> None:
    ev = _event([], author=app_agent.COORDINATOR_NAME)
    ev.actions = SimpleNamespace(transfer_to_agent="researcher")
    sse = app_agent.event_to_sse(ev)
    assert sse == {"type": "delegate", "from": app_agent.COORDINATOR_NAME,
                   "to": "researcher"}
    # function_response с именем transfer_to_agent не показываем как инструмент
    fr = SimpleNamespace(name="transfer_to_agent", response={"result": "ok"})
    assert app_agent.event_to_sse(_event([], responses=[fr])) is None


def test_final_answer_across_agents() -> None:
    """Финал пишет coordinator; реплики researcher с calls/responses не берём."""
    call_round = _event([_part("вызываю")], author=app_agent.RESEARCHER_NAME)
    call_round.get_function_calls = lambda: ["c"]
    resp_round = _event([_part("факты")], author=app_agent.RESEARCHER_NAME)
    resp_round.get_function_responses = lambda: ["r"]
    final = _event([_part("Итого: 330 аптек")], author=app_agent.COORDINATOR_NAME)
    session = SimpleNamespace(events=[call_round, resp_round, final])
    assert app_agent._final_answer(session) == "Итого: 330 аптек"
    # если transfer не случился и ответил researcher — его текст тоже валиден
    only_res = SimpleNamespace(events=[call_round,
                                       _event([_part("коротко")],
                                              author=app_agent.RESEARCHER_NAME)])
    assert app_agent._final_answer(only_res) == "коротко"


def test_history_legacy_author_answers_not_questions(monkeypatch) -> None:
    """Старые сессии, где агент назывался иначе (rostov_guide): его ответы
    не должны превращаться в «вопросы пользователя» — регрессия гостевого
    чата, где все реплики висели справа от пользователя."""

    def _legacy_ev(author, text, thought=False):
        return SimpleNamespace(
            author=author,
            content=SimpleNamespace(parts=[_part(text, thought=thought)]),
            get_function_calls=lambda: [],
            get_function_responses=lambda: [],
        )

    session = SimpleNamespace(events=[
        _legacy_ev("user", "где поесть пельмени?"),
        _legacy_ev("rostov_guide", "ищу по смыслу", thought=True),
        _legacy_ev("rostov_guide", "Нашёл: Пельмень (1385) и Вкусно (750)"),
    ])

    class _S:
        async def get_session(self, **kwargs):
            return session

    class _R:
        session_service = _S()

    monkeypatch.setattr(app_agent, "runner", lambda: _R())
    out = asyncio.run(app_agent.history("u", "s"))
    questions = [e for e in out if e["type"] == "question"]
    assert [q["text"] for q in questions] == ["где поесть пельмени?"]
    assert out[-1] == {"type": "final",
                       "answer": "Нашёл: Пельмень (1385) и Вкусно (750)"}
    # финал не дублируется в ленте «как думает агент»
    assert not any(e["type"] == "thinking"
                   and "Нашёл" in e.get("text", "") for e in out)


def test_journal_counts_steps_from_both_agents(tmp_path, monkeypatch) -> None:
    """steps — по событиям обоих агентов, а не только корневого."""
    monkeypatch.setattr(app_agent, "RUNS_DIR", tmp_path)

    async def fake_title(question):
        return None

    monkeypatch.setattr(app_agent, "generate_title", fake_title)

    class FakeRunner:
        async def run_async(self, **kwargs):
            yield _event([_part("ищу факты")], author=app_agent.RESEARCHER_NAME)
            yield _event([_part("Готово: 107")], author=app_agent.COORDINATOR_NAME)

        class _S:
            @staticmethod
            async def get_session(**kwargs):
                return SimpleNamespace(events=[
                    _event([_part("Готово: 107")],
                           author=app_agent.COORDINATOR_NAME)])
        session_service = _S()

    monkeypatch.setattr(app_agent, "runner", lambda: FakeRunner())

    async def run():
        return [ev async for ev in app_agent.ask("u", "s", "вопрос")]

    events = asyncio.run(run())
    assert events[0]["type"] == "start"
    assert events[-2]["type"] == "final" and events[-2]["answer"] == "Готово: 107"
    record = json.loads(next(tmp_path.glob("web_*.jsonl"))
                        .read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "ok"
    assert record["steps"] == 2


# ─────────────── маппинг событий ADK ───────────────

def test_answer_text_excludes_thoughts() -> None:
    ev = _event([_part("сначала подумаю. ", thought=True), _part("Ответ: 42")])
    assert app_agent._answer_text(ev) == "Ответ: 42"
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
    assert app_agent.event_to_sse(_event([_part("Готово")])) is None


def test_event_to_sse_skips_partial_tool_call() -> None:
    fc_partial = SimpleNamespace(id="c1", name="nearest", args={"lat": 47.2},
                                 will_continue=True, partial_args=[1])
    assert app_agent.event_to_sse(_event([], calls=[fc_partial])) is None
    fc_done = SimpleNamespace(id="c1", name="nearest", args={"lat": 47.2, "lon": 39.7},
                              will_continue=False, partial_args=None)
    sse = app_agent.event_to_sse(_event([], calls=[fc_done]))
    assert sse["type"] == "tool_call" and sse["name"] == "nearest" and sse["id"] == "c1"


def test_tool_result_unwrapped_for_map() -> None:
    """ADK оборачивает результат в {'result': …} — таблица должна дойти живой.

    id из semantic_search на карту не пинятся: это кандидаты поиска
    (среди них шум мимо запроса), а не рекомендации агента."""
    table = "id | название\n1385 | Пельмень"
    fr = SimpleNamespace(name="semantic_search", response={"result": table})
    sse = app_agent.event_to_sse(_event([], responses=[fr]))
    assert sse["result"] == table
    assert app_agent._ids_from_sse(sse) == []
    # а вот таблица find_places — это уже ответ, её id на карте нужны
    fr2 = SimpleNamespace(name="find_places", response={"result": table})
    sse2 = app_agent.event_to_sse(_event([], responses=[fr2]))
    assert app_agent._ids_from_sse(sse2) == [1385]


# ─────────────── карта: извлечение точек ───────────────

def test_ids_extracted_from_tool_events() -> None:
    call = {"type": "tool_call", "args": {"place_id": 1385}}
    assert app_agent._ids_from_sse(call) == [1385]
    table = {"type": "tool_result", "name": "find_places",
             "result": "id | название\n1385 | Пельмень\n750 | Пельмешка"}
    assert app_agent._ids_from_sse(table) == [1385, 750]
    near = {"type": "tool_result", "name": "nearest", "result": "id=43 Радуга — 79 м"}
    assert app_agent._ids_from_sse(near) == [43]
    assert app_agent._ids_from_sse({"type": "tool_result", "name": "count_places",
                                    "result": "pharmacy: 330 шт."}) == []


def test_semantic_search_noise_not_pinned_and_pin_places_is() -> None:
    """Регрессия: кандидаты semantic_search (парфюмерия, цветы…) не пинятся,
    явная отметка агента pin_places — пинится."""
    noise = {"type": "tool_result", "name": "semantic_search",
             "result": "id | название\n9 | Пена\n10 | Цветы\n1385 | Пельмень"}
    assert app_agent._ids_from_sse(noise) == []
    pinned = {"type": "tool_call", "name": "pin_places",
              "args": {"place_ids": "1385, 750"}}
    assert app_agent._ids_from_sse(pinned) == [1385, 750]
    pinned_list = {"type": "tool_call", "name": "pin_places",
                   "args": {"place_ids": [1385]}}
    assert app_agent._ids_from_sse(pinned_list) == [1385]


def test_pin_places_tool_validates_ids() -> None:
    """pin_places подтверждает только реальные id из poi.sqlite."""
    out = app_agent.pin_places("1385, 750")
    assert "отмечено на карте" in out and "id=1385" in out
    out2 = app_agent.pin_places("999999")
    assert "не найден" in out2
    assert "ошибка" in app_agent.pin_places("абв")


def test_coords_and_collect_points() -> None:
    pts = app_agent._coords([1385])          # реальная poi.sqlite, только чтение
    assert pts and pts[0]["name"] == "Пельмень"
    assert 47 < pts[0]["lat"] < 48 and 39 < pts[0]["lon"] < 40
    events = [
        {"type": "tool_result", "result": "1385 | Пельмень"},
        {"type": "tool_call", "args": {"place_id": 750}},
        {"type": "tool_call", "args": {"kind": "cafe"}},
    ]
    got = app_agent.collect_points(events)
    assert [p["id"] for p in got] == [1385, 750]


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
    out = asyncio.run(semantic.semantic_search("пельмени"))
    assert out.startswith("ошибка") and "build" in out


@pytest.mark.skipif(not semantic.VEC_DB.exists(), reason="индекс не построен")
def test_index_finds_pelmeni() -> None:
    found = semantic.top_k("пельмени", limit=5)
    assert found and found[0]["name"] in ("Пельмень", "Пельмешка")
    assert 0.0 <= found[0]["score"] <= 1.0


@pytest.mark.skipif(not semantic.VEC_DB.exists(), reason="индекс не построен")
def test_index_model_mismatch_detected(monkeypatch) -> None:
    monkeypatch.setenv("EMBED_MODEL", "some-other-model")
    assert semantic.index_model_mismatch() is not None
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    assert semantic.index_model_mismatch() is None


def test_index_model_mismatch_missing_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(semantic, "VEC_DB", tmp_path / "nope.sqlite")
    assert semantic.index_model_mismatch() is None     # нет индекса — не ошибка


def test_semantic_search_filters_low_scores(monkeypatch) -> None:
    async def fake_atop(q, limit=10):
        return [
            {"id": 1, "name": "Пельмень", "kind": "cafe", "cuisine": "пельмени",
             "addr": "ул. Ленина", "score": 0.91},
            {"id": 2, "name": "Случайный банк", "kind": "bank", "cuisine": None,
             "addr": "пр. Ворошиловский", "score": 0.12},
        ]
    monkeypatch.setattr(semantic, "atop_k", fake_atop)
    out = asyncio.run(semantic.semantic_search("пельмени"))
    assert "Пельмень" in out
    assert "Случайный банк" not in out


def test_semantic_search_empty_below_threshold(monkeypatch) -> None:
    async def fake_atop(q, limit=10):
        return [{"id": 3, "name": "X", "kind": "cafe", "cuisine": None,
                 "addr": None, "score": 0.1}]
    monkeypatch.setattr(semantic, "atop_k", fake_atop)
    out = asyncio.run(semantic.semantic_search("пельмени"))
    assert "ничего" in out


# ─────────────── журнал при отмене ───────────────

def test_journal_on_cancel(tmp_path, monkeypatch) -> None:
    """Закрытие генератора ask() (Stop у клиента) должно дожурналить статус.

    Сценарий: клиент нажал «Стоп» / закрыл вкладку → Starlette закрывает
    асинхронный генератор через aclose() → GeneratorExit в точке yield
    (или сразу при входе в run_async) → finally в ask() пишет запись
    в runs/*.jsonl со статусом cancelled.
    """
    monkeypatch.setattr(app_agent, "RUNS_DIR", tmp_path)

    async def fake_title(question):
        return None

    monkeypatch.setattr(app_agent, "generate_title", fake_title)

    class FakeRunner:
        async def run_async(self, **kwargs):
            # первый partial-чанк успевает дойти до клиента
            yield app_agent._event([_part("думаю")], partial=True)
            # «вечный» агент: клиент оборвал — GeneratorExit на yield
            while True:
                yield app_agent._event([_part(".")], partial=True)

        class _S:
            @staticmethod
            async def get_session(**kwargs):
                return None
        session_service = _S()

    monkeypatch.setattr(app_agent, "runner", lambda: FakeRunner())

    async def run():
        gen = app_agent.ask("u", "s", "вопрос")
        assert (await gen.__anext__())["type"] == "start"
        # имитация отмены клиентом
        await gen.aclose()

    asyncio.run(run())
    files = list(tmp_path.glob("web_*.jsonl"))
    assert files, "журнал не записан при отмене"
    record = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "cancelled"


# ─────────────── ограничение контекста (хвост истории для модели) ───────────────

def test_run_config_limits_context_and_offloads_tools() -> None:
    """Модель видит только последние MAX_CONTEXT_EVENTS событий сессии.

    Runner прокидывает get_session_config в DatabaseSessionService.get_session
    (ORDER BY timestamp DESC LIMIT n) — в отличие от старого trim_context,
    который пытался патчить несуществующий update_session и был no-op.
    UI читает сессию без лимита — весь диалог после F5 сохраняется.
    """
    rc = app_agent._run_config()
    assert rc.get_session_config is not None
    assert rc.get_session_config.num_recent_events == app_agent.MAX_CONTEXT_EVENTS
    assert rc.max_llm_calls == app_agent.MAX_LLM_CALLS
    # sync sqlite-тулы не должны сидеть в event loop
    assert rc.tool_thread_pool_config is not None
    assert rc.tool_thread_pool_config.max_workers >= 1


# ─────────────── LRU-кеш эмбеддингов ───────────────

class _FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    async def create(self, *, model, input):
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0]) for _ in input])


def test_embed_cache_skips_api_on_repeat() -> None:
    """Повторный запрос той же фразы не идёт в API эмбеддингов."""
    semantic._embed_cache.clear()
    semantic._embed_stats["hits"] = semantic._embed_stats["misses"] = 0
    fake = SimpleNamespace(embeddings=_FakeEmbeddings())

    v1 = asyncio.run(semantic._aembed(fake, ["пельмени"]))
    v2 = asyncio.run(semantic._aembed(fake, ["пельмени"]))
    assert fake.embeddings.calls == 1, "второй запрос должен быть кешем"
    assert v1 == v2
    stats = semantic.embed_cache_stats()
    assert stats["hits"] >= 1 and stats["size"] == 1

    asyncio.run(semantic._aembed(fake, ["кофе с завтраком"]))
    assert fake.embeddings.calls == 2, "новая фраза — промах кеша"


def test_embed_cache_lru_evicts_oldest() -> None:
    semantic._embed_cache.clear()
    semantic._embed_stats["hits"] = semantic._embed_stats["misses"] = 0
    fake = SimpleNamespace(embeddings=_FakeEmbeddings())
    for i in range(semantic._EMBED_CACHE_MAX + 10):
        asyncio.run(semantic._aembed(fake, [f"q-{i}"]))
    assert len(semantic._embed_cache) <= semantic._EMBED_CACHE_MAX
    assert "q-0" not in semantic._embed_cache      # самое старое вытеснено
    assert f"q-{semantic._EMBED_CACHE_MAX + 9}" in semantic._embed_cache


# ─────────────── глобальный семафор прогонов ───────────────

def test_run_slots_queue_emits_wait(monkeypatch) -> None:
    """При заполненных слотах 5-й прогон первым событием видит «wait»."""
    import week2.app.main as main_mod

    async def fake_session(login):
        return "s"

    async def fake_ask(user_id, session_id, question, queue_wait=0.0):
        yield {"type": "start"}
        await asyncio.sleep(30)                  # удерживает слот
        yield {"type": "final", "answer": "x"}

    monkeypatch.setattr(main_mod, "_session_for", fake_session)
    monkeypatch.setattr(main_mod.agent_mod, "ask", fake_ask)

    async def run():
        # занять все слоты
        holders = []
        for _ in range(main_mod.MAX_RUNS):
            g = main_mod._ask_sse("u", "q", ping_after=5.0)
            first = await g.__anext__()
            assert '"start"' in first, "слот должен достаться сразу"
            holders.append(g)
        assert main_mod._runs_active == main_mod.MAX_RUNS

        # 5-й — в очередь
        g5 = main_mod._ask_sse("u", "q", ping_after=5.0)
        first5 = await g5.__anext__()
        assert '"wait"' in first5, "ожидалось событие очереди"
        assert main_mod._runs_active == main_mod.MAX_RUNS

        for g in holders:
            await g.aclose()
        await g5.aclose()
        assert main_mod._runs_active == 0, "слоты не освободились"

    asyncio.run(run())


# ─────────────── heartbeat SSE: пауза не убивает поток ───────────────

def test_sse_heartbeat_does_not_kill_stream(monkeypatch) -> None:
    """Пауза дольше ping_after даёт ': ping', но прогон продолжается до конца.

    Регрессия: wait_for(__anext__) отменял ask на первом таймауте —
    после первого ping поток умирал без final/usage.
    """
    import week2.app.main as main_mod

    async def fake_session(login):
        return "s"

    async def fake_ask(user_id, session_id, question, queue_wait=0.0):
        yield {"type": "start"}
        await asyncio.sleep(0.12)               # пауза > ping_after
        yield {"type": "final", "answer": "ок"}

    monkeypatch.setattr(main_mod, "_session_for", fake_session)
    monkeypatch.setattr(main_mod.agent_mod, "ask", fake_ask)

    async def run():
        return [line async for line in main_mod._ask_sse("u", "q", ping_after=0.03)]

    lines = asyncio.run(run())
    assert any(line.startswith(": ping") for line in lines), "heartbeat не отправлен"
    assert any('"start"' in line for line in lines)
    assert any('"final"' in line for line in lines), "поток умер после ping"


def test_sse_close_propagates_to_agent(monkeypatch) -> None:
    """aclose() на SSE-потоке закрывает генератор агента (для журнала/чистоты)."""
    import week2.app.main as main_mod
    state = {"closed": False}

    async def fake_session(login):
        return "s"

    async def fake_ask(user_id, session_id, question, queue_wait=0.0):
        try:
            while True:
                yield {"type": "thinking", "text": "."}
                await asyncio.sleep(0.01)
        finally:
            state["closed"] = True

    monkeypatch.setattr(main_mod, "_session_for", fake_session)
    monkeypatch.setattr(main_mod.agent_mod, "ask", fake_ask)

    async def run():
        gen = main_mod._ask_sse("u", "q", ping_after=5.0)
        first = await gen.__anext__()
        assert "thinking" in first
        await gen.aclose()

    asyncio.run(run())
    assert state["closed"], "генератор агента не закрыт при aclose"


# ─────────────── ошибка прогона: статус error, не cancelled ───────────────

def test_journal_status_error_on_failure(tmp_path, monkeypatch) -> None:
    """Исключение в run_async → событие error в потоке + status='error' в журнале."""
    monkeypatch.setattr(app_agent, "RUNS_DIR", tmp_path)

    class FakeRunner:
        async def run_async(self, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover — делает функцию async-генератором

        class _S:
            @staticmethod
            async def get_session(**kwargs):
                return None
        session_service = _S()

    monkeypatch.setattr(app_agent, "runner", lambda: FakeRunner())

    async def run():
        return [ev async for ev in app_agent.ask("u", "s", "вопрос")]

    events = asyncio.run(run())
    assert events[0]["type"] == "start"
    assert any(e["type"] == "error" and "boom" in e["message"] for e in events)
    files = list(tmp_path.glob("web_*.jsonl"))
    assert files, "журнал не записан при ошибке"
    record = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "error"


# ─────────────── метрики ───────────────

def test_stats_wilson_known_value() -> None:
    lo, hi = stats.wilson(10, 20)
    assert abs(lo - 0.299) < 0.02 and abs(hi - 0.701) < 0.02


def _record(task: str, run: int, passed: bool, tool_ok: bool) -> dict:
    return {"task_id": task, "run": run, "passed": passed, "tool_ok": tool_ok,
            "tokens": 1000, "cost_rub": 0.1, "steps": 2, "status": "ok"}


def test_stats_summarize_and_ab(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(stats, "RUNS_DIR", tmp_path)
    (tmp_path / "main_full_n3_x.jsonl").write_text("\n".join(
        json.dumps(r) for r in [
            _record("t1", 1, True, True), _record("t1", 2, True, True),
            _record("t2", 1, False, False),
        ]), encoding="utf-8")
    (tmp_path / "ab_short_n3_x.jsonl").write_text("\n".join(
        json.dumps(_record("t1", i, True, False)) for i in range(10)), encoding="utf-8")
    (tmp_path / "ab_full_n3_x.jsonl").write_text("\n".join(
        json.dumps(_record("t1", i, True, True)) for i in range(10)), encoding="utf-8")

    m = stats.all_metrics()
    s = m["main"]["summary"]
    assert s["runs"] == 3 and s["pass_at_1"] == round(2 / 3, 4)
    assert s["pass_k"] == 0.5 and s["tool_ok_n"] == 2
    ab = m["ab"]
    assert ab["delta"]["acc_pp"] == 100.0
    assert not ab["overlap"] and "окупились" in ab["verdict"]


# ─────────────── HTTP-слой ───────────────

@pytest.fixture()
def client(auth_db, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import week2.app.main as main_mod

    monkeypatch.setenv("APP_USERS", "tester:secret")
    monkeypatch.setattr(main_mod.semantic, "top_k", lambda q, limit=10: [])
    # сессии ADK — во временный каталог: тесты не должны засорять
    # рабочий week2/app/data/sessions.db «призрачными» чатами
    monkeypatch.setattr(app_agent, "DATA_DIR", tmp_path / "sessions")
    monkeypatch.setattr(app_agent, "_RUNNER", None)
    return TestClient(main_mod.app)


def test_index_and_extra_pages(client) -> None:
    assert client.get("/").status_code == 200
    assert client.get("/stats").status_code == 200
    assert client.get("/favicon.ico").status_code == 200


def test_api_requires_auth(client) -> None:
    assert client.get("/api/history").status_code == 401
    assert client.post("/api/ask", json={"question": "привет"}).status_code == 401
    assert client.get("/api/search", params={"q": "пельмени"}).status_code == 401
    assert client.get("/api/metrics").status_code == 401


def test_login_logout_flow(client) -> None:
    assert client.post("/login", json={"login": "tester", "password": "bad"}).status_code == 401
    assert client.post("/login", json={"login": "tester", "password": "secret"}).status_code == 200
    r = client.get("/api/history")
    assert r.status_code == 200 and r.json()["login"] == "tester"
    assert client.post("/api/reset").status_code == 200
    assert client.post("/logout").status_code == 200
    assert client.get("/api/history").status_code == 401


def test_question_length_limit(client) -> None:
    client.post("/login", json={"login": "tester", "password": "secret"})
    assert client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422


def test_metrics_endpoint(client) -> None:
    client.post("/login", json={"login": "tester", "password": "secret"})
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "main" in body and "ab" in body


def test_health_endpoint(client) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and "index" in body
    assert "exists" in body["index"] and "expected_model" in body["index"]
    # семафор и кеш эмбеддингов видны из health (для смоука/демо)
    assert body["runs"]["limit"] >= 1 and body["runs"]["active"] >= 0
    assert "embed_cache" in body["index"]


def test_ask_stream_releases_user_lock(client, monkeypatch) -> None:
    """После успешного прогона user-lock снят: второй запрос не получает 409.

    Регрессия: присваивание released без nonlocal в _GuardedResponse.__call__
    давало UnboundLocalError — лок не снимался, все следующие /api/ask
    пользователя зависали с 409 «предыдущий ответ ещё готовится».
    """
    import week2.app.main as main_mod

    async def fake_session(login):
        return "s-test"

    async def fake_ask(user_id, session_id, question, queue_wait=0.0):
        yield {"type": "start", "question": question}
        yield {"type": "final", "answer": "42"}

    monkeypatch.setattr(main_mod, "_session_for", fake_session)
    monkeypatch.setattr(main_mod.agent_mod, "ask", fake_ask)

    client.post("/login", json={"login": "tester", "password": "secret"})
    r1 = client.post("/api/ask", json={"question": "первый"})
    assert r1.status_code == 200
    assert '"final"' in r1.text
    r2 = client.post("/api/ask", json={"question": "второй"})
    assert r2.status_code == 200, "лок не снят после первого прогона"
    assert '"final"' in r2.text


# ─────────────── мультичаты ───────────────

def _fake_agent_sessions(monkeypatch, sessions: dict) -> None:
    """Фейковый ADK: сессии лежат в dict sessions (id → user_id)."""
    import week2.app.main as main_mod
    counter = {"n": 0}

    async def fake_create(user_id):
        counter["n"] += 1
        sid = f"sess-{counter['n']}"
        sessions[sid] = {"user_id": user_id, "updated_at": time.time()}
        return sid

    async def fake_exists(user_id, session_id):
        s = sessions.get(session_id)
        return bool(s and s["user_id"] == user_id)

    async def fake_delete(user_id, session_id):
        sessions.pop(session_id, None)

    async def fake_list(user_id):
        return [{"id": sid, "updated_at": s["updated_at"]}
                for sid, s in sessions.items() if s["user_id"] == user_id]

    async def fake_history(user_id, session_id):
        return [{"type": "question", "text": f"q-{session_id}"}]

    async def fake_first_text(user_id, session_id):
        return None

    monkeypatch.setattr(main_mod.agent_mod, "create_session", fake_create)
    monkeypatch.setattr(main_mod.agent_mod, "session_exists", fake_exists)
    monkeypatch.setattr(main_mod.agent_mod, "delete_session", fake_delete)
    monkeypatch.setattr(main_mod.agent_mod, "list_user_sessions", fake_list)
    monkeypatch.setattr(main_mod.agent_mod, "history", fake_history)
    monkeypatch.setattr(main_mod.agent_mod, "first_user_text", fake_first_text)


def test_chats_crud_flow(client, monkeypatch) -> None:
    """Чаты создаются лениво из вопроса: список/переименование/переключение/удаление.

    Регрессия: раньше сессии создавались при каждом «Новый чат» и при чтении
    истории — в сайдбаре копились пустые «Новые чаты», а удаление одного из
    них тут же «воскрешало» новый такой же."""
    import week2.app.main as main_mod
    sessions: dict = {}
    _fake_agent_sessions(monkeypatch, sessions)

    async def fake_ask(user_id, session_id, question, queue_wait=0.0):
        yield {"type": "start", "question": question}
        yield {"type": "final", "answer": "ок"}

    monkeypatch.setattr(main_mod.agent_mod, "ask", fake_ask)
    client.post("/login", json={"login": "tester", "password": "secret"})

    # пустой аккаунт: чатов нет, активной сессии нет, ничего не создаётся
    assert client.get("/api/chats").json() == {"chats": [], "active": None}
    assert client.get("/api/history").json()["events"] == []
    assert sessions == {}

    # чат рождается из первого вопроса (сессия + название из вопроса)
    assert client.post("/api/ask", json={"question": "первый вопрос"}).status_code == 200
    id1 = client.get("/api/chats").json()["active"]
    assert id1 and len(sessions) == 1

    # «+ Новый чат» — черновик: пустая сессия не создаётся до первого вопроса
    r = client.post("/api/chats")
    assert r.status_code == 200 and r.json()["id"] is None
    assert client.get("/api/chats").json()["active"] is None
    assert len(sessions) == 1, "пустых сессий-призраков быть не должно"

    # второй чат — из вопроса в черновике
    client.post("/api/ask", json={"question": "второй вопрос"})
    id2 = client.get("/api/chats").json()["active"]
    assert id2 != id1 and len(sessions) == 2

    r = client.get("/api/chats")
    data = r.json()
    assert {c["id"] for c in data["chats"]} == {id1, id2}
    titles = {c["id"]: c["title"] for c in data["chats"]}
    assert titles[id1] == "первый вопрос" and titles[id2] == "второй вопрос"

    # история активного чата изолирована
    assert client.get("/api/history").json()["events"][0]["text"] == f"q-{id2}"

    # переключение на первый чат
    assert client.post(f"/api/chats/{id1}/activate").status_code == 200
    assert client.get("/api/chats").json()["active"] == id1
    assert client.get("/api/history").json()["events"][0]["text"] == f"q-{id1}"

    # переименование
    r = client.patch(f"/api/chats/{id1}", json={"title": "Поездка"})
    assert r.status_code == 200
    titles = {c["id"]: c["title"] for c in client.get("/api/chats").json()["chats"]}
    assert titles[id1] == "Поездка" and titles[id2] == "второй вопрос"

    # 404 для чужого/несуществующего id
    assert client.patch("/api/chats/nope", json={"title": "x"}).status_code == 404
    assert client.post("/api/chats/nope/activate").status_code == 404

    # удаление активного чата: он исчезает из списка, активного больше нет
    assert client.delete(f"/api/chats/{id1}").status_code == 200
    left = client.get("/api/chats").json()
    assert {c["id"] for c in left["chats"]} == {id2} and left["active"] is None
    assert client.delete(f"/api/chats/{id2}").status_code == 200
    assert client.get("/api/chats").json()["chats"] == []
    assert sessions == {}, "сессии должны удалиться из ADK, а не только из меты"


def test_chat_autotitle_from_first_question(client, monkeypatch) -> None:
    """Дефолтный тайтл «Новый чат» заменяется первым вопросом пользователя
    (позже вопросами не перетирается, ручное название — тоже защищено)."""
    import week2.app.main as main_mod
    sessions: dict = {}
    _fake_agent_sessions(monkeypatch, sessions)

    async def fake_ask(user_id, session_id, question, queue_wait=0.0):
        yield {"type": "start", "question": question}
        yield {"type": "final", "answer": "ок"}

    monkeypatch.setattr(main_mod.agent_mod, "ask", fake_ask)

    client.post("/login", json={"login": "tester", "password": "secret"})
    client.post("/api/chats")                      # черновик нового чата
    r = client.post("/api/ask", json={"question": "Сколько в городе аптек?"})
    assert r.status_code == 200
    chats = client.get("/api/chats").json()["chats"]
    active = [c for c in chats if c["active"]][0]
    assert active["title"] == "Сколько в городе аптек?"
    # название первого вопроса не перетирается последующими вопросами
    client.post("/api/ask", json={"question": "Ещё вопрос совсем другой длины"})
    chats = client.get("/api/chats").json()["chats"]
    active = [c for c in chats if c["active"]][0]
    assert active["title"] == "Сколько в городе аптек?"
    # дефолтное/авто название не трогаем у чатов с ручным названием
    client.patch(f"/api/chats/{active['id']}", json={"title": "Мой маршрут"})
    client.post("/api/ask", json={"question": "И третий вопрос"})
    chats = client.get("/api/chats").json()["chats"]
    active = [c for c in chats if c["active"]][0]
    assert active["title"] == "Мой маршрут", "ручное название должно сохраниться"


# ─────────────── follow-up вопросы ───────────────

def test_parse_suggestions_plain_and_fenced() -> None:
    assert app_agent._parse_suggestions(
        '["А что рядом?", "Где ещё поесть?", "Как туда добраться?"]'
    ) == ["А что рядом?", "Где ещё поесть?", "Как туда добраться?"]
    fenced = '```json\n["Где ещё поесть?", "Что работает ночью?", "Какие отзывы?"]\n```'
    assert app_agent._parse_suggestions(fenced) == [
        "Где ещё поесть?", "Что работает ночью?", "Какие отзывы?"]
    assert app_agent._parse_suggestions("не json") == []
    # слишком короткий мусор отсекается, максимум 3
    got = app_agent._parse_suggestions(
        '["ok вопрос?", "x", "' + "д" * 200 + '", "Ещё один вопрос?"]')
    assert got == ["ok вопрос?", "Ещё один вопрос?"]


def test_followups_endpoint(client, monkeypatch) -> None:
    import week2.app.main as main_mod

    async def fake_suggest(question, answer):
        return ["А круглосуточные?", "Что на Пушкинской?", "Где пельмени?"]

    monkeypatch.setattr(main_mod.agent_mod, "suggest_followups", fake_suggest)
    client.post("/login", json={"login": "tester", "password": "secret"})
    r = client.post("/api/followups",
                    json={"question": "сколько аптек", "answer": "330"})
    assert r.status_code == 200
    assert len(r.json()["suggestions"]) == 3

    async def broken(question, answer):
        raise RuntimeError("llm down")

    monkeypatch.setattr(main_mod.agent_mod, "suggest_followups", broken)
    r = client.post("/api/followups",
                    json={"question": "сколько аптек", "answer": "330"})
    assert r.status_code == 502


def test_followups_requires_auth(client) -> None:
    r = client.post("/api/followups", json={"question": "q", "answer": "a"})
    assert r.status_code == 401


# ─────────────── демо-вход ───────────────

def test_demo_login_disabled_without_env(client, monkeypatch) -> None:
    monkeypatch.delenv("DEMO_USER", raising=False)
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    assert client.post("/login/demo").status_code == 404


def test_demo_login_sets_cookie(client, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_USER", "tester")
    monkeypatch.setenv("DEMO_PASSWORD", "secret")     # tester ∈ APP_USERS
    r = client.post("/login/demo")
    assert r.status_code == 200
    assert r.json()["login"] == "tester"
    assert "rostov_session" in r.cookies
    # с кукой историю видно
    client.cookies.update(r.cookies)
    assert client.get("/api/history").status_code == 200


def test_demo_login_misconfigured_user(client, monkeypatch) -> None:
    monkeypatch.setenv("DEMO_USER", "ghost")
    monkeypatch.setenv("DEMO_PASSWORD", "nope")       # ghost ∉ APP_USERS
    assert client.post("/login/demo").status_code == 500


# ─────────────── регистрация ───────────────

def test_register_endpoint_auto_login(client) -> None:
    r = client.post("/api/register", json={"login": "новичок", "password": "secret1"})
    assert r.status_code == 200 and r.json()["login"] == "новичок"
    assert "rostov_session" in r.cookies
    client.cookies.update(r.cookies)
    assert client.get("/api/history").status_code == 200   # сразу залогинен
    # повторная регистрация того же логина — 409
    r = client.post("/api/register", json={"login": "новичок", "password": "другой1"})
    assert r.status_code == 409
    # слабый пароль и мусорный логин — 422
    assert client.post("/api/register",
                       json={"login": "второй", "password": "12345"}).status_code == 422
    assert client.post("/api/register",
                       json={"login": "a b:c", "password": "secret1"}).status_code == 422
    # созданный пользователь логинится обычной формой
    client.cookies.clear()
    r = client.post("/login", json={"login": "новичок", "password": "secret1"})
    assert r.status_code == 200


def test_login_rate_limit(client) -> None:
    for _ in range(10):
        client.post("/login", json={"login": "tester", "password": "bad"})
    r = client.post("/login", json={"login": "tester", "password": "bad"})
    assert r.status_code == 429
    # после лимита верный пароль тоже немного ждёт — но это ок для демо
