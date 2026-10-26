"""Авторизация и учётные данные: пользователи в SQLite, токены в HttpOnly-куке.

Пароли НЕ хранятся в .env: учётки лежат в SQLite (week2/app/data/auth.sqlite)
в виде PBKDF2-хэшей с солью. Регистрация — POST /api/register (или кнопка на
экране входа). Переменная APP_USERS в .env — лишь необязательный «бутстрап»:
пары логин:пароль при первом обращении один раз зашиваются в таблицу users
(хэшем) и дальше не используются — можно удалить их из .env после первого запуска.

Там же — маппинг «логин → активная сессия диалога» и метаданные чатов
(chat_meta: название, время, признак авто-названия). Всё переживает рестарт
сервера; сессии диалогов лежат в ADK sessions.db.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

COOKIE_NAME = "rostov_session"
TOKEN_TTL = 60 * 60 * 12                # срок жизни токена — как у куки, 12 часов

AUTH_DB = Path(__file__).resolve().parent / "data" / "auth.sqlite"

# PBKDF2-HMAC-SHA256: соль на пользователя, итерации пишутся в строку —
# при желании их можно поднять, старые хэши продолжат проверяться
PBKDF2_ITERATIONS = 240_000
MIN_PASSWORD_LEN = 6
LOGIN_RE = re.compile(r"^[\w.-]{3,32}$", re.UNICODE)


def _connect() -> sqlite3.Connection:
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(AUTH_DB, timeout=5.0)
    # WAL + busy_timeout: параллельные запросы (несколько пользователей)
    # не падают с «database is locked», а ждут записи доли секунды
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("CREATE TABLE IF NOT EXISTS users ("
                " login TEXT PRIMARY KEY, pw_hash TEXT NOT NULL,"
                " pw_salt TEXT NOT NULL, iterations INTEGER NOT NULL,"
                " created_at REAL NOT NULL)")
    con.execute("CREATE TABLE IF NOT EXISTS tokens ("
                " token TEXT PRIMARY KEY, login TEXT NOT NULL, expires REAL NOT NULL)")
    con.execute("CREATE TABLE IF NOT EXISTS user_sessions ("
                " login TEXT PRIMARY KEY, session_id TEXT NOT NULL)")
    # метаданные чатов: названия и время (сами сессии — в ADK sessions.db)
    con.execute("CREATE TABLE IF NOT EXISTS chat_meta ("
                " session_id TEXT PRIMARY KEY, login TEXT NOT NULL,"
                " title TEXT NOT NULL, created_at REAL NOT NULL,"
                " updated_at REAL NOT NULL)")
    # миграция старых баз: признак «название сгенерировано автоматически»
    # (0 — переименовано пользователем, его LLM-тайтл больше не трогает)
    try:
        con.execute("ALTER TABLE chat_meta ADD COLUMN title_auto INTEGER NOT NULL DEFAULT 1")
    except sqlite3.OperationalError:
        pass
    return con


# ─────────────── хэширование паролей ───────────────

def _hash_password(password: str, salt_hex: str, iterations: int) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), iterations
    ).hex()


def _new_user_row(login: str, password: str) -> tuple:
    salt = secrets.token_hex(16)
    return (login, _hash_password(password, salt, PBKDF2_ITERATIONS),
            salt, PBKDF2_ITERATIONS, time.time())


def _env_users() -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in os.getenv("APP_USERS", "").split(","):
        if ":" in pair:
            login, password = pair.split(":", 1)
            if login.strip() and password.strip():
                out[login.strip()] = password.strip()
    return out


def _seed_env_users() -> None:
    """Однократно зашивает пары из APP_USERS в таблицу users (хэшем).

    Дальше пароль проверяется только по хэшу: .env можно почистить.
    Уже заведённого пользователя перезапись не трогает.
    """
    env_users = _env_users()
    if not env_users:
        return
    with _connect() as con:
        for login, password in env_users.items():
            row = con.execute("SELECT 1 FROM users WHERE login = ?",
                              (login,)).fetchone()
            if row is None:
                con.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                            _new_user_row(login, password))


def valid_login(login: str) -> bool:
    return bool(LOGIN_RE.match(login or ""))


def register(login: str, password: str) -> bool:
    """Новый пользователь. False — логин уже занят (или невалиден)."""
    if not valid_login(login) or len(password) < MIN_PASSWORD_LEN:
        return False
    _seed_env_users()
    with _connect() as con:
        try:
            con.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                        _new_user_row(login, password))
        except sqlite3.IntegrityError:
            return False
    return True


def login(login: str, password: str) -> str | None:
    """Вернуть токен при верном пароле, иначе None.

    Учётки берутся из SQLite (users): APP_USERS из .env лишь один раз
    зашивается сюда при старте (см. _seed_env_users).
    """
    _seed_env_users()
    with _connect() as con:
        row = con.execute(
            "SELECT pw_hash, pw_salt, iterations FROM users WHERE login = ?",
            (login,),
        ).fetchone()
    if row is None:
        # одновременно сравниваем с фиктивным хэшем — не подсказываем логин
        secrets.compare_digest("-", password)
        return None
    if not secrets.compare_digest(row[0], _hash_password(password, row[1], row[2])):
        return None
    token = secrets.token_hex(32)
    now = time.time()
    with _connect() as con:
        con.execute("DELETE FROM tokens WHERE expires < ?", (now,))
        con.execute("INSERT INTO tokens VALUES (?,?,?)", (token, login, now + TOKEN_TTL))
    return token


def logout(token: str | None) -> None:
    if token:
        with _connect() as con:
            con.execute("DELETE FROM tokens WHERE token = ?", (token,))


def user_of(token: str | None) -> str | None:
    if not token:
        return None
    with _connect() as con:
        row = con.execute(
            "SELECT login FROM tokens WHERE token = ? AND expires > ?",
            (token, time.time()),
        ).fetchone()
    return row[0] if row else None


# ─────────────── «логин → активная сессия диалога» (контекст переживает рестарт) ───────────────

def get_stored_session(login: str) -> str | None:
    with _connect() as con:
        row = con.execute(
            "SELECT session_id FROM user_sessions WHERE login = ?", (login,)).fetchone()
    return row[0] if row else None


def set_stored_session(login: str, session_id: str) -> None:
    with _connect() as con:
        con.execute("INSERT INTO user_sessions VALUES (?, ?) "
                    "ON CONFLICT(login) DO UPDATE SET session_id = excluded.session_id",
                    (login, session_id))


def clear_stored_session(login: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM user_sessions WHERE login = ?", (login,))


def all_stored_sessions() -> list[tuple[str, str]]:
    with _connect() as con:
        rows = con.execute("SELECT login, session_id FROM user_sessions").fetchall()
    return [(r[0], r[1]) for r in rows]


# ─────────────── метаданные чатов (сайдбар диалогов) ───────────────

DEFAULT_CHAT_TITLE = "Новый чат"


def ensure_chat_meta(login: str, session_id: str,
                     title: str = DEFAULT_CHAT_TITLE) -> None:
    """Заводит запись чата, если её ещё нет (идемпотентно)."""
    now = time.time()
    with _connect() as con:
        con.execute(
            "INSERT OR IGNORE INTO chat_meta"
            " (session_id, login, title, created_at, updated_at, title_auto)"
            " VALUES (?,?,?,?,?,1)",
            (session_id, login, title, now, now),
        )


def get_chat_meta(login: str) -> dict[str, dict]:
    """session_id → {title, created_at, updated_at, title_auto} для пользователя."""
    with _connect() as con:
        rows = con.execute(
            "SELECT session_id, title, created_at, updated_at, title_auto"
            " FROM chat_meta WHERE login = ? ORDER BY updated_at DESC",
            (login,),
        ).fetchall()
    return {r[0]: {"title": r[1], "created_at": r[2], "updated_at": r[3],
                   "title_auto": bool(r[4])}
            for r in rows}


def all_chat_meta() -> list[tuple[str, str]]:
    """(login, session_id) всех чатов — для чистки осиротевших записей."""
    with _connect() as con:
        rows = con.execute("SELECT login, session_id FROM chat_meta").fetchall()
    return [(r[0], r[1]) for r in rows]


def set_chat_title(login: str, session_id: str, title: str) -> bool:
    """Переименовать чат вручную (снимает флаг авто-названия).
    False — чат не найден у этого пользователя."""
    now = time.time()
    with _connect() as con:
        cur = con.execute(
            "UPDATE chat_meta SET title = ?, updated_at = ?, title_auto = 0"
            " WHERE session_id = ? AND login = ?",
            (title, now, session_id, login),
        )
    return cur.rowcount > 0


def set_auto_title(login: str, session_id: str, title: str) -> bool:
    """LLM-название чата: применяется только если название ещё авто
    (пользователь не переименовал вручную). False — трогать не стали."""
    now = time.time()
    with _connect() as con:
        cur = con.execute(
            "UPDATE chat_meta SET title = ?, updated_at = ?"
            " WHERE session_id = ? AND login = ? AND title_auto = 1",
            (title, now, session_id, login),
        )
    return cur.rowcount > 0


def touch_chat(login: str, session_id: str, question_title: str | None = None) -> None:
    """Обновить время чата; если название ещё дефолтное — подставить из вопроса.

    question_title передаётся только для первого вопроса чата: дефолтное
    название заменяется один раз и дальше не перетирается последующими
    вопросами (ручное название защищено флагом title_auto=0)."""
    now = time.time()
    with _connect() as con:
        if question_title:
            con.execute(
                "UPDATE chat_meta SET updated_at = ?,"
                " title = CASE WHEN title = ? THEN ? ELSE title END"
                " WHERE session_id = ? AND login = ?",
                (now, DEFAULT_CHAT_TITLE, question_title, session_id, login),
            )
        else:
            con.execute(
                "UPDATE chat_meta SET updated_at = ?"
                " WHERE session_id = ? AND login = ?",
                (now, session_id, login),
            )


def delete_chat_meta(login: str, session_id: str) -> bool:
    with _connect() as con:
        cur = con.execute(
            "DELETE FROM chat_meta WHERE session_id = ? AND login = ?",
            (session_id, login),
        )
    return cur.rowcount > 0
