"""Авторизация MVP: пары логин:пароль из .env, токен в HttpOnly-куке.

В .env:  APP_USERS=anna:secret1,masha:secret2
Токены живут в памяти процесса — для MVP этого хватает: при рестарте
все просто входят заново.
"""

from __future__ import annotations

import os
import secrets

COOKIE_NAME = "rostov_session"

TOKENS: dict[str, str] = {}          # токен → логин


def _users() -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in os.getenv("APP_USERS", "admin:admin").split(","):
        if ":" in pair:
            login, password = pair.split(":", 1)
            out[login.strip()] = password.strip()
    return out


def login(login: str, password: str) -> str | None:
    """Вернуть токен при верном пароле, иначе None."""
    stored = _users().get(login)
    if stored is None:
        # одновременно сравниваем с фиктивным паролем — не подсказываем логин
        secrets.compare_digest("-", password)
        return None
    if not secrets.compare_digest(stored.encode(), password.encode()):
        return None
    token = secrets.token_hex(32)
    TOKENS[token] = login
    return token


def logout(token: str | None) -> None:
    if token:
        TOKENS.pop(token, None)


def user_of(token: str | None) -> str | None:
    if not token:
        return None
    return TOKENS.get(token)
