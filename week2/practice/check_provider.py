"""Проверка провайдера: один запрос к модели и проверка tool_calls и usage.

Без этих двух вещей практика курса не работает: модель должна уметь просить
вызвать инструмент и отдавать счётчик токенов.

Запуск из корня репозитория:

    uv run python week2/practice/check_provider.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "count_places",
        "description": "Сколько в городе мест такого вида. Возвращает одно число.",
        "parameters": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "вид места латиницей: pharmacy, cafe, outpost",
                }
            },
        },
    },
}


def main() -> int:
    load_dotenv()

    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL")
               if not os.getenv(name)]
    if missing:
        print("нет переменных окружения:", ", ".join(missing))
        print("заполни .env по образцу .env.example")
        return 1

    base_url = os.environ["OPENAI_BASE_URL"]
    model = os.environ["MODEL"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)

    print(f"сервис: {base_url}")
    print(f"модель: {model}")

    try:
        answer = client.chat.completions.create(
            model=model,
            tools=[TOOL_SCHEMA],
            messages=[{"role": "user", "content": "Сколько в Ростове аптек?"}],
        )
    except Exception as exc:  # noqa: BLE001 — показываем любую ошибку провайдера
        print("запрос не прошёл:", exc)
        return 1

    message = answer.choices[0].message
    has_tool_calls = bool(message.tool_calls)
    usage = answer.usage
    has_usage = usage is not None and usage.prompt_tokens is not None

    print("tool_calls:", "есть" if has_tool_calls else "НЕТ")
    if has_tool_calls:
        call = message.tool_calls[0]
        print(f"  вызов: {call.function.name}({call.function.arguments})")
    else:
        print("  ответ текстом:", message.content)

    if has_usage:
        print(f"usage: есть — {usage.prompt_tokens} вход, {usage.completion_tokens} выход")
    else:
        print("usage: НЕТ")

    problems = []
    if not has_tool_calls:
        problems.append("модель не умеет вызывать инструменты в формате tool_calls")
    if not has_usage:
        problems.append("в ответе нет usage — не посчитать токены")

    if problems:
        print("ПРОВЕРКА НЕ ПРОЙДЕНА:")
        for line in problems:
            print(" -", line)
        return 1

    print("ПРОВЕРКА ПРОЙДЕНА")
    return 0


if __name__ == "__main__":
    sys.exit(main())
