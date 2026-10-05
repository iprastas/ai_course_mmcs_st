"""Агентный цикл без фреймворка: model → tool_calls → наблюдения → model.

Что здесь сделано руками вместо Runner:
  - лимит шагов (max_steps): исчерпан — прогон закрыт со статусом max_steps;
  - ошибки как наблюдения: неизвестный инструмент, битый JSON аргументов,
    исключение внутри инструмента превращаются в строку-результат для модели,
    цикл не падает и модель получает шанс исправиться;
  - починка типичного битого JSON (без кавычек у значений) до разбора —
    модель часто присылает {"street": Пушкин}; если починить не вышло,
    в наблюдение попадают сами аргументы модели и пример корректного JSON;
  - счётчики токенов из usage каждого запроса + цена по тарифу;
  - журнал в JSONL: одна строка на прогон, включая вызовы и ответы модели.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

import tools as tools_mod

SYSTEM_PROMPT = (
    "Ты — справочник по базе мест Ростова-на-Дону (8860 мест из OpenStreetMap). "
    "Отвечай на вопросы пользователя только с помощью инструментов: "
    "сначала вызови подходящий инструмент, затем дай краткий ответ по его результату. "
    "Числа, адреса и часы не выдумывай — бери только из ответов инструментов. "
    "Если инструмент вернул ошибку, поправь аргументы и вызови его снова. "
    "Отвечай по-русски, коротко и по делу."
)

RESULT_LIMIT = 4000        # обрезка результата инструмента в наблюдении
TEXT_LIMIT = 400           # обрезка ответов модели в журнале

_BARE_KEY = re.compile(r'([{,]\s*)([^"\s,\]{}:]+)(\s*:)')
_BARE_VALUE = re.compile(r'([:{\[]\s*)([^"\s,\]{}][^,\]}]*?)(?=\s*[,\]}])')


def _quote_value(m: re.Match) -> str:
    v = m.group(2).strip()
    low = v.lower()
    if low in ("true", "false"):
        return m.group(1) + low
    if low in ("null", "none"):
        return m.group(1) + "null"
    if re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", v):
        return m.group(1) + v
    escaped = v.replace("\\", "\\\\").replace('"', '\\"')
    return f'{m.group(1)}"{escaped}"'


def _repair(text: str) -> str:
    """Кавычки вокруг голых ключей и значений: {kind: Пушкин} → {"kind": "Пушкин"}."""
    text = _BARE_KEY.sub(r'\1"\2"\3', text)
    return _BARE_VALUE.sub(_quote_value, text)


def _complete_json(text: str) -> str | None:
    """Закрыть незакрытую строку/объект: провайдер иногда обрезает аргументы
    вызова посреди значения ({"street": ← конец строки), а полный вызов
    модель дополнительно пишет в content."""
    out = text.rstrip()
    if out.count('"') % 2:
        out += '"'
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in out:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if in_str:
        return None
    return out + "".join(reversed(stack))


_TAG_CLOSE = re.escape("</" + "parameter>")
_XML_NAME_ARG = re.compile(
    r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)" + _TAG_CLOSE, re.S)
_XML_EQ_ARG = re.compile(r"<parameter=([^\s>]+)>(.*?)" + _TAG_CLOSE, re.S)


def _args_from_markup(text: str) -> dict | None:
    """Полные аргументы из текста модели со специфичной разметкой вызова:
    провайдер превращает её в tool_calls, но иногда обрезает аргументы.
    Поддерживаются обе формы записи: <parameter name=... > и <parameter=...>."""
    if not text or "<parameter" not in text:
        return None
    args: dict[str, Any] = {}
    for pattern in (_XML_NAME_ARG, _XML_EQ_ARG):
        for name, value in pattern.findall(text):
            value = value.strip()
            if not value:
                continue
            try:
                args[name] = json.loads(value)
            except json.JSONDecodeError:
                args[name] = value
    return args or None


def _parse_args(raw: str, content: str | None = None) -> tuple[dict, bool]:
    """Разбор аргументов вызова; битый или обрезанный JSON чинится.

    Возвращает (аргументы, был ли вызов починен) — фиксация в журнале.
    """
    text = (raw or "").strip() or "{}"
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')           # одинарные кавычки вместо двойных
    candidates = [text, _repair(text)]
    completed = _complete_json(_repair(text))
    if completed is not None:
        candidates.append(completed)
    last: Exception = ValueError("аргументы пусты")
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last = exc
            continue
        if isinstance(obj, dict):
            return obj, candidate != text
        last = ValueError("аргументы должны быть JSON-объектом")
    markup = _args_from_markup(content)
    if markup is not None:
        return markup, True
    raise last


class AgentLoop:
    def __init__(self, client: OpenAI, model: str,
                 price_in: float = 30.0, price_out: float = 120.0):
        self.client = client
        self.model = model
        self.price_in = price_in      # ₽ за млн входных токенов
        self.price_out = price_out    # ₽ за млн выходных

    def _complete(self, messages: list[dict], schemas: list[dict], tries: int = 3):
        """Запрос к модели с ретраем: сетевые сбои и rate limit не роняют прогон."""
        last: Exception | None = None
        for attempt in range(tries):
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=schemas)
            except Exception as exc:  # noqa: BLE001 — пробуем ещё раз, потом сдаемся
                last = exc
                if attempt < tries - 1:
                    time.sleep(2 * (attempt + 1))
        raise last

    def run(self, task_id: str, question: str, variant: str, run_index: int,
            expected: dict[str, Any], max_steps: int = 8) -> dict:
        """Один прогон задачи. Возвращает запись журнала (словарь)."""
        schemas = tools_mod.schemas(variant)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        calls: list[dict] = []
        assistant_texts: list[str] = []
        input_tokens = output_tokens = 0
        steps = 0
        status = "max_steps"
        final_answer = ""
        api_error = None
        started = time.time()

        for step in range(1, max_steps + 1):
            try:
                resp = self._complete(messages, schemas)
            except Exception as exc:  # noqa: BLE001 — ошибка среды, фиксируем и выходим
                status = "api_error"
                api_error = f"{type(exc).__name__}: {exc}"
                break

            steps = step
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                output_tokens += getattr(usage, "completion_tokens", 0) or 0

            msg = resp.choices[0].message
            if msg.content:
                assistant_texts.append(str(msg.content)[:TEXT_LIMIT])

            if not msg.tool_calls:
                status = "ok"
                final_answer = (msg.content or "").strip()
                break

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                observation, call = self._execute(tc.function.name, tc.function.arguments,
                                                  msg.content)
                call["step"] = step
                calls.append(call)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": observation})

        cost = (input_tokens / 1e6 * self.price_in
                + output_tokens / 1e6 * self.price_out)
        return {
            "task_id": task_id,
            "run": run_index,
            "variant": variant,
            "question": question,
            "expected": expected,
            "status": status,
            "api_error": api_error,
            "steps": steps,
            "calls": calls,
            "assistant_texts": assistant_texts,
            "final_answer": final_answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens": input_tokens + output_tokens,
            "cost_rub": round(cost, 4),
            "seconds": round(time.time() - started, 1),
        }

    def _execute(self, name: str, raw_args: str,
                 content: str | None = None) -> tuple[str, dict]:
        """Вызов инструмента. Любая ошибка превращается в наблюдение для модели."""
        record: dict[str, Any] = {"name": name, "args": None, "ok": False, "error": None,
                                  "raw_args": (raw_args or "")[:500], "repaired": False}

        if name not in tools_mod.TOOLS:
            available = ", ".join(tools_mod.TOOLS)
            record["error"] = "unknown_tool"
            observation = f"ошибка: инструмента {name!r} нет; доступны: {available}"
            return observation, record

        try:
            args, repaired = _parse_args(raw_args, content)
        except (json.JSONDecodeError, ValueError) as exc:
            record["error"] = f"bad_args: {exc}"
            shown_args = (raw_args or "")[:200]
            observation = (f"ошибка: твои аргументы не разбираются как JSON-объект: "
                           f"{shown_args}\n"
                           f"причина: {exc}\n"
                           f'пример корректного вызова: '
                           f'{{"kind": "cafe", "street": "Пушкин"}} — '
                           f"ключи и строковые значения в двойных кавычках; "
                           f"вызови инструмент снова с исправленными аргументами")
            return observation, record

        record["args"] = args
        record["repaired"] = repaired
        try:
            result = tools_mod.TOOLS[name](**args)
        except Exception as exc:  # noqa: BLE001 — наблюдение вместо падения цикла
            record["error"] = f"{type(exc).__name__}: {exc}"
            observation = (f"ошибка при выполнении {name}: {type(exc).__name__}: {exc}; "
                           f"поправь аргументы и вызови снова")
            return observation, record

        record["ok"] = True
        observation = str(result)[:RESULT_LIMIT]
        record["result"] = observation[:TEXT_LIMIT]
        return observation, record


def save_jsonl(path, records: list[dict]) -> None:
    """Дозаписывает прогоны в JSONL: одна строка — один прогон."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
