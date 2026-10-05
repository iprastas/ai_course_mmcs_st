"""Задание 1: прогоны десяти задач, метрики pass@1 / pass^3 и A/B описаний.

Запуск из корня репозитория:

    uv run python check.py --n 3          # основной прогон (вариант full)
    uv run python check.py --ab --n 3     # A/B: short против full, по 3 прогона

Все прогоны пишутся в week2/assignment/runs/*.jsonl.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import agent as agent_mod
import tasks as tasks_mod
from tasks import TASKS, Task, first_call_ok

RUNS_DIR = Path(__file__).resolve().parent / "runs"


# ═════════════════════════════════════════════════════════════ метрики

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Вильсона 95% для доли k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate(records: list[dict], tasks: list[Task]) -> None:
    """Проставляет passed и tool_ok каждой записи журнала."""
    by_id = {t.id: t for t in tasks}
    for rec in records:
        task = by_id[rec["task_id"]]
        rec["passed"] = bool(rec["final_answer"]) and task.judge(rec["final_answer"])
        rec["tool_ok"] = first_call_ok(task, rec["calls"])


def summarize(records: list[dict], n: int) -> dict:
    total = len(records)
    passed = sum(r["passed"] for r in records)
    tool_ok = sum(r["tool_ok"] for r in records)
    tasks_all = {r["task_id"] for r in records}
    all_pass = sum(
        all(r["passed"] for r in records if r["task_id"] == tid)
        for tid in tasks_all
    )
    return {
        "runs": total,
        "pass_at_1": passed / total if total else 0.0,
        "pass_k": all_pass / len(tasks_all) if tasks_all else 0.0,
        "pass_k_solved": all_pass,
        "tasks": len(tasks_all),
        "tool_ok": tool_ok / total if total else 0.0,
        "tool_ok_n": tool_ok,
        "avg_tokens": sum(r["tokens"] for r in records) / total if total else 0.0,
        "avg_in": sum(r["input_tokens"] for r in records) / total if total else 0.0,
        "avg_out": sum(r["output_tokens"] for r in records) / total if total else 0.0,
        "avg_cost": sum(r["cost_rub"] for r in records) / total if total else 0.0,
        "avg_steps": sum(r["steps"] for r in records) / total if total else 0.0,
        "errors": sum(r["status"] == "api_error" for r in records),
        "max_steps": sum(r["status"] == "max_steps" for r in records),
    }


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


# ═══════════════════════════════════════════════════════════ прогоны

def run_batch(loop: agent_mod.AgentLoop, variant: str, n: int,
              max_steps: int, label: str) -> list[dict]:
    records: list[dict] = []
    for i, task in enumerate(TASKS, 1):
        for run in range(1, n + 1):
            rec = loop.run(
                task.id, task.question, variant, run,
                expected={"tool": task.tool, "args": task.args},
                max_steps=max_steps,
            )
            records.append(rec)
            print(f"  [{label}] {i}/{len(TASKS)} {task.id} прогон {run}/{n}: "
                  f"{rec['status']}, {rec['steps']} шаг., {rec['tokens']} токенов, "
                  f"{rec['cost_rub']:.4f} ₽", flush=True)
    evaluate(records, TASKS)
    path = RUNS_DIR / f"{label}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    agent_mod.save_jsonl(path, records)
    print(f"журнал: {path}", flush=True)
    return records


def print_task_table(records: list[dict]) -> None:
    print(f"\n{'задача':<26} {'ответ':>8} {'инструмент':>12}  эталон")
    for task in TASKS:
        recs = [r for r in records if r["task_id"] == task.id]
        ok = sum(r["passed"] for r in recs)
        tok = sum(r["tool_ok"] for r in recs)
        mark = "✓" if ok == len(recs) else ("~" if ok else "✗")
        print(f"{task.id:<26} {mark} {ok}/{len(recs):<6} {tok}/{len(recs):<10} {task.truth}")


def print_summary(records: list[dict], n: int) -> dict:
    s = summarize(records, n)
    print("\n=== Итоги ===")
    print(f"pass@1:        {fmt_pct(s['pass_at_1'])}  "
          f"({sum(r['passed'] for r in records)}/{s['runs']} прогонов)")
    k_label = "pass^3" if n == 3 else f"pass^{n}"
    print(f"{k_label}:  {fmt_pct(s['pass_k'])}  "
          f"({s['pass_k_solved']}/{s['tasks']} задач со всеми {n} прогонами)")
    print(f"среднее на прогон: {s['avg_tokens']:.0f} токенов "
          f"(вход {s['avg_in']:.0f} / выход {s['avg_out']:.0f}), "
          f"стоимость {s['avg_cost']:.4f} ₽, шагов {s['avg_steps']:.1f}")
    if s["errors"]:
        print(f"ошибки среды: {s['errors']} прогон(ов)")
    if s["max_steps"]:
        print(f"упёрлись в лимит шагов: {s['max_steps']} прогон(ов)")
    return s


def print_ab(short: list[dict], full: list[dict], n: int) -> None:
    ss, fs = summarize(short, n), summarize(full, n)
    kn, kf = len(short), len(full)
    ci_s = wilson(ss["tool_ok_n"], kn)
    ci_f = wilson(fs["tool_ok_n"], kf)
    d_acc = fs["tool_ok"] - ss["tool_ok"]
    d_pass = fs["pass_at_1"] - ss["pass_at_1"]
    d_tok = fs["avg_tokens"] - ss["avg_tokens"]
    d_cost = fs["avg_cost"] - ss["avg_cost"]

    rows = [
        ("точность выбора инструмента и аргументов",
         f"{fmt_pct(ss['tool_ok'])} ({ss['tool_ok_n']}/{kn})",
         f"{fmt_pct(fs['tool_ok'])} ({fs['tool_ok_n']}/{kf})",
         f"{d_acc * 100:+.1f} п.п."),
        ("95% интервал Вильсона", fmt_ci(ci_s), fmt_ci(ci_f), ""),
        ("ответ верен (pass@1)",
         fmt_pct(ss["pass_at_1"]), fmt_pct(fs["pass_at_1"]),
         f"{d_pass * 100:+.1f} п.п."),
        ("средние токены на прогон",
         f"{ss['avg_tokens']:.0f}", f"{fs['avg_tokens']:.0f}", f"{d_tok:+.0f}"),
        ("средняя стоимость прогона, ₽",
         f"{ss['avg_cost']:.4f}", f"{fs['avg_cost']:.4f}", f"{d_cost:+.4f}"),
    ]

    print("\n=== A/B описаний инструментов ===")
    print(f"{'показатель':<44} {'short':>16} {'full':>16} {'Δ (full−short)':>16}")
    for name, s_cell, f_cell, d_cell in rows:
        print(f"{name:<44} {s_cell:>16} {f_cell:>16} {d_cell:>16}")

    overlap = not (ci_f[0] > ci_s[1] or ci_s[0] > ci_f[1])
    print("\nОтвет числом, окупились ли лишние токены описаний:")
    print(f"  точность: {d_acc * 100:+.1f} п.п., "
          f"токены на прогон: {d_tok:+.0f}, цена прогона: {d_cost:+.4f} ₽")
    if d_acc > 0:
        verdict = "окупились: полные описания дали прирост точности"
    elif d_acc < 0:
        verdict = "НЕ окупились: полные описания точность уменьшили"
    else:
        verdict = "НЕ окупились: точность не изменилась, лишние токены потрачены впустую"
    if overlap:
        verdict += " (но интервалы Вильсона перекрываются — разница пока не значима)"
    print(f"  вывод: {verdict}")


# ══════════════════════════════════════════════════════════════ CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Задание 1: прогоны и метрики")
    parser.add_argument("--n", type=int, default=1, help="сколько прогонов на задачу")
    parser.add_argument("--variant", choices=("short", "full"), default="full",
                        help="вариант описаний инструментов")
    parser.add_argument("--ab", action="store_true",
                        help="A/B: прогнать short и full по --n прогонов каждый")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="лимит шагов цикла на прогон")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # noqa: UP028 — кириллица в логах
    except Exception:  # noqa: BLE001
        pass

    load_dotenv()
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL")
               if not os.getenv(name)]
    if missing:
        print("нет переменных окружения:", ", ".join(missing))
        print("заполни .env по образцу .env.example")
        return 1

    price_in = float(os.getenv("PRICE_IN", "30"))
    price_out = float(os.getenv("PRICE_OUT", "120"))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"], timeout=60.0)
    loop = agent_mod.AgentLoop(client, os.environ["MODEL"],
                               price_in=price_in, price_out=price_out)

    print(f"модель: {os.environ['MODEL']} · тариф {price_in}/{price_out} ₽ за млн токенов · "
          f"задач: {len(TASKS)}, прогонов на задачу: {args.n}, "
          f"лимит шагов: {args.max_steps}")

    if args.ab:
        print("\n-- вариант short --")
        short = run_batch(loop, "short", args.n, args.max_steps,
                          f"ab_short_n{args.n}")
        print("\n-- вариант full --")
        full = run_batch(loop, "full", args.n, args.max_steps,
                         f"ab_full_n{args.n}")
        print_task_table(short)
        print_task_table(full)
        print_ab(short, full, args.n)
        return 0

    print(f"\n-- вариант {args.variant} --")
    records = run_batch(loop, args.variant, args.n, args.max_steps,
                        f"main_{args.variant}_n{args.n}")
    print_task_table(records)
    print_summary(records, args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
