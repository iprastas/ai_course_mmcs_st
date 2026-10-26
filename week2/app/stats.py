"""Метрики задания 1 для веб-страницы /stats.

Читает журналы week2/assignment/runs/*.jsonl (те же файлы, что считает
check.py): pass@1, pass^3, долю верных первых вызовов с интервалом
Вильсона и A/B двух вариантов описаний.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parents[1] / "assignment" / "runs"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал Вильсона 95% для доли k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _latest(prefix: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    found = sorted(RUNS_DIR.glob(prefix + "*.jsonl"))
    return found[-1] if found else None


def _load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def summarize(records: list[dict]) -> dict[str, Any] | None:
    if not records:
        return None
    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    passed = sum(1 for r in records if r.get("passed"))
    tool_ok = sum(1 for r in records if r.get("tool_ok"))
    n = len(records)
    solved = sum(1 for rs in by_task.values() if all(r.get("passed") for r in rs))
    return {
        "runs": n,
        "tasks": len(by_task),
        "pass_at_1": round(passed / n, 4),
        "pass_at_1_n": passed,
        "pass_k": round(solved / len(by_task), 4),
        "pass_k_solved": solved,
        "tool_ok": round(tool_ok / n, 4),
        "tool_ok_n": tool_ok,
        "tool_ok_ci": [round(x, 4) for x in wilson(tool_ok, n)],
        "avg_tokens": round(sum(r.get("tokens", 0) for r in records) / n, 1),
        "avg_cost": round(sum(r.get("cost_rub", 0) for r in records) / n, 4),
        "avg_steps": round(sum(r.get("steps", 0) for r in records) / n, 2),
        "errors": sum(1 for r in records if r.get("status") == "api_error"),
        "max_steps": sum(1 for r in records if r.get("status") == "max_steps"),
    }


def _file_info(path: Path | None) -> str:
    return path.name if path else "—"


def main_metrics() -> dict[str, Any]:
    path = _latest("main_full_n")
    data = summarize(_load(path)) if path else None
    return {"file": _file_info(path), "summary": data}


def ab_metrics() -> dict[str, Any]:
    short_path, full_path = _latest("ab_short_n"), _latest("ab_full_n")
    if not short_path or not full_path:
        return {"file": None, "short": None, "full": None}
    short = summarize(_load(short_path))
    full = summarize(_load(full_path))
    if not short or not full:
        return {"file": None, "short": short, "full": full}
    d_acc = round((full["tool_ok"] - short["tool_ok"]) * 100, 1)
    d_pass = round((full["pass_at_1"] - short["pass_at_1"]) * 100, 1)
    d_tok = round(full["avg_tokens"] - short["avg_tokens"], 1)
    d_cost = round(full["avg_cost"] - short["avg_cost"], 4)
    ci_s, ci_f = short["tool_ok_ci"], full["tool_ok_ci"]
    overlap = not (ci_f[0] > ci_s[1] or ci_s[0] > ci_f[1])
    if d_acc > 0:
        verdict = "окупились: полные описания дали прирост точности"
    elif d_acc < 0:
        verdict = "НЕ окупились: полные описания точность уменьшили"
    else:
        verdict = "НЕ окупились: точность не изменилась, лишние токены потрачены впустую"
    if not overlap:
        verdict += "; интервалы Вильсона не перекрываются — разница значима"
    else:
        verdict += "; интервалы Вильсона перекрываются — разница пока не значима"
    return {
        "file": f"{short_path.name} + {full_path.name}",
        "short": short, "full": full,
        "delta": {"acc_pp": d_acc, "pass_pp": d_pass, "tokens": d_tok, "cost": d_cost},
        "overlap": overlap, "verdict": verdict,
    }


def all_metrics() -> dict[str, Any]:
    """Сводка для /api/metrics. Если журналов нет — summary будет None."""
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main": main_metrics(),
        "ab": ab_metrics(),
    }
