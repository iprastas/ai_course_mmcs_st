"""Тесты задания 1: чистые функции без обращения к API."""

import sys
from pathlib import Path

ASSIGNMENT = Path(__file__).resolve().parents[1] / "week2" / "assignment"
sys.path.insert(0, str(ASSIGNMENT))

import agent  # noqa: E402
import check  # noqa: E402
import tasks  # noqa: E402
import tools  # noqa: E402


def test_wilson_known_value() -> None:
    lo, hi = check.wilson(10, 20)
    assert abs(lo - 0.299) < 0.02
    assert abs(hi - 0.701) < 0.02
    assert check.wilson(0, 10)[0] == 0.0
    assert check.wilson(10, 10)[1] == 1.0


def test_wilson_contains_point_estimate() -> None:
    for k, n in ((1, 3), (5, 7), (17, 30), (30, 30)):
        lo, hi = check.wilson(k, n)
        assert lo <= k / n <= hi


def test_ten_tasks_with_truth() -> None:
    assert len(tasks.TASKS) == 10
    for t in tasks.TASKS:
        assert t.question and t.tool and t.truth
        assert t.truth.strip() != ""


def test_judges_accept_truth_and_reject_wrong() -> None:
    for t in tasks.TASKS:
        # судья принимает правдоподобный ответ с эталонным фактом из базы
        assert t.judge(t.truth), f"судья не принял эталон задачи {t.id}"
        assert not t.judge("не знаю, спросите позже"), f"судья пропустил пустой ответ {t.id}"


def test_args_match() -> None:
    assert tasks.args_match({"kind": "cafe", "street": "Пушкин"},
                            {"kind": "cafe", "street": "Пушкинская улица"})
    assert not tasks.args_match({"kind": "cafe"}, {"kind": "pharmacy"})
    assert not tasks.args_match({"street": "Пушкин"}, {"street": "Доватора"})
    assert tasks.args_match({"lat": 47.2, "lon": 39.7},
                            {"lat": 47.2005, "lon": 39.6995})
    assert not tasks.args_match({"lat": 47.2}, {"lat": 47.5})
    assert tasks.args_match({"only_24_7": True}, {"kind": "pharmacy", "only_24_7": True})
    assert not tasks.args_match({"only_24_7": True}, {"kind": "pharmacy"})
    assert not tasks.args_match({"only_24_7": False}, {"kind": "pharmacy", "only_24_7": True})
    assert tasks.args_match({"only_24_7": False}, {"kind": "pharmacy"})


def test_first_call_ok() -> None:
    t = next(x for x in tasks.TASKS if x.id == "count_24_7")
    assert tasks.first_call_ok(t, [{"name": "count_places",
                                    "args": {"kind": "pharmacy", "only_24_7": True}}])
    assert not tasks.first_call_ok(t, [])
    assert not tasks.first_call_ok(t, [{"name": "find_places", "args": {"kind": "pharmacy"}}])
    assert not tasks.first_call_ok(t, [{"name": "count_places", "args": {"kind": "pharmacy"}}])


def test_schemas_two_variants() -> None:
    short = tools.schemas("short")
    full = tools.schemas("full")
    assert len(short) == len(full) == 5
    names = {s["function"]["name"] for s in short}
    assert names == {"count_places", "find_places", "top_brands", "nearest", "details"}
    for s, f in zip(short, full):
        assert s["function"]["name"] == f["function"]["name"]
        # варианты отличаются только описаниями
        assert s["function"]["description"] != f["function"]["description"]
        assert set(s["function"]["parameters"]["properties"]) == \
               set(f["function"]["parameters"]["properties"])
        assert s["function"]["parameters"]["required"] == \
               f["function"]["parameters"]["required"]
    try:
        tools.schemas("medium")
    except ValueError:
        pass
    else:
        raise AssertionError("неизвестный вариант должен отклоняться")


def test_tools_work_without_model() -> None:
    assert "330" in tools.count_places("pharmacy")
    assert "19" in tools.count_places("pharmacy", only_24_7=True)
    assert "ничего не нашлось" in tools.find_places("pharmacy", street="Несуществующая")
    assert "|" in tools.top_brands("outpost")
    spot = tools.nearest(47.2685, 39.7875, "school")
    assert "id=" in spot and " м" in spot
    assert "часы:" in tools.details(43)
    assert "ошибка" in tools.count_places("аптека")


def test_error_observations_and_max_steps() -> None:
    rec_ok, _ = agent.AgentLoop.__new__(agent.AgentLoop)._execute("count_places", '{"kind": "cafe"}')
    assert "кафе" in rec_ok or "cafe" in rec_ok
    rec_bad, call = agent.AgentLoop.__new__(agent.AgentLoop)._execute("count_places", "{битый json")
    assert "ошибка" in rec_bad and call["error"]
    rec_unknown, _ = agent.AgentLoop.__new__(agent.AgentLoop)._execute("nope", "{}")
    assert "нет" in rec_unknown


def test_bare_json_is_repaired() -> None:
    # типичный битый JSON от модели: без кавычек у кириллицы и у ключей
    assert agent._parse_args('{"kind": cafe, "street": Пушкин}')[0] == \
        {"kind": "cafe", "street": "Пушкин"}
    assert agent._parse_args("{kind: 'Пушкин'}")[0] == {"kind": "Пушкин"}
    assert agent._parse_args("{'kind': 'pharmacy'}")[0] == {"kind": "pharmacy"}
    # валидный JSON возвращается как есть, флаг починки не ставится
    assert agent._parse_args('{"kind": "cafe"}') == ({"kind": "cafe"}, False)
    # числа, булевы и None остаются своими типами
    assert agent._parse_args('{"lat": 47.2685, "only_24_7": True, "x": None}')[0] == \
        {"lat": 47.2685, "only_24_7": True, "x": None}
    # отремонтированный вызов реально исполняется и не считается ошибкой
    rec, call = agent.AgentLoop.__new__(agent.AgentLoop)._execute(
        "count_places", '{"kind": pharmacy, "only_24_7": True}')
    assert call["ok"] and "ошибка" not in rec
    # неподдевающийся мусор по-прежнему ошибка с пояснением для модели
    rec, call = agent.AgentLoop.__new__(agent.AgentLoop)._execute(
        "count_places", "просто текст")
    assert call["error"] and "корректного вызова" in rec


def test_truncated_args_recovered_from_markup() -> None:
    # провайдер обрезает аргументы посреди значения, полный вызов — в content
    truncated = '{"kind": "cafe", "street": '
    open_tag = "<" + "parameter name="
    close_tag = "</" + "parameter>"
    content = (f"резонирование…\n{open_tag}\"kind\">cafe{close_tag}\n"
               f"{open_tag}\"street\">Пушкин{close_tag}\n"
               f"{close_tag}")
    args, repaired = agent._parse_args(truncated, content)
    assert args == {"kind": "cafe", "street": "Пушкин"}
    assert repaired
    # форма провайдера: parameter без пробела, через равенство
    eq_content = ("резонирование…\n" + "<" + "parameter=kind" + ">cafe" + close_tag + "\n"
                  + "<" + "parameter=street" + ">Пушкин" + close_tag + "\n" + close_tag)
    assert agent._parse_args(truncated, eq_content)[0] == {"kind": "cafe", "street": "Пушкин"}
    # аргументы починены — вызов проходит, ошибки нет
    rec, call = agent.AgentLoop.__new__(agent.AgentLoop)._execute(
        "find_places", truncated, content)
    assert call["ok"] and call["repaired"] and "ошибка" not in rec
    # обрезанный JSON без content-дубля чинится до того, как сдаться
    assert agent._parse_args('{"kind": "cafe", "street": "Пуш')[0] == \
        {"kind": "cafe", "street": "Пуш"}
