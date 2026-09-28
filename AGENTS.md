# AGENTS.md

Оглавление для агента для кода. Короткие инструкции, не энциклопедия.

## Команды

- Зависимости: `uv sync` (Python 3.12, окружение в `.venv`).
- Тесты: `uv run pytest`.
- Проверка провайдера: `uv run python week2/practice/check_provider.py`.
- Ноутбук практики: `week2/practice/week02_agents.ipynb` — ядро `.venv`, рабочая директория `week2/practice` (иначе `import sdk_tasks` не находится). Первая ячейка ноутбука (`pip install` + `git clone` + `%cd`) — для Colab, локально не запускать.

## Где что лежит

- `week2/practice/sdk_tasks.py` — данные, инструменты и проверки практики; студент его не правит.
- `week2/practice/poi.sqlite` — база мест Ростова; пересобирается из `week2/practice/data/rostov_poi.json` командой `python make_poi_dataset.py`.
- `week2/practice/check_provider.py` — проверка, что сервис отдаёт `tool_calls` и `usage`.

## Не трогать

- `.env` — секреты, не читать и не коммитить; в коммит не попадает (`.gitignore`).
- `.env.example` — только без значений ключа.
- Слайды `week*/slides/`, изображения `week2/practice/img/`.
