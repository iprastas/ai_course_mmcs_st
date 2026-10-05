"""Точка входа задания 1 из корня репозитория: uv run python check.py --n 3.

Переходник к week2/assignment/check.py — сам код задания лежит там же.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "week2" / "assignment"))

from check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
