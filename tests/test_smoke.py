import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "week2" / "practice" / "poi.sqlite"


def test_poi_database_has_rows() -> None:
    with sqlite3.connect(DB) as con:
        count = con.execute("SELECT COUNT(*) FROM poi").fetchone()[0]
    assert count > 0
