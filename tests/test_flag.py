"""ANNEAL_TABLETALK leaves the original agent on the path."""

import json
from pathlib import Path

from tabletalk.agent.runner import Session

FIXTURES = Path(__file__).parent / "fixtures_optimization.jsonl"


def test_fixtures_from_the_optimization_set_are_present() -> None:
    text = FIXTURES.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(rows) == 42
    assert rows[0]["input"]["question"]
    assert rows[0]["input"]["db_id"]


def test_flag_off_calls_the_original(monkeypatch) -> None:
    monkeypatch.delenv("ANNEAL_TABLETALK", raising=False)
    monkeypatch.setattr(Session, "_ask_original", lambda self, question, on_step=None: "original")
    session = Session.__new__(Session)
    assert session.ask("How many customers are there?") == "original"


def test_flag_on_uses_direct_sql_and_returns_the_original_on_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ANNEAL_TABLETALK", "1")
    monkeypatch.setattr(Session, "_ask_original", lambda self, question, on_step=None: "original")

    def answer(db_path, question, original):
        assert original(question) == "original"
        return {"sql": "select 1", "answer": "1", "fallback": True, "turn": "original"}

    monkeypatch.setattr("tabletalk.agent.direct_sql.answer", answer)
    session = Session.__new__(Session)
    session.db_path = "db"
    assert session.ask("How many customers are there?") == "original"
