"""Session: one database, one conversation thread, many questions."""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from tabletalk.agent.graph import build_graph
from tabletalk.config import settings
from tabletalk.trace.store import RunTrace, Step, new_run_id, write_trace


@dataclass
class Turn:
    answer: str
    status: str  # answered | clarify | failed
    sql: str | None
    chart_path: str | None
    trace: RunTrace


class Session:
    def __init__(self, db_path: str, model=None, thread_id: str | None = None,
                 persist: bool = False, embed=None, write_traces: bool = True):
        self.db_path = str(db_path)
        self.thread_id = thread_id or uuid.uuid4().hex[:8]
        self.write_traces = write_traces
        if model is None:
            from tabletalk.llm.client import make_chat_model

            model = make_chat_model()
        if persist:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        else:
            checkpointer = MemorySaver()
        self.graph = build_graph(model, checkpointer=checkpointer, embed=embed)

    def ask(self, question: str, on_step: Callable[[Step], None] | None = None) -> Turn:
        if os.environ.get("ANNEAL_TABLETALK"):
            from tabletalk.agent.direct_sql import answer

            result = answer(
                self.db_path,
                question,
                lambda text: self._ask_original(text, on_step),
            )
            if result.get("fallback"):
                return result["turn"]
            trace = RunTrace(run_id=new_run_id(), thread_id=self.thread_id, question=question)
            trace.status = "answered"
            trace.answer = result["answer"]
            return Turn(
                answer=result["answer"],
                status="answered",
                sql=result.get("sql"),
                chart_path=None,
                trace=trace,
            )
        return self._ask_original(question, on_step)

    def _ask_original(self, question: str, on_step: Callable[[Step], None] | None = None) -> Turn:
        trace = RunTrace(run_id=new_run_id(), thread_id=self.thread_id, question=question)
        if on_step:
            trace.on_step = on_step  # type: ignore[attr-defined]
        config = {"configurable": {"thread_id": self.thread_id, "tracer": trace},
                  "recursion_limit": 40}
        state = self.graph.invoke(
            {"messages": [HumanMessage(question)], "question": question,
             "db_path": self.db_path},
            config,
        )
        trace.status = state.get("status", "unknown")
        trace.answer = state.get("answer", "")
        if self.write_traces:
            write_trace(trace, settings.trace_dir)
        return Turn(
            answer=state.get("answer", ""),
            status=state.get("status", "unknown"),
            sql=state.get("last_sql"),
            chart_path=state.get("chart_path"),
            trace=trace,
        )
