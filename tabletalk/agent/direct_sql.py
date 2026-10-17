"""One model call that returns SQL. On an execution error, the original agent runs.

This is the candidate measured in the P1b holdout. It uses the model's settings
from the environment. The harness meter it was scored with stays in the anneal pack.
"""

import re

from tabletalk.config import settings
from tabletalk.llm.client import make_chat_model
from tabletalk.schema.index import load_schema, select_tables
from tabletalk.tools.sql import run_sql


def answer(db_path: str, question: str, original) -> dict:
    cards = load_schema(db_path)
    selection = select_tables(cards, question, settings.schema_token_budget)
    model = make_chat_model()
    prompt = (
        "Generate a SQL query to answer this question about a SQLite database.\n\n"
        f"Schema:\n{selection.text}\n\n"
        f"Question: {question}\n\n"
        "Return ONLY the SQL query, no explanation. Use SQLite syntax."
    )
    response = model.invoke([{"role": "user", "content": prompt}])
    sql_text = str(response.content or "").strip()
    if "```" in sql_text:
        match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", sql_text, re.DOTALL)
        if match:
            sql_text = match.group(1).strip()
    result = run_sql(db_path, sql_text, settings.max_rows, settings.sql_timeout_s)
    if not result.error and result.row_count >= 0:
        preview = result.preview(n=10)
        text = f"Found {result.row_count} rows.\n\n```sql\n{sql_text}\n```\n{preview}"
        return {"sql": sql_text, "answer": text, "fallback": False}
    model.invoke(
        [
            {
                "role": "user",
                "content": (
                    "This SQL query failed or returned suspicious results.\n"
                    f"Question: {question}\nSQL: {sql_text}\n"
                    f"Error: {result.error or 'No error but 0 rows'}\n\n"
                    "Should we fallback to the full agent? Answer only 'yes' or 'no'."
                ),
            }
        ]
    )
    turn = original(question)
    return {"sql": turn.sql, "answer": turn.answer, "fallback": True, "turn": turn}
