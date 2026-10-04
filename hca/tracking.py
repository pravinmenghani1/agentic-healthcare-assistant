"""Tool-call logging for LLMOps: every tool invocation records args, success/failure and latency."""
import contextvars
import functools
import json
import time

from . import db

current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_run_id", default=None)


class ToolError(Exception):
    """A tool ran but could not fulfil the request (e.g. no slots, patient not found)."""


def _preview(value, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def logged_tool(name: str):
    """Decorator: log each call to tool_logs. A result dict with ok=False counts as a failure."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            start = time.perf_counter()
            success, error, result = True, None, None
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and result.get("ok") is False:
                    success, error = False, result.get("error")
                return result
            except Exception as exc:
                success, error = False, f"{type(exc).__name__}: {exc}"
                raise
            finally:
                db.execute(
                    "INSERT INTO tool_logs (run_id, tool, args, success, error, latency_ms, output_preview, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (current_run_id.get() or "ui", name, _preview({"args": args, "kwargs": kwargs}, 1000), int(success),
                     error, int((time.perf_counter() - start) * 1000), _preview(result), db.now()))
        inner.tool_name = name
        return inner
    return wrap
