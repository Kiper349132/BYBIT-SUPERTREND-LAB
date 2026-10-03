"""Lightweight, thread-safe runtime log with size-based rotation.

Kept free of heavy imports so every module (including worker processes) may
use it without pulling numpy/pandas/GUI libraries.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# LAB_HOME relocates data/, results/, reports/, research/ and logs/ (tests use it
# so they never write into the program folder).
HOME_DIR = Path(os.environ.get("LAB_HOME", str(BASE_DIR)))
LOGS_DIR = HOME_DIR / "logs"
RUNTIME_LOG = LOGS_DIR / "runtime.log"
MAX_LOG_BYTES = 20 * 1024 * 1024
KEEP_ROTATED = 3

_lock = threading.Lock()


def _rotate_if_needed() -> None:
    try:
        if RUNTIME_LOG.exists() and RUNTIME_LOG.stat().st_size > MAX_LOG_BYTES:
            oldest = RUNTIME_LOG.with_name(f"runtime.log.{KEEP_ROTATED}")
            if oldest.exists():
                oldest.unlink()
            for i in range(KEEP_ROTATED - 1, 0, -1):
                src = RUNTIME_LOG.with_name(f"runtime.log.{i}")
                if src.exists():
                    os.replace(src, RUNTIME_LOG.with_name(f"runtime.log.{i + 1}"))
            os.replace(RUNTIME_LOG, RUNTIME_LOG.with_name("runtime.log.1"))
    except Exception:
        pass


def file_log(text: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}\n"
    with _lock:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed()
            with RUNTIME_LOG.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:
            # Logging must never crash the program, but the failure should not
            # disappear silently either.
            try:
                fallback = Path.home() / "BYBIT_SUPERTREND_LAB_runtime.log"
                with fallback.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.write(f"[{stamp}] PRIMARY LOG ERROR: {exc}\n")
            except Exception:
                pass


def summarize_payload(value, limit: int = 300) -> str:
    """Short description of a queue payload WITHOUT building its full repr.

    v0.7.1 called str() on whole result lists (tens of MB) only to keep the
    first 500 characters.
    """
    if isinstance(value, (list, tuple, set)):
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, dict):
        return f"<dict keys={list(value)[:12]}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        s = str(value)
        return s if len(s) <= limit else s[:limit] + "…"
    return f"<{type(value).__name__}>"
