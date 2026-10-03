from __future__ import annotations
import json
import os
import socket
import ssl
import sys
import traceback
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

VERSION = "0.8.0"
BASE = Path(__file__).resolve().parent
LOGS = BASE / "logs"
LOGS.mkdir(exist_ok=True)
OUT = LOGS / "diagnose.txt"
API = "https://api.bybit.com/v5/market/kline"

def say(line=""):
    print(line, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

OUT.write_text("", encoding="utf-8")
say(f"Bybit Supertrend Lab v{VERSION} diagnostic | {datetime.now().isoformat(timespec='seconds')}")
say(f"Python: {sys.version}")
say(f"Executable: {sys.executable}")
say(f"Folder: {BASE}")
say(f"OS: {os.name}")
say()

try:
    ip = socket.gethostbyname("api.bybit.com")
    say(f"DNS api.bybit.com: OK -> {ip}")
except Exception as e:
    say(f"DNS api.bybit.com: ERROR -> {e}")

try:
    params = urllib.parse.urlencode({"category":"linear","symbol":"BTCUSDT","interval":"60","limit":2})
    url = API + "?" + params
    req = urllib.request.Request(url, headers={"User-Agent":f"BybitSupertrendLabDiagnostic/{VERSION}"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=12, context=ctx) as r:
        body = r.read().decode("utf-8", errors="replace")
        say(f"HTTP status: {getattr(r, 'status', 'unknown')}")
        say(f"URL: {url}")
        say(f"Body first 500 chars: {body[:500]}")
    payload = json.loads(body)
    if payload.get("retCode") == 0 and payload.get("result", {}).get("list"):
        say("BYBIT API: OK")
    else:
        say(f"BYBIT API: unexpected response -> {payload}")
except Exception as e:
    say(f"BYBIT API: ERROR -> {e}")
    say(traceback.format_exc())

say()
say(f"Saved: {OUT}")
