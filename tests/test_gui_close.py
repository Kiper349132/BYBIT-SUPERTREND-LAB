"""Real Tk GUI: the window close button and Ctrl+C perform the same safe stop
as the STOP button, and the GUI resume reproduces the uninterrupted result."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lab_search import ResearchRunner
from lab_storage import ResearchStore
from resume_helpers import canonical, diff_summary, synthetic_set

HERE = Path(__file__).resolve().parent
XVFB = shutil.which("xvfb-run")

try:
    import tkinter  # noqa: F401
    HAVE_TK = True
except ImportError:
    HAVE_TK = False

pytestmark = pytest.mark.skipif(not HAVE_TK or (XVFB is None and not os.environ.get("DISPLAY")),
                                reason="tkinter or X display not available")


def _drive(mode, home, out):
    env = dict(os.environ, LAB_HOME=str(home))
    cmd = [sys.executable, str(HERE / "gui_driver.py"), mode, str(out)]
    if not os.environ.get("DISPLAY"):
        cmd = [XVFB, "-a", *cmd]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(out.read_text())


@pytest.mark.parametrize("mode", ["close", "sigint"])
def test_close_or_ctrl_c_saves_checkpoint_and_gui_resume_matches(tmp_path, mode):
    home = tmp_path / "home"
    first = _drive(mode, home, tmp_path / "first.json")
    assert first["closed_by"] in ("WM_DELETE_WINDOW", "SIGINT")
    assert first["seconds_from_close_to_exit"] < 30
    assert not [e for e in first["events"] if e and e[0] == "error"], first["events"]
    dirs = [Path(d) for d in first["research_dirs"]]
    assert len(dirs) == 1 and not (dirs[0] / "COMPLETE.json").exists()
    store = ResearchStore.open(dirs[0])
    assert store.committed_useful > 0
    log = (home / "logs" / "runtime.log").read_text(encoding="utf-8")
    assert "CLOSE: requested while busy" in log and "CLOSE: worker finished, exiting" in log

    # Reopen the GUI, accept "continue unfinished run", let it finish.
    second = _drive("resume", home, tmp_path / "second.json")
    assert "ask_resume" in second["events"]
    assert second["summary"]["status"] == "complete" and second["summary"]["session_dir"] == str(dirs[0])
    assert Path(second["report"]).exists()

    # Reference: one uninterrupted run with the same configuration and candles.
    cfg = second["cfg"]
    ref_store = ResearchStore.create(tmp_path / "ref", cfg, synthetic_set())
    ref = ResearchRunner(cfg, ref_store, workers=3).run()
    a = canonical(ref.top, ref.report_pool, {})
    b = canonical(second["top"], second["report_pool"], {})
    assert a == b, diff_summary(a, b)
