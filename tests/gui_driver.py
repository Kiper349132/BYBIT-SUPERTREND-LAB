"""Drives the real Tk GUI (run under Xvfb by tests/test_gui_close.py).

mode=close   start research, then invoke the window manager close protocol
             (exactly what the X button does) while the search is running.
mode=sigint  same, but deliver SIGINT (Ctrl+C in the console) to the process.
mode=resume  reopen, answer YES to "continue unfinished run", wait for the end
             and dump the final TOP / report pool.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]



def main():
    # Heavy imports only in the real main process: spawned workers re-import
    # this file as __mp_main__ and must not pull in the GUI.
    import lab_storage
    lab_storage.CHECKPOINT_USEFUL_EVERY = 400
    import lab_core as core
    import lab_gui
    from lab_data import market_to_df
    from resume_helpers import synthetic_set
    mode = sys.argv[1]
    out_path = Path(sys.argv[2])
    events = []
    lab_gui.messagebox.askyesnocancel = lambda *a, **k: (events.append("ask_resume"), mode == "resume")[1]
    lab_gui.messagebox.showerror = lambda *a, **k: events.append(("error", a))
    lab_gui.messagebox.showinfo = lambda *a, **k: events.append(("info", a[0]))
    lab_gui.messagebox.showwarning = lambda *a, **k: events.append(("warning", a))

    app = lab_gui.App()
    app.symbol_var.set("SYNTH")
    app.months_var.set("12")
    app.depth_var.set("Тест")
    app.workers_var.set("2")
    for tf, var in app.tf_vars.items():
        var.set(tf in ("30", "60"))
    cfg = app._read_config()
    market = synthetic_set()
    app.data_by_tf = {tf: market_to_df(m) for tf, m in market.items()}
    app.loaded_key = app._config_key(cfg)
    state = {"t0": time.monotonic(), "close_at": None, "closed_by": None}

    def tick():
        if mode in ("close", "sigint") and state["close_at"] is None and app.status_checked > 3000:
            state["close_at"] = time.monotonic()
            if mode == "close":
                state["closed_by"] = "WM_DELETE_WINDOW"
                app.tk.call(app.protocol("WM_DELETE_WINDOW"))     # the real X-button path
            else:
                state["closed_by"] = "SIGINT"
                os.kill(os.getpid(), signal.SIGINT)
        if mode == "resume" and app.top_results and not (app.worker and app.worker.is_alive()):
            out_path.write_text(json.dumps({
                "events": events, "cfg": {k: v for k, v in cfg.items() if k not in ("start_dt", "end_dt")},
                "top": [core.autoresult_to_dict(r) for r in app.top_results],
                "report_pool": [core.autoresult_to_dict(r) for r in app.all_results],
                "report": str(app.last_report_path), "summary": app.last_research_summary,
            }, default=list))
            app._on_close()
            return
        if time.monotonic() - state["t0"] > 300:
            events.append("timeout")
            app._destroy()
            return
        app.after(50, tick)


    app.after(100, app.start_auto_search)
    app.after(200, tick)
    app.mainloop()
    if mode != "resume":
        out_path.write_text(json.dumps({
            "events": events, "closed_by": state["closed_by"],
            "seconds_from_close_to_exit": round(time.monotonic() - (state["close_at"] or time.monotonic()), 3),
            "cfg": {k: v for k, v in cfg.items() if k not in ("start_dt", "end_dt")},
            "research_dirs": [str(p) for p in sorted(lab_gui.RESEARCH_DIR.iterdir())],
        }, default=list))


if __name__ == "__main__":
    main()
