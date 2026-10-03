"""Bybit Supertrend Lab launcher.

This file must stay tiny. On Windows every worker process re-imports the
launched script (as __mp_main__) before it can run a task; in v0.7.1 that
re-imported tkinter, matplotlib, pandas and requests in every worker. All GUI
code now lives in lab_gui.py and is imported only when the script really runs
as the main program. Workers import just lab_core (numpy + numba).
"""
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    try:
        import lab_gui
    except ImportError as exc:  # pragma: no cover - user environment problem
        raise SystemExit(
            "Не хватает библиотек. Запустите START.bat или выполните: pip install -r requirements.txt\n"
            f"Ошибка: {exc}"
        )
    lab_gui.main()
