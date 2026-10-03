"""Tkinter GUI of Bybit Supertrend Lab (parent process only).

Worker processes never import this module: they only import lab_core.
"""
from __future__ import annotations

import gc
import math
import os
import queue
import signal
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from lab_core import AutoResult, NUMBA_AVAILABLE, VERSION
from lab_data import (
    AUTO_INTERVALS, StopRequested, download_bybit_klines, df_to_market, find_reusable_candles, market_to_df, save_candles,
)
from lab_log import LOGS_DIR, RUNTIME_LOG, file_log, summarize_payload
from lab_report import (
    backtest_desired_with_trades, build_selected_signal, build_walk_forward_plan, create_chatgpt_report, save_results,
)
from lab_search import ResearchRunner, recommended_workers
from lab_storage import CHECKPOINT_USEFUL_EVERY, ResearchStore

APP_NAME = "Bybit Supertrend Lab"
BASE_DIR = Path(__file__).resolve().parent
from lab_log import HOME_DIR  # noqa: E402
DATA_DIR = HOME_DIR / "data"
RESEARCH_DIR = HOME_DIR / "research"
for _d in (DATA_DIR, RESEARCH_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def fmt_seconds(seconds) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class PlotWindow(tk.Toplevel):
    def __init__(self, master, title: str, fig: Figure):
        super().__init__(master)
        self.title(title)
        self.geometry("1050x720")
        self.minsize(780, 540)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="Panel.TFrame")
        self.canvas = tk.Canvas(self, bg="#151c25", highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Panel.TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _on_inner(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_wheel(self, event):
        try:
            x, y = self.winfo_pointerxy()
            widget = self.winfo_containing(x, y)
            if widget is not None and str(widget).startswith(str(self)):
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1420x860")
        self.minsize(900, 620)
        self.configure(bg="#0b0f14")

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.data_by_tf: dict[str, pd.DataFrame] = {}
        self.loaded_key = None
        self.all_results: list[AutoResult] = []
        self.top_results: list[AutoResult] = []
        self.selected_row: Optional[AutoResult] = None
        self.last_config = {}
        self.cpu_count = max(1, os.cpu_count() or 1)
        self.is_busy = False
        self.busy_started_at = 0.0
        self.last_ui_activity = time.monotonic()
        self.search_started_at = 0.0
        self.last_search_seconds = 0.0
        self.search_stats: list[dict] = []
        self.last_report_path: Optional[Path] = None
        self.last_research_summary: dict = {}
        self.status_phase = "ГОТОВО"
        self.status_checked = 0
        self.status_total = 0
        self.status_speed = 0.0
        self.status_eta = "—"
        self.status_workers = 0
        self.last_result_files: list = []
        self.close_requested = False
        self.close_requested_at = 0.0
        self.active_task = ""

        file_log(f"START {APP_NAME} v{VERSION} | base={BASE_DIR} | python={os.sys.executable}")
        self._setup_style()
        self._build_ui()
        # The window close button (X) and Ctrl+C in the console go through the
        # same safe path as the STOP button: stop -> checkpoint -> exit.
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            signal.signal(signal.SIGINT, lambda *_: self.after(0, self._on_close))
        except Exception:
            pass
        self.after(100, self._drain_queue)
        self.after(500, self._heartbeat)
        self.after(250, lambda: self._log(f"Запущена версия {VERSION} · журнал: {RUNTIME_LOG}"))

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background="#0b0f14", foreground="#e7edf5", font=("Segoe UI", 10))
        style.configure("TFrame", background="#0b0f14")
        style.configure("Panel.TFrame", background="#151c25")
        style.configure("TLabel", background="#0b0f14", foreground="#dbe5ef")
        style.configure("Panel.TLabel", background="#151c25", foreground="#dbe5ef")
        style.configure("Muted.TLabel", background="#0b0f14", foreground="#91a0b2")
        style.configure("BigStatus.TLabel", background="#151c25", foreground="#ffffff", font=("Segoe UI Semibold", 12))
        style.configure("Title.TLabel", background="#0b0f14", foreground="#ffffff", font=("Segoe UI Semibold", 20))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 11), padding=(12, 10))
        style.configure("Danger.TButton", font=("Segoe UI", 10), padding=(10, 8))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("TEntry", fieldbackground="#0e141b", foreground="#ffffff", insertcolor="#ffffff", padding=6)
        style.configure("TCombobox", fieldbackground="#0e141b", foreground="#ffffff", padding=5)
        style.configure("TCheckbutton", background="#151c25", foreground="#dbe5ef")
        style.configure("Treeview", background="#111821", fieldbackground="#111821", foreground="#e7edf5", rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background="#263242", foreground="#ffffff", font=("Segoe UI Semibold", 9), relief="flat")
        style.map("Treeview", background=[("selected", "#314c70")], foreground=[("selected", "#ffffff")])
        style.configure("Horizontal.TProgressbar", troughcolor="#202a36", background="#4b8edb")
        style.configure("TLabelframe", background="#151c25", foreground="#ffffff")
        style.configure("TLabelframe.Label", background="#151c25", foreground="#ffffff", font=("Segoe UI Semibold", 10))

    def _build_ui(self):
        # Grid keeps the status panel visible at every window size.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="BYBIT SUPERTREND LAB", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="selection → validation · поиск в ширину и глубину · Adaptive V2", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(7, 0))
        ttk.Label(header, text="БЭКТЕСТ · ОРДЕРА НЕ ОТКРЫВАЕТ", style="Muted.TLabel").grid(row=0, column=2, sticky="e", pady=(7, 0))

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 6))
        left_host = ttk.Frame(main, style="Panel.TFrame", width=370)
        right = ttk.Frame(main, style="Panel.TFrame")
        main.add(left_host, weight=0)
        main.add(right, weight=1)

        left_scroll = ScrollableFrame(left_host)
        left_scroll.pack(fill=tk.BOTH, expand=True)
        self._build_left(left_scroll.inner)
        self._build_right(right)
        self._build_bottom()

    def _build_left(self, parent):
        box = ttk.LabelFrame(parent, text="Автоматический поиск", padding=10)
        box.pack(fill=tk.X, padx=8, pady=(8, 5))

        self.symbol_var = tk.StringVar(value="BTCUSDT")
        self.months_var = tk.StringVar(value="18")
        self.fee_var = tk.StringVar(value="0.055")
        self.slippage_var = tk.StringVar(value="0.02")
        self.depth_var = tk.StringVar(value="Глубокий")
        self._labeled_entry(box, "Инструмент", self.symbol_var, 0)
        self._labeled_combo(box, "История, месяцев", self.months_var, ["6", "12", "18", "24", "36"], 1)

        ttk.Label(box, text="РЕЖИМ ПОИСКА", style="Panel.TLabel").grid(row=2, column=0, sticky="nw", padx=12, pady=(7, 4))
        depth_frame = ttk.Frame(box, style="Panel.TFrame")
        depth_frame.grid(row=2, column=1, sticky="ew", padx=12, pady=(4, 2))
        for i, (value, caption) in enumerate((
            ("Стандартный", "СТАНДАРТНЫЙ"),
            ("Глубокий", "ГЛУБОКИЙ"),
            ("Максимальный", "МАКСИМАЛЬНЫЙ"),
        )):
            ttk.Radiobutton(depth_frame, text=caption, variable=self.depth_var, value=value).grid(row=i, column=0, sticky="w", pady=2)
        ttk.Label(
            box, text="Глубокий — рекомендуемый. LONG / SHORT / BOTH программа проверяет сама.",
            style="Muted.TLabel", wraplength=210, justify=tk.LEFT,
        ).grid(row=3, column=1, sticky="w", padx=12, pady=(0, 6))

        ttk.Label(box, text="Таймфреймы", style="Panel.TLabel").grid(row=4, column=0, sticky="nw", padx=12, pady=6)
        tf_frame = ttk.Frame(box, style="Panel.TFrame")
        tf_frame.grid(row=4, column=1, sticky="ew", padx=12, pady=4)
        self.tf_vars = {}
        defaults = {"5", "15", "30", "60"}
        for i, tf in enumerate(AUTO_INTERVALS):
            var = tk.BooleanVar(value=tf in defaults)
            self.tf_vars[tf] = var
            ttk.Checkbutton(tf_frame, text=f"{tf}м", variable=var).grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 12), pady=2)

        self._labeled_entry(box, "Комиссия / сторона, %", self.fee_var, 5)
        self._labeled_entry(box, "Проскальзывание / сторона, %", self.slippage_var, 6)
        box.columnconfigure(1, weight=1)

        research = ttk.LabelFrame(parent, text="Новая логика проверки", padding=9)
        research.pack(fill=tk.X, padx=8, pady=5)
        ttk.Label(
            research,
            text=(
                "• Последний участок истории — финальный HOLDOUT. Он не влияет на выбор.\n"
                "• До него идут последовательные walk-forward окна по 21 дню.\n"
                "• Обычный Supertrend и адаптивная версия сравниваются одновременно.\n"
                "• Адаптивная версия: свой Supertrend для роста и падения; во флэте — пауза.\n"
                "• Режим рынка определяется только по прошлым закрытым свечам.\n"
                "• Отчёт отдельно показывает выживших на HOLDOUT и устойчивость соседних параметров."
            ),
            style="Panel.TLabel", justify=tk.LEFT, wraplength=315,
        ).pack(anchor="w")

        cpu_box = ttk.LabelFrame(parent, text="Процессор", padding=9)
        cpu_box.pack(fill=tk.X, padx=8, pady=5)
        auto_workers, why = recommended_workers(0, self.cpu_count)
        self.workers_var = tk.StringVar(value=str(auto_workers))
        ttk.Label(cpu_box, text=f"Windows видит: {self.cpu_count} логических потоков", style="Panel.TLabel").pack(anchor="w")
        ttk.Label(cpu_box, text=f"Процессов (уточняется перед запуском по RAM): {auto_workers}", style="Panel.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(cpu_box, text=why, style="Muted.TLabel", wraplength=315, justify=tk.LEFT).pack(anchor="w", pady=(2, 0))
        ttk.Label(cpu_box, text=f"Ускорение Numba: {'ВКЛЮЧЕНО' if NUMBA_AVAILABLE else 'НЕТ (будет заметно медленнее)'}", style="Panel.TLabel").pack(anchor="w", pady=(2, 0))

        action = ttk.Frame(parent, style="Panel.TFrame")
        action.pack(fill=tk.X, padx=8, pady=(7, 5))
        self.download_btn = ttk.Button(action, text="1. СКАЧАТЬ ИСТОРИЮ", command=self._on_download_click, style="Accent.TButton")
        self.download_btn.pack(fill=tk.X)
        self.auto_btn = ttk.Button(action, text="2. ЗАПУСТИТЬ ИССЛЕДОВАНИЕ", command=self.start_auto_search, style="Accent.TButton")
        self.auto_btn.pack(fill=tk.X, pady=(7, 0))
        self.report_btn = ttk.Button(action, text="3. СОЗДАТЬ ОТЧЁТ ДЛЯ CHATGPT", command=self.start_report_export)
        self.report_btn.pack(fill=tk.X, pady=(7, 0))
        self.stop_btn = ttk.Button(action, text="ОСТАНОВИТЬ", command=self.request_stop, state=tk.DISABLED, style="Danger.TButton")
        self.stop_btn.pack(fill=tk.X, pady=(7, 0))

        hint = ttk.LabelFrame(parent, text="Какую глубину выбирать", padding=8)
        hint.pack(fill=tk.X, padx=8, pady=(5, 10))
        ttk.Label(
            hint,
            text=(
                "Стандартный — быстрая проверка.\n"
                "Глубокий — основной режим, рассчитан на большой поиск.\n"
                "Максимальный — значительно тяжелее; имеет смысл после того, как Глубокий покажет устойчивые области."
            ), style="Panel.TLabel", wraplength=315, justify=tk.LEFT,
        ).pack(anchor="w")

    def _build_right(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        top = ttk.Frame(parent, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 5))
        ttk.Label(top, text="ТОП-100: research-рейтинг BASE + ADAPTIVE V2", style="Panel.TLabel").pack(side=tk.LEFT)
        ttk.Button(top, text="Тепловая карта", command=self.show_heatmap).pack(side=tk.RIGHT)
        ttk.Button(top, text="Кривая капитала", command=self.show_equity).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(top, text="Сделки", command=self.show_trades).pack(side=tk.RIGHT, padx=(0, 5))

        table_host = ttk.Frame(parent, style="Panel.TFrame")
        table_host.grid(row=1, column=0, sticky="nsew", padx=8)
        table_host.rowconfigure(0, weight=1)
        table_host.columnconfigure(0, weight=1)
        cols = ("rank", "tf", "mode", "atr", "mult", "sel", "val", "score", "wf", "median", "research", "holdout", "pf", "dd", "trades")
        self.tree = ttk.Treeview(table_host, columns=cols, show="headings", selectmode="browse")
        labels = {
            "rank":"#", "tf":"TF", "mode":"Режим", "atr":"ATR", "mult":"Множ.", "sel":"Selection", "val":"Validation", "score":"Итог",
            "wf":"+ окон", "median":"Медиана окна %", "research":"Research %", "holdout":"Dev holdout %",
            "pf":"PF", "dd":"Просадка %", "trades":"Сделок"
        }
        widths = {"rank":42,"tf":50,"mode":80,"atr":60,"mult":70,"sel":78,"val":78,"score":78,"wf":74,"median":105,"research":82,"holdout":98,"pf":58,"dd":82,"trades":70}
        for c in cols:
            self.tree.heading(c, text=labels[c])
            self.tree.column(c, width=widths[c], anchor=tk.CENTER, stretch=False)
        yscroll = ttk.Scrollbar(table_host, orient=tk.VERTICAL, command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_host, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        detail = ttk.LabelFrame(parent, text="Подробности выбранного варианта", padding=7)
        detail.grid(row=2, column=0, sticky="ew", padx=8, pady=(5, 8))
        self.detail_text = tk.Text(detail, height=7, bg="#0e141b", fg="#dce7f3", insertbackground="#ffffff", relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD)
        self.detail_text.pack(fill=tk.X)
        self.detail_text.insert("1.0", "После исследования здесь появятся SELECTION, VALIDATION, development holdout и результаты по режимам рынка.")
        self.detail_text.configure(state=tk.DISABLED)

    def _build_bottom(self):
        # Fixed bottom row: it never disappears behind the main content.
        panel = ttk.Frame(self, style="Panel.TFrame")
        panel.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        panel.columnconfigure(0, weight=1)
        self.big_status_var = tk.StringVar(value="ГОТОВО · выберите параметры и запустите исследование")
        ttk.Label(panel, textvariable=self.big_status_var, style="BigStatus.TLabel", anchor="w").grid(row=0, column=0, sticky="ew", padx=9, pady=(7, 4))
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", padx=9)

        stat = ttk.Frame(panel, style="Panel.TFrame")
        stat.grid(row=2, column=0, sticky="ew", padx=9, pady=(4, 2))
        for i in range(6):
            stat.columnconfigure(i, weight=1)
        self.stage_var = tk.StringVar(value="Этап: —")
        self.checked_var = tk.StringVar(value="Проверок: 0")
        self.speed_var = tk.StringVar(value="Скорость: —")
        self.elapsed_var = tk.StringVar(value="Время: 00:00")
        self.eta_var = tk.StringVar(value="Осталось: —")
        self.cpu_var = tk.StringVar(value="CPU: —")
        for i, var in enumerate([self.stage_var, self.checked_var, self.speed_var, self.elapsed_var, self.eta_var, self.cpu_var]):
            ttk.Label(stat, textvariable=var, style="Panel.TLabel", anchor="w").grid(row=0, column=i, sticky="ew", padx=(0, 8))

        info = ttk.Frame(panel, style="Panel.TFrame")
        info.grid(row=3, column=0, sticky="ew", padx=9, pady=(1, 2))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        self.data_var = tk.StringVar(value="История ещё не загружена")
        self.counter_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.data_var, style="Panel.TLabel", anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Label(info, textvariable=self.counter_var, style="Panel.TLabel", anchor="e").grid(row=0, column=1, sticky="ew")

        self.log_text = tk.Text(panel, height=4, bg="#0b1016", fg="#aebdcd", relief=tk.FLAT, font=("Consolas", 8), wrap=tk.WORD)
        self.log_text.grid(row=4, column=0, sticky="ew", padx=9, pady=(2, 8))
        self.log_text.configure(state=tk.DISABLED)

    def _labeled_entry(self, parent, label, variable, row):
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", padx=12, pady=5)
        ttk.Entry(parent, textvariable=variable, width=16).grid(row=row, column=1, sticky="ew", padx=12, pady=5)

    def _labeled_combo(self, parent, label, variable, values, row):
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", padx=12, pady=5)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=14).grid(row=row, column=1, sticky="ew", padx=12, pady=5)

    def _selected_tfs(self):
        return [tf for tf in AUTO_INTERVALS if self.tf_vars[tf].get()]

    def _read_config(self):
        symbol = self.symbol_var.get().upper().strip()
        if not symbol:
            raise ValueError("Укажите инструмент, например BTCUSDT")
        months = int(self.months_var.get())
        tfs = self._selected_tfs()
        if not tfs:
            raise ValueError("Выберите хотя бы один таймфрейм")
        fee = float(self.fee_var.get().replace(",", ".")) / 100.0
        slippage = float(self.slippage_var.get().replace(",", ".")) / 100.0
        if not (0 <= fee <= 0.02):
            raise ValueError("Комиссия выглядит ошибочной. Например: 0.055")
        if not (0 <= slippage <= 0.02):
            raise ValueError("Проскальзывание выглядит ошибочным. Например: 0.02")
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=round(months * 30.4375))
        workers = max(1, min(32, int(self.workers_var.get()), self.cpu_count))
        depth = self.depth_var.get().strip() if hasattr(self, "depth_var") else "Глубокий"
        return {
            "symbol": symbol, "months": months, "tfs": tfs, "fee": fee, "slippage": slippage,
            "start_dt": start_dt, "end_dt": end_dt, "workers": workers, "depth": depth,
        }

    def _config_key(self, cfg):
        return (cfg["symbol"], cfg["months"], tuple(cfg["tfs"]))

    def _set_busy(self, busy: bool):
        self.is_busy = busy
        if busy:
            self.busy_started_at = time.monotonic()
            self.last_ui_activity = time.monotonic()
            self.status_checked = 0
            self.status_total = 0
            self.status_speed = 0.0
            self.status_eta = "—"
            self.elapsed_var.set("Время: 00:00") if hasattr(self, "elapsed_var") else None
            try:
                self.progress.stop()
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
            except Exception:
                pass
        else:
            try:
                self.progress.stop()
                self.progress.configure(mode="determinate")
            except Exception:
                pass
        self.download_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.auto_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.report_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if busy else tk.DISABLED)
        if not busy:
            self.download_btn.configure(text="1. СКАЧАТЬ ИСТОРИЮ")

    def _post(self, kind, *payload):
        # Only meaningful events are logged, and never the repr of big objects
        # (v0.7.1 built a ~60 MB string per finished search just to log 500 chars).
        if kind not in ("progress", "search_status"):
            try:
                file_log(f"QUEUE {kind}: " + " | ".join(summarize_payload(x) for x in payload))
            except Exception:
                pass
        self.msg_queue.put((kind, payload))

    def _log(self, text: str):
        file_log(text)
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{stamp}] {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                self.last_ui_activity = time.monotonic()
                try:
                    if kind == "status":
                        self.big_status_var.set(payload[0])
                        if hasattr(self, "stage_var"):
                            short_phase = str(payload[0]).split("·")[0].strip()
                            self.stage_var.set(f"Этап: {short_phase}")
                        if len(payload) > 1 and payload[1]:
                            self._log(payload[1])
                    elif kind == "progress":
                        pct = float(payload[0])
                        if pct > 0:
                            try:
                                self.progress.stop()
                                self.progress.configure(mode="determinate")
                            except Exception:
                                pass
                            self.progress["value"] = pct
                        if len(payload) > 1:
                            self.counter_var.set(payload[1])
                    elif kind == "search_status":
                        info = payload[0] if payload else {}
                        phase = str(info.get("phase", "ПОИСК"))
                        self.stage_var.set(f"Этап: {phase}")
                        checked = int(info.get("checked", 0) or 0)
                        total = int(info.get("total", 0) or 0)
                        self.status_checked = checked
                        self.status_total = total
                        if total > 0:
                            self.checked_var.set(f"Проверок: {checked:,} / {total:,}")
                        else:
                            self.checked_var.set(f"Проверок: {checked:,}")
                        speed = float(info.get("speed", 0.0) or 0.0)
                        self.status_speed = speed
                        self.speed_var.set(f"Скорость: {speed:,.0f}/с" if speed > 0 else "Скорость: —")
                        eta = info.get("eta")
                        self.status_eta = fmt_seconds(float(eta)) if eta is not None else "—"
                        self.eta_var.set(f"Осталось: {self.status_eta}")
                        workers = int(info.get("workers", 0) or 0)
                        self.status_workers = workers
                        self.cpu_var.set(f"CPU: {workers} проц.") if workers else self.cpu_var.set("CPU: —")
                        detail = str(info.get("detail", ""))
                        if detail:
                            self.counter_var.set(detail)
                    elif kind == "data_set":
                        self.data_by_tf, self.loaded_key = payload
                    elif kind == "data_done":
                        self.data_by_tf, self.loaded_key = payload
                        self._after_data_loaded()
                    elif kind == "search_done":
                        res, paths, report_path, summary, stats, seconds, data = payload
                        self.all_results, self.top_results = res.report_pool, res.top
                        self.last_report_path = Path(report_path) if report_path else None
                        self.last_research_summary = summary
                        self.search_stats = stats
                        self.last_search_seconds = seconds
                        self.last_result_files = list(paths)
                        if data is not None:
                            self.data_by_tf = data
                        self._after_search(paths)
                    elif kind == "report_done":
                        self.last_report_path = Path(payload[0])
                        self._set_busy(False)
                        self.progress["value"] = 100
                        self.big_status_var.set("ОТЧЁТ ДЛЯ CHATGPT ГОТОВ")
                        self.counter_var.set(self.last_report_path.name)
                        self._log(f"Отчёт для ChatGPT: {self.last_report_path}")
                        messagebox.showinfo("Отчёт готов", f"Создан один ZIP-файл:\n\n{self.last_report_path}\n\nПрикрепите его к сообщению в ChatGPT.")
                    elif kind == "error":
                        self._set_busy(False)
                        if self.close_requested:
                            self._log(payload[0]); continue
                        self.big_status_var.set("ОШИБКА · подробности записаны в logs")
                        self._log(payload[0])
                        err_path = LOGS_DIR / "last_error.txt"
                        try:
                            err_path.write_text(payload[0], encoding="utf-8")
                        except Exception:
                            pass
                        messagebox.showerror("Ошибка", payload[0])
                    elif kind == "stopped":
                        self._set_busy(False)
                        msg = payload[0] if payload else "Операция остановлена пользователем"
                        self.big_status_var.set("ОСТАНОВЛЕНО · CHECKPOINT СОХРАНЁН" if payload else "ОСТАНОВЛЕНО ПОЛЬЗОВАТЕЛЕМ")
                        self._log(msg)
                        if len(payload) > 1 and payload[1] == "memory_stop" and not self.close_requested:
                            messagebox.showwarning("Защитная остановка", msg)
                    elif kind == "summary":
                        self.last_research_summary = payload[0]
                except Exception as exc:
                    file_log(f"QUEUE HANDLER ERROR kind={kind}: {exc}\n{traceback.format_exc(limit=5)}")
        except queue.Empty:
            pass
        except Exception as exc:
            file_log(f"QUEUE DRAIN ERROR: {exc}\n{traceback.format_exc(limit=5)}")
        finally:
            try:
                self.after(100, self._drain_queue)
            except tk.TclError:
                pass

    def _heartbeat(self):
        try:
            if self.is_busy:
                elapsed = max(0.0, time.monotonic() - self.busy_started_at)
                self.elapsed_var.set(f"Время: {fmt_seconds(elapsed)}")
                worker_state = "работает" if (self.worker and self.worker.is_alive()) else "завершается"
                if self.status_speed > 0:
                    self.speed_var.set(f"Скорость: {self.status_speed:,.0f}/с")
                    self.eta_var.set(f"Осталось: {self.status_eta}")
                if time.monotonic() - self.last_ui_activity > 3.0:
                    self.counter_var.set(f"Операция {worker_state} · подробности: logs/runtime.log")
            self.after(500, self._heartbeat)
        except tk.TclError:
            pass
        except Exception as exc:
            file_log(f"HEARTBEAT ERROR: {exc}")
            try:
                self.after(1000, self._heartbeat)
            except Exception:
                pass

    def request_stop(self):
        self.stop_event.set()
        self.big_status_var.set("ОСТАНАВЛИВАЮ · сохраняю checkpoint…")
        self._log("Запрошена остановка: расчётные процессы завершаются, готовые блоки записываются на диск.")

    def _on_close(self):
        """Window X / Ctrl+C: same safe path as the STOP button, then exit."""
        busy = bool(self.worker and self.worker.is_alive())
        if not busy:
            file_log("CLOSE: idle, exiting")
            self._destroy()
            return
        if self.close_requested:
            # second request while stopping: offer forced exit (still crash-safe:
            # only the not yet committed part would be recomputed later)
            if time.monotonic() - self.close_requested_at > 1.0 and messagebox.askyesno(
                    "Закрыть немедленно?",
                    "Идёт безопасное сохранение checkpoint. Закрыть программу немедленно?\n\n"
                    "Уже записанные на диск результаты не пострадают; незаписанный пакет будет пересчитан при продолжении."):
                file_log("CLOSE: forced by user")
                os._exit(3)
            return
        self.close_requested = True
        self.close_requested_at = time.monotonic()
        file_log(f"CLOSE: requested while busy ({self.active_task}); performing safe stop")
        self.request_stop()
        self.big_status_var.set("ЗАКРЫТИЕ · сохраняю checkpoint, окно закроется автоматически…")
        self.after(200, self._close_when_idle)

    def _close_when_idle(self):
        if self.worker and self.worker.is_alive():
            self.after(200, self._close_when_idle)
            return
        try:
            self._drain_once()
        except Exception:
            pass
        file_log("CLOSE: worker finished, exiting")
        self._destroy()

    def _drain_once(self):
        while True:
            try:
                kind, payload = self.msg_queue.get_nowait()
            except queue.Empty:
                return
            if kind in ("status", "error", "stopped"):
                file_log(f"CLOSE drain {kind}: " + " | ".join(summarize_payload(x) for x in payload))

    def _destroy(self):
        try:
            self.quit()
            self.destroy()
        except Exception:
            pass

    def _download_all(self, cfg):
        data = {}
        tfs = cfg["tfs"]
        for idx, tf in enumerate(tfs):
            if self.stop_event.is_set():
                raise StopRequested()
            self._post("status", f"СКАЧИВАЮ {tf} МИН · {idx+1}/{len(tfs)}", f"Начинаю загрузку {cfg['symbol']} {tf}м")

            def cb(local_pct, msg, idx=idx, tf=tf):
                overall = ((idx + local_pct / 100.0) / len(tfs)) * 100.0
                self._post("progress", overall, f"Загрузка {tf}м · {local_pct:.1f}%")
                self._post("status", f"ЗАГРУЗКА BYBIT · {msg}")
                self._post("search_status", {
                    "phase": f"Загрузка {tf}м ({idx+1}/{len(tfs)})",
                    "checked": 0, "total": 0, "speed": 0, "eta": None, "workers": 0,
                    "detail": f"{local_pct:.1f}% · {msg}",
                })

            reused_df, reused_path = find_reusable_candles(cfg["symbol"], tf, cfg["start_dt"], cfg["end_dt"], DATA_DIR)
            if reused_df is not None:
                data[tf] = reused_df
                self._post("status", f"{tf} МИН · ИСПОЛЬЗУЮ СОХРАНЁННУЮ ИСТОРИЮ", f"{reused_path.name}: {len(reused_df):,} свечей · повторно не скачиваю")
                continue

            partial = DATA_DIR / f"{cfg['symbol']}_{tf}m_DOWNLOADING.csv"
            self._post("status", f"СКАЧИВАЮ {tf} МИН · создаю временный файл", f"Промежуточные данные: {partial.name}")
            df = download_bybit_klines(
                cfg["symbol"], tf, cfg["start_dt"], cfg["end_dt"], cb, self.stop_event, partial_path=partial
            )
            path = save_candles(df, cfg["symbol"], tf, DATA_DIR)
            df.attrs["source_path"] = str(path)
            try:
                partial.unlink(missing_ok=True)
            except Exception as exc:
                file_log(f"PARTIAL cleanup error {partial}: {exc}")
            data[tf] = df
            self._post("status", f"{tf} МИН ГОТОВО · {len(df):,} свечей", f"Сохранено: {path.name}")
        return data

    def _on_download_click(self):
        """GUI wrapper: a click must never fail silently."""
        file_log("CLICK download button")
        try:
            self.download_btn.configure(text="НАЖАТО · ЗАПУСКАЮ…")
            self.big_status_var.set("КНОПКА НАЖАТА · ПРОВЕРЯЮ ПАРАМЕТРЫ…")
            self.counter_var.set("Подготовка загрузки…")
            self.progress["value"] = 0
            self.update_idletasks()
            self.start_download_only()
        except Exception as exc:
            details = f"Ошибка обработчика кнопки: {exc}\n\n{traceback.format_exc(limit=8)}"
            file_log(details)
            try:
                self._set_busy(False)
                self.big_status_var.set("ОШИБКА ПРИ НАЖАТИИ КНОПКИ")
                self._log(details)
                messagebox.showerror("Ошибка запуска загрузки", details)
            except Exception:
                pass

    def start_download_only(self):
        file_log("ENTER start_download_only")
        if self.worker and self.worker.is_alive():
            self._log("Загрузка не запущена: уже выполняется другая операция")
            self.big_status_var.set("УЖЕ ВЫПОЛНЯЕТСЯ ДРУГАЯ ОПЕРАЦИЯ")
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            messagebox.showerror("Параметры", str(exc))
            return
        self.stop_event.clear()
        self._set_busy(True)
        self.progress["value"] = 0
        self.counter_var.set("")
        self.big_status_var.set("ПОДКЛЮЧАЮСЬ К BYBIT…")
        self.counter_var.set("Создаю рабочий поток загрузки…")
        self.update_idletasks()
        self._log(f"Загрузка истории {cfg['symbol']}: {', '.join(x+'м' for x in cfg['tfs'])}, {cfg['months']} мес.")

        def run():
            file_log("DOWNLOAD WORKER enter")
            try:
                data = self._download_all(cfg)
                self._post("data_done", data, self._config_key(cfg))
                file_log("DOWNLOAD WORKER success")
            except StopRequested:
                file_log("DOWNLOAD WORKER stopped")
                self._post("stopped")
            except Exception as exc:
                details = f"{exc}\n\n{traceback.format_exc(limit=8)}"
                file_log(f"DOWNLOAD WORKER error: {details}")
                self._post("error", details)
            finally:
                file_log("DOWNLOAD WORKER exit")

        self.worker = threading.Thread(target=run, daemon=True, name="BybitDownloadWorker")
        self.worker.start()
        file_log(f"THREAD started name={self.worker.name} alive={self.worker.is_alive()}")
        self.counter_var.set("Рабочий поток запущен · соединяюсь с api.bybit.com…")


    def _after_data_loaded(self):
        self._set_busy(False)
        self.progress.configure(mode="determinate")
        self.progress["value"] = 100
        total = sum(len(df) for df in self.data_by_tf.values())
        detail = " · ".join(f"{tf}м: {len(df):,}" for tf, df in sorted(self.data_by_tf.items(), key=lambda x: int(x[0])))
        self.data_var.set(f"Загружено {total:,} свечей · {detail}")
        self.counter_var.set("")
        self.big_status_var.set("ИСТОРИЯ ЗАГРУЖЕНА · можно запускать исследование")
        self.stage_var.set("Этап: история готова")
        self.checked_var.set(f"Свечей: {total:,}")
        self.speed_var.set("Скорость: —")
        self.eta_var.set("Осталось: —")
        self.cpu_var.set("CPU: —")
        self._log("Загрузка истории полностью завершена")

    def _on_runner_status(self, d: dict):
        if d.get("log"):
            self._post("status", f"{d.get('phase', '')}", d["log"])
        if d.get("progress") is not None:
            self._post("progress", float(d["progress"]), d.get("detail", ""))
        self._post("search_status", d)

    def start_auto_search(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            messagebox.showerror("Параметры", str(exc))
            return
        resume_root = None
        try:
            incomplete = ResearchStore.find_incomplete(RESEARCH_DIR, cfg)
        except Exception as exc:
            file_log(f"find_incomplete failed: {exc}")
            incomplete = []
        if incomplete:
            root = incomplete[0]
            ans = messagebox.askyesnocancel(
                "Незавершённое исследование",
                f"Найден незавершённый прогон с теми же настройками:\n{root.name}\n\n"
                "ДА — продолжить его. Будут использованы свечи, сохранённые в этом прогоне, "
                "поэтому результат совпадёт с непрерывным расчётом.\n"
                "НЕТ — начать новый прогон на актуальной истории.\n"
                "ОТМЕНА — ничего не делать.")
            if ans is None:
                return
            if ans:
                resume_root = root

        self.stop_event.clear()
        self._set_busy(True)
        self.active_task = "research"
        self.progress["value"] = 0
        self.tree.delete(*self.tree.get_children())
        self.all_results = []
        self.top_results = []
        self.selected_row = None
        self.last_config = cfg.copy()
        self.search_started_at = time.monotonic()
        self.last_search_seconds = 0.0
        self.search_stats = []
        current_data = dict(self.data_by_tf)
        need_download = self.loaded_key != self._config_key(cfg) or any(tf not in current_data for tf in cfg["tfs"])
        self._log(f"{cfg['depth']} автопоиск: {cfg['symbol']} · TF {', '.join(cfg['tfs'])}" + (f" · ПРОДОЛЖЕНИЕ {resume_root.name}" if resume_root else ""))

        def run():
            store = None
            try:
                if resume_root is not None:
                    store = ResearchStore.open(resume_root)
                    market = store.load_market()
                    data = {tf: market_to_df(market[tf]) for tf in market}
                    notes = "; ".join(store.recovery_notes) or "без замечаний"
                    self._post("status", "НАЙДЕН CHECKPOINT · ПРОДОЛЖАЮ ИССЛЕДОВАНИЕ",
                               f"Уже на диске: {store.committed_useful:,} проверок · {store.committed_rows:,} результатов · "
                               f"{len(store.chunks)} chunk-файлов · восстановление: {notes}")
                else:
                    data = current_data
                    if need_download:
                        self._post("status", "СНАЧАЛА СКАЧИВАЮ НУЖНУЮ ИСТОРИЮ…", "Текущая история не совпадает с выбранными настройками")
                        data = self._download_all(cfg)
                        self._post("data_set", data, self._config_key(cfg))
                    if self.stop_event.is_set():
                        raise StopRequested()
                    preflight, viable = [], []
                    for tf in cfg["tfs"]:
                        if tf not in data:
                            preflight.append(f"{tf}м: данные отсутствуют"); continue
                        try:
                            plan = build_walk_forward_plan(data[tf])
                            preflight.append(f"{tf}м: {len(data[tf]):,} свечей, {plan['total_days']:.1f} дней, WF {len(plan['windows'])}: "
                                             f"SELECTION {len(plan['selection_windows'])} + VALIDATION {len(plan['validation_windows'])}; DEV HOLDOUT {plan['holdout_days']} дн.")
                            if len(plan["windows"]) >= 4:
                                viable.append(tf)
                        except Exception as exc:
                            preflight.append(f"{tf}м: ошибка плана — {exc}")
                    self._post("status", f"ПРЕДПРОВЕРКА · режим {cfg['depth']}", " | ".join(preflight))
                    if not viable:
                        raise RuntimeError("Ни один выбранный таймфрейм не имеет 4 walk-forward окон. Предпроверка: " + " | ".join(preflight))
                    store = ResearchStore.create(RESEARCH_DIR, cfg, {tf: df_to_market(data[tf]) for tf in cfg["tfs"] if tf in data})
                    self._post("status", "ДИСКОВОЕ ХРАНИЛИЩЕ ГОТОВО", f"{store.root.name} · checkpoint каждые ~{CHECKPOINT_USEFUL_EVERY:,} полезных проверок")
                max_candles = max(len(df) for df in data.values())
                mem_workers, why = recommended_workers(max_candles, self.cpu_count)
                workers = max(1, min(int(cfg["workers"]), mem_workers))
                self._post("status", f"ПРОЦЕССОВ: {workers}", f"Число процессов: {workers} ({why})")
                runner = ResearchRunner(cfg, store, workers=workers, on_status=self._on_runner_status, stop_event=self.stop_event)
                res = runner.run()
                if res.status != "complete":
                    self._post("summary", store.summary())
                    self._post("stopped", f"{res.message} · на диске {store.committed_useful:,} проверок", res.status)
                    return
                seconds = max(0.0, time.monotonic() - self.search_started_at)
                store.finalize({"search_seconds": seconds, "retained_report_rows": len(res.report_pool), "top_rows": len(res.top),
                                "search_stats": res.search_stats})
                summary = store.summary()
                paths = save_results(res.report_pool, res.top, cfg["symbol"], research_summary=summary)
                self._post("status", "ПОИСК ГОТОВ · СОБИРАЮ ОТЧЁТ ДЛЯ CHATGPT…", "Полные сырые результаты остаются в research/")
                report_path = create_chatgpt_report(cfg, res.search_stats, res.report_pool, res.top, data, seconds,
                                                    research_summary=summary, result_files=list(paths))
                self._post("search_done", res, paths, str(report_path), summary, res.search_stats, seconds, data)
            except StopRequested:
                self._post("stopped", "Остановлено до начала расчёта")
            except Exception as exc:
                details = f"{exc}\n\n{traceback.format_exc(limit=8)}"
                file_log(f"SEARCH WORKER error: {details}")
                if store is not None:
                    self._post("summary", store.summary())
                self._post("error", details + ("\n\nВсё, что успело записаться на диск, сохранено; повторный запуск предложит продолжить." if store else ""))
            finally:
                self.active_task = ""
                gc.collect()

        self.worker = threading.Thread(target=run, daemon=True, name="ResearchWorker")
        self.worker.start()

    def _after_search(self, paths):
        self._set_busy(False)
        self.progress["value"] = 100
        all_path, top_path = paths[:2]
        self.big_status_var.set("ГОТОВО · ТОП-100 ПО SELECTION → VALIDATION СФОРМИРОВАН")
        total_useful = int((self.last_research_summary or {}).get("committed_useful_checks",0) or 0)
        if total_useful <= 0:
            total_useful = sum(int(x.get("useful_backtest_evaluations", 0) or 0) for x in self.search_stats)
        total_rows = int((self.last_research_summary or {}).get("committed_result_rows",0) or len(self.all_results))
        self.counter_var.set(f"На диске: {total_rows:,} стратегий · полезных проверок: {total_useful:,}")
        self._log(f"Рабочая выборка: {all_path.name}")
        self._log(f"ТОП-100: {top_path.name}")
        if self.last_research_summary:
            self._log(f"Полное хранилище: {self.last_research_summary.get('session_dir','research/')} · chunks {self.last_research_summary.get('chunk_count',0)}")
        if self.last_report_path:
            self._log(f"Полный отчёт для ChatGPT: {self.last_report_path.name}")
        total_candles = sum(len(df) for df in self.data_by_tf.values())
        self.data_var.set(f"Данные: {total_candles:,} свечей · SELECTION расширяет поиск · VALIDATION только проверяет · Dev holdout вне рейтинга")

        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for rank, row in enumerate(self.top_results, start=1):
            t = row.test
            wf = f"{row.wf_positive_windows}/{row.wf_windows}"
            if row.strategy_type != "BASE":
                mode_text = "ADAPT.V2" if row.strategy_type == "ADAPTIVE_V2" else "АДАПТ."
                atr_text = f"L{row.long_atr_period}/S{row.short_atr_period}"
                mult_text = f"{row.long_multiplier:g}/{row.short_multiplier:g}"
            else:
                mode_text = row.mode
                atr_text = str(row.atr_period)
                mult_text = f"{row.multiplier:g}"
            self.tree.insert("", tk.END, iid=str(rank - 1), values=(
                rank, f"{row.interval}м", mode_text, atr_text, mult_text,
                f"{row.selection_score:.3f}", f"{row.validation_score:.3f}", f"{row.robust_score:.3f}",
                wf, f"{row.wf_median_return_pct:+.2f}",
                f"{row.train.total_return_pct:+.2f}", f"{t.total_return_pct:+.2f}",
                f"{t.profit_factor:.2f}", f"{t.max_drawdown_pct:.2f}", t.trades,
            ))
        if self.top_results:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.tree.see("0")
            self.on_tree_select()

    def on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self.top_results):
            return
        row = self.top_results[idx]
        self.selected_row = row
        research, holdout = row.train, row.test
        regimes = (
            f"РОСТ: {row.up_windows} окон, прибыльных {row.up_positive_ratio*100:.0f}%, ср. {row.up_avg_return_pct:+.2f}% · "
            f"ПАДЕНИЕ: {row.down_windows} окон, прибыльных {row.down_positive_ratio*100:.0f}%, ср. {row.down_avg_return_pct:+.2f}% · "
            f"БОКОВИК: {row.flat_windows} окон, прибыльных {row.flat_positive_ratio*100:.0f}%, ср. {row.flat_avg_return_pct:+.2f}%"
        )
        if row.strategy_type != "BASE":
            if row.strategy_type == "ADAPTIVE_V2":
                regime_desc = (
                    f"EMA {row.ema_fast}/{row.ema_slow} · ADX{row.adx_period}>={row.adx_threshold:g} · "
                    f"sep={row.regime_separation_atr:g}ATR · confirm={row.regime_confirm_bars} · min-hold={row.regime_min_hold_bars}"
                )
            else:
                regime_desc = f"{row.regime_hours}ч / ±{row.regime_threshold_pct:g}%"
            strategy_line = (
                f"TF {row.interval} минут · {row.strategy_type} · "
                f"РОСТ: ATR={row.long_atr_period}, ×{row.long_multiplier:g} · "
                f"ПАДЕНИЕ: ATR={row.short_atr_period}, ×{row.short_multiplier:g} · {regime_desc} · "
                f"research score={row.robust_score:.3f}"
            )
            online_line = (
                f"Past-only классификатор: РОСТ {row.online_up_ratio*100:.1f}% · "
                f"ПАДЕНИЕ {row.online_down_ratio*100:.1f}% · БОКОВИК/ПАУЗА {row.online_flat_ratio*100:.1f}% · "
                f"переключений {row.regime_switches}\n"
            )
        else:
            strategy_line = f"TF {row.interval} минут · {row.mode} · ATR={row.atr_period} · множитель={row.multiplier:g} · research score={row.robust_score:.3f}"
            online_line = ""
        text = (
            strategy_line + "\n\n"
            f"SELECTION: score {row.selection_score:.3f} · окон {row.selection_windows} · + {row.selection_positive_ratio*100:.1f}% · медиана {row.selection_median_return_pct:+.2f}% · худшее {row.selection_worst_return_pct:+.2f}%\n"
            f"VALIDATION: score {row.validation_score:.3f} · окон {row.validation_windows} · + {row.validation_positive_ratio*100:.1f}% · медиана {row.validation_median_return_pct:+.2f}% · худшее {row.validation_worst_return_pct:+.2f}%\n"
            f"ВСЕ RESEARCH WF: прибыльных окон {row.wf_positive_windows}/{row.wf_windows} ({row.wf_positive_ratio*100:.1f}%) · медиана {row.wf_median_return_pct:+.2f}%\n"
            f"Research aggregate: сделок {research.trades} · результат {research.total_return_pct:+.2f}% · "
            f"PF {research.profit_factor:.2f} · просадка {research.max_drawdown_pct:.2f}%\n"
            f"{regimes}\n"
            f"{online_line}\n"
            f"DEVELOPMENT HOLDOUT — НЕ УЧАСТВУЕТ В РЕЙТИНГЕ, НО УЖЕ НЕ ЯВЛЯЕТСЯ НОВЫМ ЭКЗАМЕНОМ:\n"
            f"сделок {holdout.trades} · результат {holdout.total_return_pct:+.2f}% · PF {holdout.profit_factor:.2f} · "
            f"просадка {holdout.max_drawdown_pct:.2f}% · Win rate {holdout.win_rate:.1f}%\n\n"
            "Комиссия и проскальзывание учтены. Funding пока не учитывается. Новый честный forward-test потребует будущих свечей. Сигнал формируется после закрытия свечи, исполнение — на следующем open."
        )
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state=tk.DISABLED)

    def _selected_calc(self):
        if self.selected_row is None:
            raise ValueError("Сначала выполните автопоиск и выберите строку")
        row = self.selected_row
        if row.interval not in self.data_by_tf:
            raise ValueError("Нет данных выбранного таймфрейма")
        df = self.data_by_tf[row.interval]
        desired, label = build_selected_signal(df, row)
        plan = build_walk_forward_plan(df)
        return df, row, desired, label, plan

    def start_report_export(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Отчёт", "Сначала дождитесь завершения текущей операции или остановите её.")
            return
        cfg = self.last_config.copy() if self.last_config else {}
        if not cfg:
            try:
                cfg = self._read_config()
            except Exception:
                cfg = {"symbol": self.symbol_var.get().upper().strip() or "UNKNOWN", "tfs": self._selected_tfs()}
        self._set_busy(True)
        self.big_status_var.set("СОБИРАЮ ПОЛНЫЙ ОТЧЁТ ДЛЯ CHATGPT…")
        self.stage_var.set("Этап: отчёт")
        self.checked_var.set("Упаковка файлов")
        self.counter_var.set("Упаковываю код, логи, результаты и данные…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

        def run():
            try:
                path = create_chatgpt_report(
                    cfg,
                    self.search_stats,
                    self.all_results,
                    self.top_results,
                    self.data_by_tf,
                    self.last_search_seconds,
                    research_summary=self.last_research_summary or None,
                    result_files=self.last_result_files,
                )
                self._post("report_done", str(path))
            except Exception as exc:
                self._post("error", f"Ошибка создания отчёта: {exc}\n\n{traceback.format_exc(limit=8)}")

        self.worker = threading.Thread(target=run, daemon=True, name="ReportExportWorker")
        self.worker.start()

    def show_heatmap(self):
        if not self.all_results or self.selected_row is None:
            messagebox.showinfo("Тепловая карта", "Сначала выполните автопоиск и выберите результат.")
            return
        row = self.selected_row
        if row.strategy_type == "ADAPTIVE_V2":
            subset = [r for r in self.all_results if r.strategy_type == "ADAPTIVE_V2" and r.interval == row.interval
                      and r.long_atr_period == row.long_atr_period and abs(r.long_multiplier-row.long_multiplier) < 1e-9
                      and r.short_atr_period == row.short_atr_period and abs(r.short_multiplier-row.short_multiplier) < 1e-9
                      and r.regime_separation_atr == row.regime_separation_atr
                      and r.regime_confirm_bars == row.regime_confirm_bars and r.regime_min_hold_bars == row.regime_min_hold_bars]
            pairs = sorted({(r.ema_fast,r.ema_slow) for r in subset})
            thrs = sorted({round(r.adx_threshold,3) for r in subset})
            if not pairs or not thrs:
                messagebox.showinfo("Тепловая карта", "Недостаточно соседних Adaptive V2 вариантов для карты."); return
            matrix=np.full((len(pairs),len(thrs)),np.nan); pi={v:i for i,v in enumerate(pairs)}; ti={v:i for i,v in enumerate(thrs)}
            for r in subset: matrix[pi[(r.ema_fast,r.ema_slow)],ti[round(r.adx_threshold,3)]]=r.robust_score
            fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); im=ax.imshow(matrix,aspect="auto",origin="lower")
            ax.set_title(f"{row.interval}м · Adaptive V2 · research score"); ax.set_xlabel("ADX threshold"); ax.set_ylabel("EMA fast/slow")
            ax.set_xticks(range(len(thrs))); ax.set_xticklabels([f"{x:g}" for x in thrs])
            ax.set_yticks(range(len(pairs))); ax.set_yticklabels([f"{a}/{b}" for a,b in pairs])
            fig.colorbar(im,ax=ax,label="Research score"); fig.tight_layout(); PlotWindow(self,"Тепловая карта",fig); return
        if row.strategy_type == "ADAPTIVE":
            subset = [r for r in self.all_results if r.strategy_type == "ADAPTIVE" and r.interval == row.interval
                      and r.long_atr_period == row.long_atr_period and abs(r.long_multiplier-row.long_multiplier) < 1e-9
                      and r.short_atr_period == row.short_atr_period and abs(r.short_multiplier-row.short_multiplier) < 1e-9]
            hours = sorted({r.regime_hours for r in subset}); thrs = sorted({round(r.regime_threshold_pct, 4) for r in subset})
            if not hours or not thrs:
                messagebox.showinfo("Тепловая карта", "Недостаточно соседних адаптивных вариантов для карты."); return
            matrix=np.full((len(hours),len(thrs)),np.nan); hi={v:i for i,v in enumerate(hours)}; ti={v:i for i,v in enumerate(thrs)}
            for r in subset: matrix[hi[r.regime_hours],ti[round(r.regime_threshold_pct,4)]]=r.robust_score
            fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); im=ax.imshow(matrix,aspect="auto",origin="lower")
            ax.set_title(f"{row.interval}м · adaptive v1"); ax.set_xlabel("Порог, %"); ax.set_ylabel("Часы")
            ax.set_xticks(range(len(thrs))); ax.set_xticklabels([f"{x:g}" for x in thrs],rotation=45,ha="right"); ax.set_yticks(range(len(hours))); ax.set_yticklabels([str(x) for x in hours])
            fig.colorbar(im,ax=ax,label="Research score"); fig.tight_layout(); PlotWindow(self,"Тепловая карта",fig); return
        subset = [r for r in self.all_results if r.strategy_type == "BASE" and r.interval == row.interval and r.mode == row.mode]
        periods = sorted({r.atr_period for r in subset}); mults = sorted({round(r.multiplier, 4) for r in subset})
        matrix = np.full((len(periods), len(mults)), np.nan); pi={p:i for i,p in enumerate(periods)}; mi={m:i for i,m in enumerate(mults)}
        for r in subset: matrix[pi[r.atr_period], mi[round(r.multiplier,4)]] = r.robust_score
        fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); im=ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_title(f"{row.interval}м · {row.mode} · устойчивость walk-forward"); ax.set_xlabel("Множитель Supertrend"); ax.set_ylabel("ATR период")
        xs=max(1,len(mults)//12); ys=max(1,len(periods)//15); ax.set_xticks(range(0,len(mults),xs)); ax.set_xticklabels([f"{mults[i]:g}" for i in range(0,len(mults),xs)], rotation=45,ha="right")
        ax.set_yticks(range(0,len(periods),ys)); ax.set_yticklabels([str(periods[i]) for i in range(0,len(periods),ys)])
        fig.colorbar(im,ax=ax,label="Robust score"); fig.tight_layout(); PlotWindow(self,"Тепловая карта",fig)

    def show_equity(self):
        try:
            df, row, desired, label, plan = self._selected_calc()
        except Exception as exc:
            messagebox.showinfo("Кривая капитала", str(exc)); return
        fee=self.last_config["fee"]; slip=self.last_config["slippage"]; research_start=int(plan["research_start"]); holdout_start=int(plan["holdout_start"])
        _,_,eq1=backtest_desired_with_trades(df, desired, research_start, holdout_start-1, fee, slip)
        _,_,eq2=backtest_desired_with_trades(df, desired, holdout_start, len(df)-1, fee, slip)
        fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); ax.plot(np.arange(len(eq1)),eq1,label="Walk-forward research")
        off=len(eq1)-1; ax.plot(off+np.arange(len(eq2)),eq2*eq1[-1],label="Development holdout"); ax.axvline(off,linestyle="--",linewidth=1)
        ax.set_title(f"Кривая капитала · {row.interval}м · {label}"); ax.set_xlabel("Закрытые сделки"); ax.set_ylabel("Капитал, старт = 1.0"); ax.grid(True,alpha=0.25); ax.legend(); fig.tight_layout(); PlotWindow(self,"Кривая капитала",fig)

    def show_trades(self):
        try:
            df, row, desired, label, plan = self._selected_calc()
        except Exception as exc:
            messagebox.showinfo("Сделки", str(exc)); return
        fee=self.last_config["fee"]; slip=self.last_config["slippage"]; research_start=int(plan["research_start"]); holdout_start=int(plan["holdout_start"])
        _,t1,_=backtest_desired_with_trades(df, desired, research_start, holdout_start-1, fee, slip)
        _,t2,_=backtest_desired_with_trades(df, desired, holdout_start, len(df)-1, fee, slip)
        trades=[("RESEARCH",t) for t in t1]+[("DEV HOLDOUT",t) for t in t2]
        if not trades:
            messagebox.showinfo("Сделки","Сделок нет"); return
        win=tk.Toplevel(self); win.title(f"Сделки · {row.interval}м · {label}"); win.geometry("1200x650")
        cols=("part","side","entry_t","exit_t","entry","exit","gross","net"); tree=ttk.Treeview(win,columns=cols,show="headings")
        labels={"part":"Участок","side":"Сторона","entry_t":"Вход UTC","exit_t":"Выход UTC","entry":"Цена входа","exit":"Цена выхода","gross":"До расходов %","net":"Чистыми %"}
        for c in cols: tree.heading(c,text=labels[c]); tree.column(c,width=140,anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        for part,t in trades:
            tree.insert("",tk.END,values=(part,t.side,t.entry_time.strftime("%Y-%m-%d %H:%M"),t.exit_time.strftime("%Y-%m-%d %H:%M"),f"{t.entry_price:.2f}",f"{t.exit_price:.2f}",f"{t.gross_return_pct:+.4f}",f"{t.net_return_pct:+.4f}"))



def main():
    app = App()
    app.mainloop()
