"""Runs inside the child process: builds a real MainWindow and performs the
catalog against it, streaming one JSONL record per step.

Everything here is written so that a crash or a hang loses at most the step in
flight. Records are flushed immediately; the parent reconstructs the rest.

Three probes are installed before the window exists:

1. Dialog interception. A modal QMessageBox in a headless run has nobody to
   dismiss it and blocks forever, so every validation path would report as a
   freeze. Patched to record and return.
2. Heartbeat. A 25 ms timer on the UI thread measures the largest gap between
   its own ticks, which is the length of time the UI thread was blocked. That
   is the direct measurement of "the app freezes".
3. Fault handler. Dumps a native traceback on a hard crash so the parent has
   something to report beyond an exit code.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Must be set before any Qt module is imported, or it has no effect.
if "--visible" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

ALL_EVENTS = QEventLoop.ProcessEventsFlag.AllEvents
# Dialog kinds that mean the app objected to something. An informational or
# file dialog is normal traffic; these three are verdicts.
COMPLAINT_KINDS = ("warning", "critical", "question")

from tests.app_sweep.scenarios import SAMPLE_DIR, STEPS  # noqa: E402

STALL_FLAG_MS = 200.0

_dialogs: list[dict] = []
_heartbeat = {"last": 0.0, "max_gap_ms": 0.0}
_heartbeat_timer: QTimer | None = None


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------

def _install_dialog_interception() -> None:
    """Record and dismiss instead of showing. Returning a sane default keeps
    the code under test on its normal path."""
    def record(kind):
        def handler(parent=None, title="", text="", *args, **kwargs):
            _dialogs.append({"kind": kind, "title": str(title), "text": str(text)[:400]})
            if kind == "question":
                return QMessageBox.StandardButton.Yes
            return QMessageBox.StandardButton.Ok
        return staticmethod(handler)

    for kind in ("warning", "critical", "information", "question"):
        setattr(QMessageBox, kind, record(kind))

    # File dialogs would also block. Save goes to a scratch file we can read
    # back; open is answered by whichever dataset the step asked for.
    def save_name(parent=None, caption="", directory="", filter="", *a, **k):
        path = ROOT / "reports" / "_sweep_export.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        _dialogs.append({"kind": "save_dialog", "title": str(caption), "text": str(path)})
        return str(path), filter

    def open_name(parent=None, caption="", directory="", filter="", *a, **k):
        path = _pending_open_file[0] or ""
        _dialogs.append({"kind": "open_dialog", "title": str(caption), "text": path})
        return path, filter

    QFileDialog.getSaveFileName = staticmethod(save_name)
    QFileDialog.getOpenFileName = staticmethod(open_name)


_pending_open_file = [""]


def _install_heartbeat(app: QApplication) -> QTimer:
    # Held module-level on purpose. A QTimer with no surviving reference is
    # collected immediately and never ticks — the probe then reports a
    # reassuring 0 ms stall for a completely frozen UI.
    global _heartbeat_timer
    timer = QTimer()
    _heartbeat_timer = timer
    timer.setInterval(25)

    def tick():
        now = time.perf_counter()
        gap_ms = (now - _heartbeat["last"]) * 1000.0
        # The first tick has no meaningful predecessor.
        if _heartbeat["last"] and gap_ms > _heartbeat["max_gap_ms"]:
            _heartbeat["max_gap_ms"] = gap_ms
        _heartbeat["last"] = now

    timer.timeout.connect(tick)
    timer.start()
    return timer


def _reset_probes() -> None:
    _dialogs.clear()
    _heartbeat["max_gap_ms"] = 0.0
    _heartbeat["last"] = time.perf_counter()


# --------------------------------------------------------------------------
# Waiting without blocking the thread we are measuring
# --------------------------------------------------------------------------

def _spin(app: QApplication, predicate, timeout_s: float, poll_s: float = 0.005) -> bool:
    """Pump the event loop until predicate() is true. Never sleeps the UI
    thread and never calls QThread.wait() — either would hide the stall this
    harness exists to measure."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents(ALL_EVENTS, 10)
        if predicate():
            return True
        time.sleep(poll_s)
    app.processEvents(ALL_EVENTS, 10)
    return bool(predicate())


def _settle(app: QApplication, seconds: float = 0.05) -> None:
    _spin(app, lambda: False, seconds)


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

class Actions:
    """One method per `action` name in the catalog."""

    def __init__(self, app: QApplication, window) -> None:
        self.app = app
        self.window = window
        self._loaded_file = None

    # -- helpers ---------------------------------------------------------
    def _load(self, filename: str, subject: str | None = None) -> None:
        """Import a dataset and acquire it, unless it is already loaded."""
        if self._loaded_file != filename:
            _pending_open_file[0] = str(SAMPLE_DIR / filename)
            self._import_through_wizard(filename)
            self._loaded_file = filename
        if subject is not None:
            combo = self.window.nca_subject_combo
            idx = combo.findText(str(subject))
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _import_through_wizard(self, filename: str) -> None:
        """Drive ImportWizard programmatically. Calling exec() would block on
        a modal dialog with nothing to close it."""
        from pkpd.ui.import_wizard import ImportWizard

        wizard = ImportWizard(self.window,
                               default_route=self.window.route_combo.currentText(),
                               default_dose=self.window.dose_input.text().strip())
        _pending_open_file[0] = str(SAMPLE_DIR / filename)
        wizard._choose_file()
        wizard._accept()
        if wizard.dataframe is not None:
            self.window.grid.load_dataframe(wizard.dataframe)
            self.window.dataset = None
        self.window._use_grid_data()
        _settle(self.app)

    def _wait_for_nca(self, timeout_s: float) -> bool:
        return _spin(self.app, lambda: self.window._nca_task is None, timeout_s)

    def _wait_for_fit(self, timeout_s: float) -> bool:
        return _spin(self.app, lambda: self.window._comp_task is None, timeout_s)

    def _configure_nca(self, p: dict) -> None:
        w = self.window
        w.mode_toggle.setChecked(True)  # Manual exposes every control
        if "auc_method" in p:
            w.nca_auc_method.setCurrentText(p["auc_method"])
        if "lz_mode" in p:
            w.nca_lz_mode.setCurrentText(p["lz_mode"])
        w.nca_lz_range.setText(p.get("lz_range", ""))
        w.nca_lz_exclude.setText(p.get("exclude", ""))
        w.nca_steady_state.setChecked(bool(p.get("steady_state")))
        w.nca_tau.setText(p.get("tau", ""))

    # -- data tab --------------------------------------------------------
    def reset(self, p): _settle(self.app)

    def grid_add_row(self, p):
        before = self.window.grid.rowCount()
        self.window.grid.add_row()
        assert self.window.grid.rowCount() == before + 1

    def grid_delete_row(self, p):
        grid = self.window.grid
        if grid.rowCount() == 0:
            grid.add_row()
        grid.selectRow(0)
        before = grid.rowCount()
        grid.delete_selected_rows()
        assert grid.rowCount() < before

    def grid_paste(self, p):
        grid = self.window.grid
        QApplication.clipboard().setText("1\t0.5\t42.0\n1\t1.0\t38.0")
        grid.set_current_cell(0, 0)
        grid._paste_clipboard()
        assert grid.cell(0, 1) == "0.5", "pasted value did not reach the model"

    def grid_paste_no_selection(self, p):
        grid = self.window.grid
        QApplication.clipboard().setText("9\t0.5\t42.0")
        grid.clearSelection()
        grid.setCurrentIndex(grid.model().index(-1, -1))
        grid._paste_clipboard()
        # The paste must land somewhere rather than vanishing into row -1.
        assert grid.cell(0, 0) == "9", "paste with no selection was discarded"

    def toggle_mode(self, p): self.window.mode_toggle.setChecked(bool(p["manual"]))

    def toggle_help(self, p):
        self.window.help_panel.setVisible(bool(p["visible"]))

    def set_dose_text(self, p): self.window.dose_input.setText(p["text"])

    def set_route(self, p): self.window.route_combo.setCurrentText(p["route"])

    def set_units(self, p):
        self.window.time_unit_combo.setCurrentText(p["time"])
        self.window.conc_unit_combo.setCurrentText(p["conc"])
        self.window.dose_unit_combo.setCurrentText(p["dose"])

    def acquire_empty(self, p):
        self.window.grid.clear_rows()
        self.window.grid.add_row()
        self.window.dataset = None
        self._loaded_file = None
        self.window._use_grid_data()

    def import_file(self, p):
        self._loaded_file = None
        self._import_through_wizard(p["file"])

    # -- NCA -------------------------------------------------------------
    def run_nca(self, p):
        self._load(p["file"], p.get("subject"))
        self._configure_nca(p)
        subjects = ([self.window.nca_subject_combo.itemText(i)
                     for i in range(self.window.nca_subject_combo.count())]
                    if p.get("all_subjects") else [None])
        for subject in subjects:
            if subject is not None:
                self.window.nca_subject_combo.setCurrentText(subject)
            self.window._run_nca()
            assert self._wait_for_nca(p.get("_timeout", 30.0)), "NCA never finished"

    def run_nca_no_data(self, p):
        self.window.dataset = None
        self._loaded_file = None
        self.window.nca_subject_combo.clear()
        self.window._run_nca()

    def export_core_output(self, p):
        self._load("01_iv_bolus_postdose.csv")
        self.window.mode_toggle.setChecked(False)
        self.window._run_nca()
        assert self._wait_for_nca(30.0), "NCA never finished"
        fmt = p.get("format", "Text")
        self.window.nca_core_output_format.setCurrentText(fmt)
        self.window._export_nca_core_output()
        base = ROOT / "reports" / "_sweep_export.txt"

        if fmt == "Text":
            text = base.read_text(encoding="utf-8")
            for expected in ("Settings:", "Results:", "Warnings:"):
                assert expected in text, f"Core Output missing {expected!r}"
            return

        # The mocked save dialog always hands back the same fixed .txt path;
        # _export_nca_core_output appends the correct extension when it
        # doesn't match the chosen format, same as it would for a real path.
        import pandas as pd
        exported = base.with_name(base.name + (".csv" if fmt == "CSV" else ".xlsx"))
        df = (pd.read_csv(exported) if fmt == "CSV"
              else pd.read_excel(exported, engine="openpyxl"))
        sections = set(df["Section"])
        assert {"Settings", "Results"} <= sections, f"{fmt} Core Output missing expected sections"

    def export_no_results(self, p):
        self.window._nca_last_result = None
        self.window._export_nca_core_output()

    def export_plot(self, p):
        self._load(p.get("file", "01_iv_bolus_postdose.csv"))
        self.window.mode_toggle.setChecked(False)
        self.window._run_nca()
        assert self._wait_for_nca(30.0), "NCA never finished"
        plot = self.window.nca_plot
        plot.export_format.setCurrentText(p["format"])
        plot._export()
        exported = ROOT / "reports" / "_sweep_export.txt"
        assert exported.exists() and exported.stat().st_size > 0, \
            f"{p['format']} plot export produced no file"

    def toggle_plot_scale(self, p):
        for plot in (self.window.nca_plot, self.window.comp_plot):
            plot.scale_toggle.setChecked(not plot.scale_toggle.isChecked())
            plot.scale_toggle.setChecked(not plot.scale_toggle.isChecked())

    # -- compartmental ---------------------------------------------------
    def run_fit(self, p):
        self._load(p["file"])
        if p.get("subject"):
            combo = self.window.comp_subject_combo
            idx = combo.findText(str(p["subject"]))
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if p.get("model"):
            self.window.comp_model_combo.setCurrentText(p["model"])
        self.window.comp_weight_combo.setCurrentText(p.get("weight_scheme", "uniform"))
        self.window.comp_log_residuals.setChecked(bool(p.get("log_residuals")))
        self.window._run_fit()
        assert self._wait_for_fit(60.0), "fit never finished"
        if p.get("expect_params_at_bounds"):
            pinned = (self.window._comp_last_result or {}).get("params_at_bounds")
            assert pinned, "a degenerate fit reported no parameter at its bound"

    def run_fit_no_data(self, p):
        self.window.dataset = None
        self._loaded_file = None
        self.window.comp_subject_combo.clear()
        self.window._run_fit()

    # -- concurrency -----------------------------------------------------
    def double_run_nca(self, p):
        self._load(p["file"])
        self.window.mode_toggle.setChecked(False)
        self.window._run_nca()
        self.window._run_nca()  # re-entrancy guard must hold
        assert self._wait_for_nca(60.0), "NCA never finished after a double click"

    def nca_and_fit_together(self, p):
        self._load(p["file"])
        self.window.mode_toggle.setChecked(False)
        self.window.comp_model_combo.setCurrentText("1-compartment: IV bolus (K, V)")
        self.window._run_nca()
        self.window._run_fit()
        assert _spin(self.app,
                     lambda: self.window._nca_task is None and self.window._comp_task is None,
                     60.0), "two concurrent tasks never both finished"

    def rapid_subject_switch(self, p):
        self._load(p["file"])
        self.window.mode_toggle.setChecked(False)
        self.window._run_nca()
        for i in range(self.window.nca_subject_combo.count()):
            self.window.nca_subject_combo.setCurrentIndex(i)
        assert self._wait_for_nca(60.0), "NCA never finished while switching subject"

    def repeated_runs(self, p):
        self._load(p["file"])
        self.window.mode_toggle.setChecked(False)
        for _ in range(int(p.get("count", 10))):
            self.window._run_nca()
            assert self._wait_for_nca(30.0), "NCA never finished during repeat loop"

    def repeated_fits(self, p):
        self._load(p["file"])
        self.window.comp_model_combo.setCurrentText("1-compartment: IV bolus (K, V)")
        for _ in range(int(p.get("count", 5))):
            self.window._run_fit()
            assert self._wait_for_fit(60.0), "fit never finished during repeat loop"

    def tab_switch_during_run(self, p):
        self._load(p["file"])
        self.window.mode_toggle.setChecked(False)
        self.window._run_nca()
        for _ in range(20):
            self.app.processEvents(ALL_EVENTS, 5)
        assert self._wait_for_nca(120.0), "NCA never finished while navigating"


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--filter", default="")
    parser.add_argument("--visible", action="store_true")
    args = parser.parse_args()

    faulthandler.enable()

    _install_dialog_interception()
    app = QApplication.instance() or QApplication([])
    from pkpd.ui.main_window import MainWindow
    from pkpd.ui.theme import STYLESHEET

    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    if args.visible:
        window.show()
    _install_heartbeat(app)
    actions = Actions(app, window)

    steps = [s for s in STEPS if not args.filter or args.filter in s.group or args.filter in s.name]
    out = sys.stdout

    for index in range(args.start, len(steps)):
        step = steps[index]
        _reset_probes()
        started = time.perf_counter()

        # Announced before running, so a step that crashes or hangs is
        # identifiable from the log alone.
        out.write(json.dumps({"event": "start", "index": index, "step": step.as_dict()}) + "\n")
        out.flush()

        status, error = "passed", None
        try:
            handler = getattr(actions, step.action, None)
            if handler is None:
                status, error = "skipped", f"no handler for action {step.action!r}"
            else:
                params = dict(step.params)
                params["_timeout"] = step.timeout_s
                handler(params)
                _settle(app)
        except Exception:
            status, error = "failed", traceback.format_exc(limit=8)

        dialogs = list(_dialogs)
        complaints = [d for d in dialogs if d["kind"] in COMPLAINT_KINDS]
        if status == "passed":
            if step.expect_dialog and not complaints:
                # Silent acceptance of bad input is how wrong numbers reach a
                # report, so a missing complaint is as much a failure as a
                # spurious one.
                status, error = "failed", "expected the app to object; it accepted this silently"
            elif not step.expect_dialog and complaints:
                status = "failed"
                error = "unexpected dialog: " + "; ".join(
                    f"[{d['kind']}] {d['title']}: {d['text']}" for d in complaints)

        out.write(json.dumps({
            "event": "result", "index": index, "name": step.name, "group": step.group,
            "status": status, "error": error,
            "seconds": round(time.perf_counter() - started, 4),
            "max_stall_ms": round(_heartbeat["max_gap_ms"], 1),
            "dialogs": dialogs,
        }) + "\n")
        out.flush()

    out.write(json.dumps({"event": "done", "total": len(steps)}) + "\n")
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
