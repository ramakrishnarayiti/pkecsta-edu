"""Generic background-thread wrapper. One class for any long computation
(NCA batch, curve fit) — never run numerics on the Qt UI thread."""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class _Runner(QObject):
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, fn: Callable[..., Any], args: tuple, kwargs: dict):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.error.emit(exc)
            return
        self.finished.emit(result)


class BackgroundTask:
    """Runs `fn(*args, **kwargs)` on a QThread.

    Usage:
        task = BackgroundTask(some_slow_fn, arg1, arg2)
        task.finished.connect(on_done)
        task.error.connect(on_error)
        task.start()
    Keep a reference to `task` alive until it finishes (e.g. store on self).
    """

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        self._thread = QThread()
        self._runner = _Runner(fn, args, kwargs)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.finished.connect(self._thread.quit)
        self._runner.error.connect(self._thread.quit)

        self.finished = self._runner.finished
        self.error = self._runner.error

    def start(self) -> None:
        self._thread.start()

    def wait(self, timeout_ms: int = -1) -> bool:
        return self._thread.wait(timeout_ms)
