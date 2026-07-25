"""Generic background-thread wrapper. One class for any long computation
(NCA batch, curve fit) — never run numerics on the Qt UI thread."""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal

# Tasks with a thread still running. A QThread object collected before its
# thread has actually stopped aborts the process (0xC0000409 on Windows, no
# traceback, no stderr — it just vanishes).
#
# Callers cannot prevent this on their own: the only completion signal they
# see is `finished`, which fires from inside the worker *before* the thread
# has wound down, so any caller that drops its reference in a finished-handler
# is already too late. That is exactly what the UI does, so the lifetime is
# owned here instead of being a rule callers are asked to follow.
_ACTIVE: set["BackgroundTask"] = set()


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

    The task keeps itself alive until its thread has genuinely stopped, so
    callers are free to drop their reference as soon as `finished` fires.
    """

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        self._thread = QThread()
        self._runner = _Runner(fn, args, kwargs)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.finished.connect(self._thread.quit)
        self._runner.error.connect(self._thread.quit)
        # QThread.finished is emitted on the owning (UI) thread once the
        # worker thread's event loop has exited — the only point at which
        # this object is safe to release.
        self._thread.finished.connect(self._release)

        self.finished = self._runner.finished
        self.error = self._runner.error

    def start(self) -> None:
        _ACTIVE.add(self)
        self._thread.start()

    def _release(self) -> None:
        self._thread.wait()          # returns at once; the thread is done
        self._runner.deleteLater()
        # Dropping the last reference inside the emission that is still
        # running would destroy the QThread mid-signal, so defer one turn.
        QTimer.singleShot(0, lambda: _ACTIVE.discard(self))

    def wait(self, timeout_ms: int = -1) -> bool:
        return self._thread.wait(timeout_ms)
