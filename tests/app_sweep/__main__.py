"""python -m tests.app_sweep

Drives every control in the app against the sample datasets and reports what
crashed, hung, or stalled the UI thread. Not part of `pytest` — it takes
minutes and spawns subprocesses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.app_sweep.runner import run_sweep, summarize, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="app_sweep")
    parser.add_argument("--visible", action="store_true",
                        help="show a real window instead of running offscreen")
    parser.add_argument("--filter", default="",
                        help="only steps whose group or name contains this text")
    parser.add_argument("--timeout", type=float, default=None,
                        help="override the per-step hang timeout, in seconds")
    args = parser.parse_args()

    sweep = run_sweep(visible=args.visible, filter_text=args.filter,
                      timeout_override=args.timeout)
    path = write_report(sweep)
    summary, exit_code = summarize(sweep)
    print(summary)
    print(f"report: {path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
