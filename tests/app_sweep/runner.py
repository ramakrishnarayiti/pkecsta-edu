"""Parent process: spawns the driver, survives its death, writes the report.

The driver runs the app for real, so it can abort hard (a Qt fault takes the
whole process with it) or wedge forever (a deadlock). Neither may cost us the
rest of the sweep, so the parent:

  - reads JSONL as it arrives and knows which step was in flight,
  - restarts the child at the next index when it dies, recording a CRASHED
    step with its stderr,
  - kills and restarts it when no record arrives inside the step's timeout,
    recording a HUNG step.

That resume-on-death loop is the whole reason the sweep is a subprocess
rather than a pytest file.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
STALL_FLAG_MS = 200.0
FREEZE_MS = 1000.0


def _read_lines(pipe, queue: list, lock: threading.Lock) -> None:
    """Drain the child's stdout on a thread so the parent can apply its own
    timeout instead of blocking forever in readline()."""
    for line in iter(pipe.readline, ""):
        line = line.strip()
        if not line:
            continue
        with lock:
            queue.append(line)
    pipe.close()


def run_sweep(visible: bool = False, filter_text: str = "",
              timeout_override: float | None = None) -> dict:
    from tests.app_sweep.scenarios import STEPS

    steps = [s for s in STEPS
             if not filter_text or filter_text in s.group or filter_text in s.name]
    total = len(steps)
    results: list[dict] = []
    start_index = 0
    started_at = time.perf_counter()
    restarts = 0

    while start_index < total:
        cmd = [sys.executable, "-u", "-m", "tests.app_sweep.driver", "--start", str(start_index)]
        if filter_text:
            cmd += ["--filter", filter_text]
        if visible:
            cmd.append("--visible")

        child = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        queue: list[str] = []
        lock = threading.Lock()
        reader = threading.Thread(target=_read_lines, args=(child.stdout, queue, lock), daemon=True)
        reader.start()

        in_flight: dict | None = None
        last_activity = time.perf_counter()
        finished_cleanly = False

        while True:
            with lock:
                pending, queue[:] = list(queue), []

            for line in pending:
                last_activity = time.perf_counter()
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # stray print from a library, not our protocol
                if record.get("event") == "start":
                    in_flight = record
                elif record.get("event") == "result":
                    results.append(record)
                    start_index = record["index"] + 1
                    in_flight = None
                elif record.get("event") == "done":
                    finished_cleanly = True

            if finished_cleanly:
                start_index = total
                break

            if child.poll() is not None:
                break  # child died; handled below

            budget = timeout_override or (in_flight["step"]["timeout_s"] if in_flight else 60.0)
            if time.perf_counter() - last_activity > budget:
                child.kill()
                results.append(_incident("hung", in_flight, start_index, steps,
                                          f"no output for {budget:.0f}s — killed"))
                start_index = (in_flight["index"] if in_flight else start_index) + 1
                restarts += 1
                break

            time.sleep(0.02)

        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
        stderr = (child.stderr.read() or "") if child.stderr else ""

        if finished_cleanly or start_index >= total:
            break

        if child.returncode not in (0, None) and in_flight is not None:
            results.append(_incident("crashed", in_flight, start_index, steps,
                                      f"exit code {child.returncode}\n{stderr[-2000:]}"))
            start_index = in_flight["index"] + 1
            restarts += 1
        elif child.returncode not in (0, None) and in_flight is None:
            # Died between steps — usually a startup failure. Without this
            # guard the loop would respawn forever at the same index.
            results.append(_incident("crashed", None, start_index, steps,
                                      f"exit code {child.returncode} between steps\n{stderr[-2000:]}"))
            start_index += 1
            restarts += 1

    return {
        "results": results,
        "total": total,
        "restarts": restarts,
        "seconds": round(time.perf_counter() - started_at, 1),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }


def _incident(status: str, in_flight: dict | None, index: int, steps, detail: str) -> dict:
    step = in_flight["step"] if in_flight else (
        steps[index].as_dict() if index < len(steps) else {"name": "?", "group": "?"})
    return {
        "event": "result", "index": in_flight["index"] if in_flight else index,
        "name": step.get("name", "?"), "group": step.get("group", "?"),
        "status": status, "error": detail, "seconds": None,
        "max_stall_ms": None, "dialogs": [],
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(sweep: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    results = sweep["results"]
    by = lambda s: [r for r in results if r["status"] == s]  # noqa: E731
    passed, failed, crashed, hung, skipped = (
        by("passed"), by("failed"), by("crashed"), by("hung"), by("skipped"))

    lines = [
        "# App sweep report",
        "",
        f"_{sweep['generated']} — {sweep['seconds']}s, {len(results)}/{sweep['total']} steps_",
        "",
        f"**{len(passed)} passed · {len(failed)} failed · {len(crashed)} crashed · "
        f"{len(hung)} hung · {len(skipped)} skipped**",
        "",
    ]
    if crashed or hung:
        lines += [f"Child process restarted {sweep['restarts']} time(s) to get past crashes/hangs.", ""]
    if not (failed or crashed or hung or skipped):
        lines += ["No crashes, no hangs, no unexpected dialogs.", ""]

    def table(title, rows, body):
        if not rows:
            return []
        out = [f"## {title}", ""]
        for r in rows:
            out.append(f"### `{r['group']}` — {r['name']}")
            out.append("")
            out += body(r)
            out.append("")
        return out

    lines += table("Crashes", crashed, lambda r: [
        "The process died on this step. Everything after it ran in a fresh child.",
        "", "```", (r["error"] or "").strip(), "```"])
    lines += table("Hangs", hung, lambda r: [
        "No output inside the step's timeout — the UI thread never came back.",
        "", "```", (r["error"] or "").strip(), "```"])
    lines += table("Failures", failed, lambda r: ["```", (r["error"] or "").strip(), "```"])
    lines += table("Skipped", skipped, lambda r: [(r["error"] or "").strip()])

    stalls = sorted((r for r in results if (r.get("max_stall_ms") or 0) >= STALL_FLAG_MS),
                    key=lambda r: -r["max_stall_ms"])
    if stalls:
        lines += ["## UI-thread stalls", "",
                  f"Longest gap between heartbeats while the step ran. Over {FREEZE_MS:.0f} ms "
                  "the window is visibly frozen.", "",
                  "| Stall (ms) | Frozen | Group | Step |", "|---:|:---:|---|---|"]
        for r in stalls[:25]:
            frozen = "yes" if r["max_stall_ms"] >= FREEZE_MS else ""
            lines.append(f"| {r['max_stall_ms']:.0f} | {frozen} | {r['group']} | {r['name']} |")
        lines.append("")

    slowest = sorted((r for r in results if r.get("seconds")), key=lambda r: -r["seconds"])[:10]
    if slowest:
        lines += ["## Slowest steps", "", "| Seconds | Group | Step |", "|---:|---|---|"]
        lines += [f"| {r['seconds']:.2f} | {r['group']} | {r['name']} |" for r in slowest]
        lines.append("")

    lines += ["## All steps", "", "| # | Status | Stall (ms) | Group | Step |",
              "|---:|---|---:|---|---|"]
    for r in results:
        stall = f"{r['max_stall_ms']:.0f}" if r.get("max_stall_ms") is not None else "—"
        lines.append(f"| {r['index']} | {r['status']} | {stall} | {r['group']} | {r['name']} |")
    lines.append("")

    report_path = REPORTS / "app_sweep.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "app_sweep.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    return report_path


def summarize(sweep: dict) -> tuple[str, int]:
    results = sweep["results"]
    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("passed", "failed", "crashed", "hung", "skipped")}
    bad = counts["failed"] + counts["crashed"] + counts["hung"]
    summary = (f"{counts['passed']} passed, {counts['failed']} failed, "
               f"{counts['crashed']} crashed, {counts['hung']} hung, "
               f"{counts['skipped']} skipped in {sweep['seconds']}s")
    return summary, (1 if bad else 0)
