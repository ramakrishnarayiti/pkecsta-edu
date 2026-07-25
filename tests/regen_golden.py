"""Rewrite the committed golden values.

    python -m tests.regen_golden

Only run this when a number is *meant* to change, and read the resulting
git diff before committing it — that diff is the review artifact for any
change to the numerics. Regenerating to silence a red test throws away the
only protection the golden matrix provides.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.golden_cases import CASES, run_case

GOLDEN_PATH = Path(__file__).parent / "golden" / "nca_golden.json"


def main() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    values = {name: run_case(name) for name in sorted(CASES)}
    # sort_keys + trailing newline keep the diff readable and stable
    GOLDEN_PATH.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(values)} cases to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
