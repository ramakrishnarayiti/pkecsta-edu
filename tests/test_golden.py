"""Golden-master matrix: every dose route crossed with every AUC method,
single-dose and steady-state, locked to values committed in
`tests/golden/nca_golden.json`.

This is the test that catches the failure mode NCA clones actually die of:
one rule quietly changed for one route/method combination, every number
still plausible, nobody notices for a year. Unit tests elsewhere check that
each piece is right in isolation; this checks that no combination moved.

A failure here is not automatically a bug — it means a number changed, and
you now have to say why. If the change is intended, regenerate and *read the
diff*, which is the entire point of committing the values:

    python -m tests.regen_golden

Never regenerate to make a red test go green without reading that diff.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.golden_cases import CASES, LOCKED_PARAMS, run_case

GOLDEN_PATH = Path(__file__).parent / "golden" / "nca_golden.json"

# Tight enough that any real change in the numerics trips it, loose enough
# to survive a last-ulp difference between BLAS/libm builds.
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.fail(f"missing golden file {GOLDEN_PATH} — run: python -m tests.regen_golden")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_golden_case_matches_committed_values(golden: dict, case_name: str) -> None:
    if case_name not in golden:
        pytest.fail(f"case '{case_name}' has no committed values — run: python -m tests.regen_golden")

    expected = golden[case_name]
    actual = run_case(case_name)

    for param in LOCKED_PARAMS:
        want, got = expected.get(param), actual.get(param)
        if want is None or got is None:
            assert want == got, f"{case_name}.{param}: {want!r} -> {got!r}"
        else:
            assert got == pytest.approx(want, rel=TOLERANCE), f"{case_name}.{param}"


def test_golden_file_covers_every_case(golden: dict) -> None:
    """A case added to the matrix without regenerating would otherwise sit
    silently unlocked."""
    assert sorted(golden) == sorted(CASES)


def test_matrix_covers_every_route_and_method_combination() -> None:
    from pkpd.pk.nca import AUC_METHODS

    routes = {case["route"] for case in CASES.values()}
    methods = {case["auc_method"] for case in CASES.values()}
    assert routes == {"iv_bolus", "iv_infusion", "extravascular"}
    assert methods == set(AUC_METHODS)
    # every route x method x dosing-mode combination present, none missing
    assert len(CASES) == len(routes) * len(methods) * 2
