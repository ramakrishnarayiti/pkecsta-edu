# PKecsta

Windows desktop pharmacokinetic (PK) analysis tool. Non-compartmental
analysis (NCA) and compartmental model fitting over manually entered or
imported (CSV/Excel) plasma concentration data. PySide6 UI, NumPy/SciPy
numerics. No cloud, no telemetry, no audit trail.

Developed by Rayiti Ramakrishna, PharmD, PhD candidate at CSIR-CDRI. Built
with the assistance of Claude (Anthropic).

## Install

```
pip install -r requirements.txt
```

## Run

```
python -m pkpd.app
```

## Testing

Three layers, catching different things:

| Command | Time | Catches |
|---|---|---|
| `pytest -q` | ~2 s | Numerics. Run on every change. |
| `python -m tests.app_sweep` | ~35 s | Qt layer: crashes, freezes, dialogs, wiring. |
| `python -m tests.regen_golden` | — | Regenerates locked NCA values. Read the diff. |

`pytest` never constructs a `MainWindow`, so a green suite alone doesn't
confirm the app still starts.

## Scope

- Plasma NCA only; no urine or drug-effect model families.
- 1-compartment across IV bolus, IV infusion, and extravascular dosing
  (including an optional lag-time variant). 2-compartment covers IV bolus
  only.
- One profile at a time — no batch run across subjects yet.
- Research tool: no 21 CFR Part 11 compliance claimed.
- BQL/missing values are dropped, counted as `n_excluded`.

Not yet independently cross-checked against PKNCA or a published dataset.

See `PLANNING.md` for the full spec and debugging notes.
