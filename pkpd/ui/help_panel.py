"""Persistent right-side help panel: how to use the app, with a worked
example. Static HTML in a read-only QTextBrowser — no per-tab logic needed."""
from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser

HELP_HTML = """
<h2>How to use this software</h2>

<h3>1. Enter data — Data tab</h3>
<p>Either:</p>
<ul>
  <li><b>Import File...</b> — load a CSV/Excel file, then map its columns
      to the required fields.</li>
  <li>Or type/paste directly into the grid (Ctrl+V pastes from Excel),
      then click <b>Use Grid Data</b>.</li>
</ul>
<p>Required columns per row:</p>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>subject_id</th><th>time</th><th>concentration</th><th>dose</th><th>route</th></tr>
<tr><td>1</td><td>0</td><td>100</td><td>500</td><td>iv_bolus</td></tr>
<tr><td>1</td><td>1</td><td>86.1</td><td>500</td><td>iv_bolus</td></tr>
<tr><td>1</td><td>2</td><td>74.1</td><td>500</td><td>iv_bolus</td></tr>
<tr><td>1</td><td>4</td><td>54.9</td><td>500</td><td>iv_bolus</td></tr>
<tr><td>1</td><td>8</td><td>30.1</td><td>500</td><td>iv_bolus</td></tr>
<tr><td>1</td><td>12</td><td>16.5</td><td>500</td><td>iv_bolus</td></tr>
<tr><td>1</td><td>24</td><td>2.7</td><td>500</td><td>iv_bolus</td></tr>
</table>
<p><i>route</i> must be one of: <code>iv_bolus</code>, <code>iv_infusion</code>,
<code>extravascular</code>. For <code>iv_infusion</code>, also fill
<code>infusion_duration</code>.</p>
<p>Every row for the same subject repeats the same <code>dose</code> and
<code>route</code> — only <code>time</code>/<code>concentration</code> change
row to row.</p>

<h3>2. Non-compartmental analysis — NCA tab</h3>
<p>Pick the subject, pick an AUC method (<code>linear</code> is the safe
default), click <b>Run NCA</b>. Results table fills in on the left,
concentration-time plot on the right. The button above the plot switches
between <b>Semi-log</b> and <b>Linear</b> y-axis scale — semi-log is the
default and best for reading the terminal slope, linear is useful for
absorption-phase shape.</p>
<p><b>Example result</b> for the table above: Cmax=100, λz≈0.15/hr,
half-life≈4.6hr, CL≈0.73 L/hr, Vz≈4.9 L.</p>

<h3>3. Compartmental fitting — Compartmental tab</h3>
<p>Pick the subject, pick a model, pick a weighting scheme
(<code>uniform</code> is fine to start), click <b>Fit</b>. Initial guesses
are computed automatically from your data — no need to type starting
values. The button stays disabled and shows <b>Fitting...</b> (with a busy
cursor) until the fit finishes — this can take a couple of seconds for
harder models, that's normal, not a freeze.</p>
<p>Model choices include two parameterizations of the same 1-compartment
IV bolus model: <b>K, V</b> (rate constant) and <b>Cl, V</b> (clearance) —
same underlying curve, different parameters. Results include each
parameter's <b>CV%</b> (relative standard error) alongside its estimate, so
you can judge how precisely each parameterization was determined.</p>
<p><b>Example result</b> for the table above (1-compartment IV bolus, K/V):
k≈0.15/hr, V≈5 L, R²≈1.0 — the fitted orange curve should sit right on top
of the blue observed points. Same plot has the Semi-log/Linear toggle
described above.</p>

<h3>Tips</h3>
<ul>
  <li>Every subject needs its own <code>subject_id</code> value repeated
      across its rows.</li>
  <li>If a fit looks wrong, check <code>route</code> matches how the dose
      was actually given.</li>
  <li>R² near 1.0, small residuals, and low CV% (Compartmental results
      table) mean a good fit; low R² or high CV% means try a different
      model (e.g. 2-compartment) or the other K/V vs Cl/V parameterization.</li>
</ul>
"""


class HelpPanel(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setHtml(HELP_HTML)
        self.setMinimumWidth(320)
        self.setMaximumWidth(420)
