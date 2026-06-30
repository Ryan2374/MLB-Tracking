#!/usr/bin/env python3
"""Render HTML evaluation report from compiled Fastball predictions JSON."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_html_report(payload: dict) -> str:
    comparison = payload.get("comparison_summary_trusted") or payload.get("comparison_summary") or []
    labels = [str(row["run_id"]) for row in comparison]
    medians = [float(row["median_error_px"]) if row.get("median_error_px") is not None else 0.0 for row in comparison]
    p90s = [float(row["p90_error_px"]) if row.get("p90_error_px") is not None else 0.0 for row in comparison]

    champions = payload.get("champions") or {}
    rows_html = []
    for row in comparison:
        rows_html.append(
            "<tr>"
            f"<td>{_esc(row.get('run_id'))}</td>"
            f"<td>{_esc(row.get('method'))}</td>"
            f"<td>{_esc(row.get('n_points'))}</td>"
            f"<td>{row.get('count', '')}</td>"
            f"<td>{row.get('median_error_px', '')}</td>"
            f"<td>{row.get('p90_error_px', '')}</td>"
            f"<td>{row.get('max_error_px', '')}</td>"
            "</tr>"
        )

    champ_lines = []
    for key, champ in champions.items():
        summary = champ.get("summary_trusted") or champ.get("summary") or {}
        champ_lines.append(
            f"<li><b>{_esc(key)}</b> ({_esc(champ.get('run_id'))}): "
            f"median={summary.get('median_error_px', 'n/a')} px</li>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{_esc(payload.get('title', 'MLB Eval Report'))}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #111; color: #eee; }}
    h1, h2 {{ color: #fff; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
    th {{ background: #222; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .card {{ background: #1a1a1a; padding: 16px; border-radius: 8px; }}
    @media (max-width: 900px) {{ .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>{_esc(payload.get('title', 'MLB Eval Report'))}</h1>
  <p>Exported: {_esc(payload.get('exported_at'))} | Pitches: {_esc(payload.get('pitch_count'))} |
     Schema: {_esc(payload.get('schema_version'))}</p>
  <h2>Champions</h2>
  <ul>{''.join(champ_lines)}</ul>
  <div class="charts">
    <div class="card"><canvas id="medianChart"></canvas></div>
    <div class="card"><canvas id="p90Chart"></canvas></div>
  </div>
  <h2>Model comparison (trusted)</h2>
  <table>
    <thead><tr><th>run_id</th><th>method</th><th>n_points</th><th>count</th>
    <th>median_error_px</th><th>p90_error_px</th><th>max_error_px</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
  <script>
    const labels = {json.dumps(labels)};
    const medians = {json.dumps(medians)};
    const p90s = {json.dumps(p90s)};
    const baseOpts = {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'pixels' }} }} }}
    }};
    new Chart(document.getElementById('medianChart'), {{
      type: 'bar',
      data: {{ labels, datasets: [{{ label: 'median_error_px', data: medians, backgroundColor: '#4e9af1' }}] }},
      options: {{ ...baseOpts, plugins: {{ title: {{ display: true, text: 'Median error (trusted)' }} }} }}
    }});
    new Chart(document.getElementById('p90Chart'), {{
      type: 'bar',
      data: {{ labels, datasets: [{{ label: 'p90_error_px', data: p90s, backgroundColor: '#f1a34e' }}] }},
      options: {{ ...baseOpts, plugins: {{ title: {{ display: true, text: 'P90 error (trusted)' }} }} }}
    }});
  </script>
</body>
</html>
"""


def write_html_report(payload: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html_report(payload), encoding="utf-8")
    return out_path
