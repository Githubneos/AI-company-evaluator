"""CLI: static HTML monitoring report (spec 8.2).

    python -m scripts.monitoring_report

Writes artifacts/monitoring/report.html. Static because the alternative is
running Grafana, and everything here is a scheduled batch summary rather than a
live stream.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path

from evaluator.config import ARTIFACT_DIR
from evaluator.feedback.postmortem import tag_summary
from evaluator.model.predict import available_targets, load_model
from evaluator.monitoring import system_report

REPORT_DIR = Path(ARTIFACT_DIR) / "monitoring"

CSS = """
:root { --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --bad:#b3261e; --ok:#1e6b3a; --warn:#8a6d000; }
* { box-sizing:border-box; }
body { font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:var(--fg);
       margin:0; padding:32px; background:#fafafa; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:15px; margin:28px 0 10px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.sub { color:var(--muted); margin-bottom:24px; }
table { border-collapse:collapse; width:100%; background:#fff; border:1px solid var(--line); }
th,td { text-align:left; padding:7px 11px; border-bottom:1px solid var(--line); }
th { background:#f4f4f4; font-weight:600; }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.bad { color:var(--bad); font-weight:600; }
.ok { color:var(--ok); font-weight:600; }
.note { background:#fff8e6; border-left:3px solid #d4a017; padding:11px 14px; margin:18px 0; }
"""


def _cell(value, good_above: float = 0.0) -> str:
    if value is None:
        return '<td class="num">n/a</td>'
    css = "ok" if value > good_above else "bad"
    return f'<td class="num {css}">{value:+.4f}</td>'


def build_html() -> str:
    report = system_report()
    report["post_mortem_tags"] = tag_summary()

    rows = []
    for name in available_targets():
        try:
            meta = load_model(name).metadata
        except Exception:  # noqa: BLE001
            continue
        pooled = meta["validation"]["pooled_out_of_sample"]
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"{_cell(pooled.get('brier_skill'))}"
            f'<td class="num">{pooled.get("macro_auc") or float("nan"):.4f}</td>'
            f'<td class="num">{pooled.get("calibration_error") or float("nan"):.4f}</td>'
            f'<td class="num">{meta["train_rows"]:,}</td>'
            f'<td class="num">{meta["train_tickers"]}</td></tr>'
        )

    feedback = report["feedback_loop"]
    predictions = report["predictions"]
    healthy = feedback.get("healthy")

    tags = report["post_mortem_tags"]
    tag_rows = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="num">{v}</td></tr>' for k, v in tags.items()
    ) or '<tr><td colspan="2">No post-mortem tags yet.</td></tr>'

    return f"""<!doctype html><meta charset="utf-8">
<title>AI Company Evaluator - Monitoring</title><style>{CSS}</style>
<h1>AI Company Evaluator</h1>
<div class="sub">System monitoring report (spec 8.2). Research tooling, not investment advice.</div>

<div class="note"><strong>Skill, not accuracy.</strong> Accuracy near 70% is what
you get by always predicting "no large move". Brier skill is measured against
predicting the historical base rates; a value at or below zero means the model
carries no demonstrated information.</div>

<h2>Model skill</h2>
<table><tr><th>target</th><th>Brier skill</th><th>AUC</th><th>calib. error</th>
<th>rows</th><th>tickers</th></tr>
{''.join(rows) or '<tr><td colspan="6">No trained models.</td></tr>'}</table>

<h2>Feedback loop</h2>
<table>
<tr><th>metric</th><th>value</th></tr>
<tr><td>predictions logged</td><td class="num">{feedback['predictions_total']:,}</td></tr>
<tr><td>resolved</td><td class="num">{feedback['predictions_resolved']:,}</td></tr>
<tr><td>resolution rate</td><td class="num">{feedback['resolution_rate']}</td></tr>
<tr><td>mean days to resolution</td><td class="num">{feedback['mean_days_to_resolution']}</td></tr>
<tr><td>post-mortem coverage</td><td class="num">{feedback['post_mortem_coverage']}</td></tr>
<tr><td>healthy</td><td class="num {'ok' if healthy else 'bad'}">{healthy}</td></tr>
</table>

<h2>Prediction distribution</h2>
<pre>{html.escape(json.dumps(predictions, indent=2, default=str))}</pre>

<h2>Post-mortem tags</h2>
<table><tr><th>tag</th><th>count</th></tr>{tag_rows}</table>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    path = Path(args.out) if args.out else REPORT_DIR / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
