"""Line-graph artifacts for external benchmark scores."""

from __future__ import annotations

import html
from pathlib import Path

from ultron.eval.benchmarks import (
    PLOT_HTML_NAME,
    PLOT_MD_NAME,
    PLOT_SVG_NAME,
    BenchmarkId,
    ScoreRow,
    axis_labels,
    series_points,
)
from ultron.train.schema_v1 import Role

WIDTH = 960
HEIGHT = 560
MARGIN_LEFT = 72
MARGIN_RIGHT = 28
MARGIN_TOP = 48
MARGIN_BOTTOM = 108
LEGEND_Y = 28

SERIES: tuple[tuple[Role, BenchmarkId, str, str, str], ...] = (
    (Role.ATTACKER, BenchmarkId.EXPLOITBENCH, "attacker · ExploitBench", "#f38ba8", "0"),
    (Role.DEFENDER, BenchmarkId.EXPLOITBENCH, "defender · ExploitBench", "#fab387", "6 4"),
    (Role.ATTACKER, BenchmarkId.DEEP_SWE, "attacker · DeepSWE", "#89b4fa", "0"),
    (Role.DEFENDER, BenchmarkId.DEEP_SWE, "defender · DeepSWE", "#74c7ec", "6 4"),
    (Role.ATTACKER, BenchmarkId.TERMINAL_BENCH, "attacker · Terminal-Bench", "#a6e3a1", "0"),
    (Role.DEFENDER, BenchmarkId.TERMINAL_BENCH, "defender · Terminal-Bench", "#94e2d5", "6 4"),
)


def write_plots(rows: list[ScoreRow], output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    svg = render_svg(rows)
    markdown = render_markdown(rows)
    html_doc = render_html(svg)
    paths = {
        "svg": output / PLOT_SVG_NAME,
        "html": output / PLOT_HTML_NAME,
        "markdown": output / PLOT_MD_NAME,
    }
    paths["svg"].write_text(svg)
    paths["html"].write_text(html_doc)
    paths["markdown"].write_text(markdown)
    return paths


def render_svg(rows: list[ScoreRow]) -> str:
    labels = axis_labels(rows)
    plot_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img">',
        "<title>External benchmark scores by iteration stage</title>",
        '<rect width="100%" height="100%" fill="#1e1e2e"/>',
        f'<text x="{WIDTH / 2:.1f}" y="22" text-anchor="middle" fill="#cdd6f4" font-size="16" font-family="sans-serif">External benchmark scores</text>',
        f'<text x="{MARGIN_LEFT + plot_w / 2:.1f}" y="{HEIGHT - 16}" text-anchor="middle" fill="#a6adc8" font-size="12" font-family="sans-serif">iteration stage</text>',
        f'<text x="18" y="{MARGIN_TOP + plot_h / 2:.1f}" fill="#a6adc8" font-size="12" font-family="sans-serif" transform="rotate(-90 18 {MARGIN_TOP + plot_h / 2:.1f})">score</text>',
    ]
    origin_x = MARGIN_LEFT
    origin_y = MARGIN_TOP + plot_h
    parts.append(
        f'<rect x="{origin_x}" y="{MARGIN_TOP}" width="{plot_w}" height="{plot_h}" fill="#11111b" stroke="#45475a"/>'
    )
    for index in range(5):
        frac = index / 4
        y = origin_y - frac * plot_h
        label = f"{frac:.2f}"
        parts.append(
            f'<line x1="{origin_x}" y1="{y:.1f}" x2="{origin_x + plot_w}" y2="{y:.1f}" stroke="#313244" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{origin_x - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#a6adc8" font-size="11" font-family="sans-serif">{label}</text>'
        )
    if not labels:
        parts.append(
            f'<text x="{origin_x + plot_w / 2:.1f}" y="{MARGIN_TOP + plot_h / 2:.1f}" text-anchor="middle" fill="#6c7086" font-size="14" font-family="sans-serif">no scores yet</text>'
        )
    else:
        xs = _x_positions(labels, origin_x, plot_w)
        for label, x in xs.items():
            parts.append(
                f'<line x1="{x:.1f}" y1="{MARGIN_TOP}" x2="{x:.1f}" y2="{origin_y}" stroke="#313244" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{origin_y + 18}" text-anchor="end" fill="#cdd6f4" font-size="11" font-family="sans-serif" transform="rotate(-38 {x:.1f} {origin_y + 18})">{_xml(label)}</text>'
            )
        for role, benchmark, _title, color, dash in SERIES:
            points = series_points(rows, role=role, benchmark=benchmark)
            if not points:
                continue
            mapped = [(xs[label], origin_y - score * plot_h) for label, score in points if label in xs]
            if len(mapped) >= 2:
                point_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in mapped)
                dash_attr = "" if dash == "0" else f' stroke-dasharray="{dash}"'
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{dash_attr} points="{point_attr}"/>'
                )
            for x, y in mapped:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#11111b" stroke-width="1"/>')
    legend_x = MARGIN_LEFT
    for _role, _benchmark, title, color, dash in SERIES:
        dash_attr = "" if dash == "0" else f' stroke-dasharray="{dash}"'
        parts.append(
            f'<line x1="{legend_x}" y1="{LEGEND_Y + 18}" x2="{legend_x + 18}" y2="{LEGEND_Y + 18}" stroke="{color}" stroke-width="3"{dash_attr}/>'
        )
        parts.append(
            f'<text x="{legend_x + 22}" y="{LEGEND_Y + 22}" fill="#cdd6f4" font-size="11" font-family="sans-serif">{_xml(title)}</text>'
        )
        legend_x += 155
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_html(svg: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8"/>'
        "<title>Ultron external benchmark scores</title>"
        "<style>body{margin:0;background:#1e1e2e;color:#cdd6f4;font-family:sans-serif}"
        "main{max-width:960px;margin:24px auto;padding:0 16px}svg{width:100%;height:auto}</style>"
        "</head><body><main>"
        "<h1>External benchmark scores</h1>"
        "<p>X axis is iteration stage. Y axis is score. Attacker and defender are separate series.</p>"
        f"{svg}"
        "</main></body></html>\n"
    )


def render_markdown(rows: list[ScoreRow]) -> str:
    labels = axis_labels(rows)
    lines = [
        "# External benchmark scores",
        "",
        "X axis is iteration stage. Y axis is score. Attacker and defender are scored separately.",
        "",
        "| role | benchmark | stage | score | status | n |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    if not rows:
        lines.append("| — | — | — | n/a | none | n/a |")
    else:
        for row in rows:
            score = "n/a" if row.score is None else f"{row.score:.3f}"
            n = "n/a" if row.n is None else str(row.n)
            lines.append(
                f"| `{row.role.value}` | `{row.benchmark.value}` | `{row.stage_label}` | {score} | `{row.status.value}` | {n} |"
            )
    lines.extend(["", "## Line graph", "", "Open `scores.svg` or `scores.html` for the plotted series.", ""])
    if labels and any(row.score is not None for row in rows):
        lines.extend(_mermaid(rows, labels))
    return "\n".join(lines) + "\n"


def _mermaid(rows: list[ScoreRow], labels: list[str]) -> list[str]:
    lines = [
        "```mermaid",
        "xychart-beta",
        '    title "External benchmark scores"',
        "    x-axis [" + ", ".join(_quoted(label) for label in labels) + "]",
        '    y-axis "score" 0 --> 1',
    ]
    for role, benchmark, title, _color, _dash in SERIES:
        lookup = dict(series_points(rows, role=role, benchmark=benchmark))
        if not lookup:
            continue
        values = [f"{lookup[label]:.3f}" if label in lookup else "none" for label in labels]
        if all(item == "none" for item in values):
            continue
        numeric = ["0" if item == "none" else item for item in values]
        lines.append(f"    line [{', '.join(numeric)}]")
        _ = title
    lines.append("```")
    lines.append("")
    return lines


def _x_positions(labels: list[str], origin_x: float, plot_w: float) -> dict[str, float]:
    if len(labels) == 1:
        return {labels[0]: origin_x + plot_w / 2}
    return {label: origin_x + index * plot_w / (len(labels) - 1) for index, label in enumerate(labels)}


def _xml(value: str) -> str:
    return html.escape(value, quote=True)


def _quoted(value: str) -> str:
    return '"' + value.replace('"', "") + '"'
