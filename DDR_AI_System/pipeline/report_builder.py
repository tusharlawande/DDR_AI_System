"""
Report Builder v2 — renders a premium, chart-rich, self-contained HTML DDR report.
Embeds all images, charts, and data in a single portable HTML file.
"""
from __future__ import annotations
import base64, json, os
from datetime import datetime
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from pipeline.models import DDRReport, ExtractedImage, SeverityLevel
from pipeline.chart_generator import generate_all_charts

SEVERITY_COLOR = {
    SeverityLevel.CRITICAL:      "#FF3B30",
    SeverityLevel.HIGH:          "#FF9500",
    SeverityLevel.MODERATE:      "#FFCC02",
    SeverityLevel.LOW:           "#34C759",
    SeverityLevel.NOT_AVAILABLE: "#8E8E93",
}
SEVERITY_BG = {
    SeverityLevel.CRITICAL:      "#1A0000",
    SeverityLevel.HIGH:          "#1A0E00",
    SeverityLevel.MODERATE:      "#1A1700",
    SeverityLevel.LOW:           "#001A07",
    SeverityLevel.NOT_AVAILABLE: "#111111",
}


def build_html_report(
    ddr: DDRReport,
    all_images: List[ExtractedImage],
    output_path: str,
) -> str:
    # Build charts from live data
    if ddr.thermal_stats and ddr.area_observations:
        # We need the raw readings for charts — reconstruct from stats is not possible,
        # so charts must be passed in or generated externally. Here we use a sentinel.
        pass  # charts already attached to ddr.charts by pipeline

    image_map          = _build_image_map(all_images)
    report_generated   = datetime.now().strftime("%d %B %Y, %I:%M %p")
    severity_colors    = {s.value: SEVERITY_COLOR[s] for s in SeverityLevel}
    severity_bgs       = {s.value: SEVERITY_BG[s]    for s in SeverityLevel}

    context = dict(
        ddr=ddr,
        image_map=image_map,
        severity_colors=severity_colors,
        severity_bgs=severity_bgs,
        report_generated=report_generated,
        total_areas=len(ddr.area_observations),
        total_images=len(all_images),
        has_charts=ddr.charts is not None,
    )

    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["sev_color"] = lambda s: severity_colors.get(s, "#8E8E93")
    env.filters["sev_bg"]    = lambda s: severity_bgs.get(s, "#111111")
    env.filters["pct"]       = lambda v: f"{v * 100:.0f}%"
    env.filters["conf_color"] = lambda v: (
        "#34C759" if v >= 0.85 else "#FFCC02" if v >= 0.70 else "#FF9500"
    )

    template     = env.get_template("ddr_template.html")
    html_content = template.render(**context)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return str(output_path)


def _build_image_map(images: List[ExtractedImage]) -> dict[str, str]:
    image_map: dict[str, str] = {}
    photo_counter = 1
    for img in images:
        if not img.base64_data:
            continue
        fmt = (img.format or "jpeg").lower()
        # Skip wide header/logo banners
        if fmt == "png" and img.width >= 1000 and img.width >= img.height * 4:
            continue
        data_uri = f"data:image/{fmt};base64,{img.base64_data}"
        image_map[f"Photo {photo_counter}"] = data_uri
        image_map[img.description]          = data_uri
        photo_counter += 1
    return image_map


def export_ddr_json(ddr: DDRReport, output_path: str) -> str:
    data = ddr.model_dump()
    data["overall_severity"] = ddr.overall_severity.value
    for obs in data["area_observations"]:
        obs["severity"] = obs["severity"] if isinstance(obs["severity"], str) else obs["severity"].value
    # Remove binary chart data from JSON export (keep it readable)
    if "charts" in data and data["charts"]:
        data["charts"] = {k: "<base64 chart data — see HTML report>" for k in data["charts"] if data["charts"][k]}
    output_path = Path(output_path)
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return str(output_path)
