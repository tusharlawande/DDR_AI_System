"""
generate_report.py — CLI runner for the DDR AI pipeline v2.
Orchestrates: Extract → Parse Thermal → Generate Charts → LLM Reasoning → Build Report

Usage:
    python generate_report.py
    python generate_report.py --api-key YOUR_GEMINI_KEY
    python generate_report.py --inspection path/to/report.pdf --thermal path/to/thermal.pdf
"""
import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.extractor      import extract_document
from pipeline.thermal_parser import parse_thermal_report, compute_thermal_statistics
from pipeline.llm_engine     import LLMEngine
from pipeline.chart_generator import generate_all_charts
from pipeline.report_builder import build_html_report, export_ddr_json
from pipeline.models         import ThermalStats

DEFAULT_INSPECTION = str(Path(__file__).parent.parent / "Sample Report.pdf")
DEFAULT_THERMAL    = str(Path(__file__).parent.parent / "Thermal Images.pdf")
DEFAULT_OUTPUT     = str(Path(__file__).parent / "output")


def banner():
    print("\n" + "═"*62)
    print("  🏗️   DDR AI REPORT GENERATION SYSTEM  v2.0")
    print("  Advanced 3-Stage AI Pipeline · Charts · Confidence Scores")
    print("═"*62 + "\n")

def log(msg, icon="→"):
    print(f"  {icon}  {msg}")


def run_pipeline(inspection_pdf, thermal_pdf, api_key, model, output_dir):
    banner()
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ── 1. Extract inspection document ────────────────────────────────
    log("Extracting inspection document …", "📄")
    inspection_doc = extract_document(inspection_pdf, "inspection")
    log(f"Pages: {inspection_doc.total_pages} | Images: {len(inspection_doc.images)}", "  ✓")
    for k, v in inspection_doc.metadata.items():
        if k != "doc_type":
            log(f"{k}: {v}", "  ·")

    # ── 2. Extract & parse thermal document ───────────────────────────
    log("Extracting thermal document …", "🌡️")
    thermal_doc      = extract_document(thermal_pdf, "thermal")
    thermal_readings = parse_thermal_report(thermal_doc.full_text, thermal_doc.images)
    stats            = compute_thermal_statistics(thermal_readings)
    anomaly_count    = sum(1 for r in thermal_readings if r.anomaly_flag)
    log(f"Readings: {len(thermal_readings)} | Anomalies: {anomaly_count}", "  ✓")
    if stats:
        log(f"Hotspot range: {stats['min_hotspot']}°C – {stats['max_hotspot']}°C", "  📊")
        log(f"Avg ΔT: {stats['avg_delta']}°C | Max ΔT: {stats['max_delta']}°C @ reading #{stats.get('max_delta_reading_id','?')}", "  📊")

    # ── 3. Generate charts ────────────────────────────────────────────
    log("Generating analytical charts …", "📈")
    charts = generate_all_charts(thermal_readings, [])  # placeholder, obs added after LLM

    # ── 4. LLM reasoning ─────────────────────────────────────────────
    mode = f"Gemini ({model})" if api_key else "Advanced Rule-based Engine"
    log(f"Running AI analysis [{mode}] …", "🤖")
    engine = LLMEngine(api_key=api_key, model=model)
    ddr    = engine.generate_ddr(inspection_doc, thermal_readings)
    log(f"Overall severity: {ddr.overall_severity.value}", "  ✓")
    log(f"Areas observed:   {len(ddr.area_observations)}", "  ✓")
    log(f"Confidence:       {ddr.overall_confidence*100:.0f}%", "  ✓")
    log(f"Actions:          {len(ddr.recommended_actions)}", "  ✓")
    log(f"Missing info:     {len(ddr.missing_or_unclear_info)}", "  ✓")

    # ── 5. Attach charts (now we have observations) ───────────────────
    log("Generating area-specific charts …", "📊")
    ddr.charts = generate_all_charts(thermal_readings, ddr.area_observations)
    if stats:
        ddr.thermal_stats = ThermalStats(**{
            k: v for k, v in stats.items()
            if k in ThermalStats.model_fields
        })

    # ── 6. Build HTML report ──────────────────────────────────────────
    log("Building premium HTML report …", "📝")
    all_images = inspection_doc.images + thermal_doc.images
    html_path  = build_html_report(ddr, all_images, os.path.join(output_dir, "DDR_Report.html"))
    json_path  = export_ddr_json(ddr,              os.path.join(output_dir, "DDR_Report.json"))
    elapsed    = round(time.time() - t0, 1)

    print()
    print("═"*62)
    log(f"✅ Done in {elapsed}s!", "🎉")
    log(f"HTML Report → {html_path}", "📁")
    log(f"JSON Data   → {json_path}", "📁")
    html_kb = Path(html_path).stat().st_size // 1024
    log(f"Report size → {html_kb} KB (self-contained, shareable)", "📦")
    print("═"*62 + "\n")
    return html_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDR AI Report Generator v2")
    parser.add_argument("--inspection", default=DEFAULT_INSPECTION)
    parser.add_argument("--thermal",    default=DEFAULT_THERMAL)
    parser.add_argument("--api-key",   default=os.getenv("GOOGLE_API_KEY", ""))
    parser.add_argument("--model",     default="gemini-1.5-pro")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_pipeline(
        inspection_pdf=args.inspection,
        thermal_pdf=args.thermal,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output,
    )
