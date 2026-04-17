"""
app.py — DDR AI System v2  ·  Streamlit Web Application
Premium dark UI with live progress, in-browser preview, and multi-format download.
"""
import os, sys, json, time, tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.extractor       import extract_document
from pipeline.thermal_parser  import parse_thermal_report, compute_thermal_statistics
from pipeline.llm_engine      import LLMEngine
from pipeline.chart_generator import generate_all_charts
from pipeline.report_builder  import build_html_report, export_ddr_json
from pipeline.models          import ThermalStats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDR AI System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0A1628; }
.block-container { padding-top: 1.5rem !important; }

/* Hero */
.hero {
  background: linear-gradient(145deg, #060F1F 0%, #0A1628 50%, #0D1E35 100%);
  border: 1px solid #1B3050;
  border-radius: 16px; padding: 40px 44px; margin-bottom: 28px;
  position: relative; overflow: hidden;
}
.hero::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 70% 50% at 80% 30%, rgba(10,132,255,.07) 0%, transparent 70%);
}
.hero-tag {
  display: inline-flex; align-items: center; gap: 8px;
  border: 1px solid rgba(10,132,255,.4); border-radius: 20px;
  padding: 5px 14px; font-size: 11px; font-weight: 700; letter-spacing: .7px;
  text-transform: uppercase; color: #0A84FF; background: rgba(10,132,255,.08);
  margin-bottom: 20px;
}
.hero-tag .dot { width: 7px; height: 7px; background: #34C759; border-radius: 50%; animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.hero-title { font-size: 32px; font-weight: 900; color: white; letter-spacing: -1px; margin-bottom: 8px; }
.hero-sub   { font-size: 15px; color: #7A9CC0; margin: 0; }

/* Sidebar */
.sidebar-title { font-size: 18px; font-weight: 800; color: white; text-align: center; }
.sidebar-ver   { font-size: 11px; color: #3A5A7A; text-align: center; margin-top: 4px; }
.step-box {
  background: rgba(10,132,255,.08); border: 1px solid rgba(10,132,255,.2);
  border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
  color: rgba(255,255,255,.8); font-size: 13px;
}
.step-num {
  display: inline-block; background: #0A84FF; color: white;
  width: 20px; height: 20px; border-radius: 50%;
  text-align: center; line-height: 20px; font-size: 11px; font-weight: 800;
  margin-right: 8px;
}

/* Upload cards */
.upload-card {
  background: rgba(255,255,255,.03); border: 1px dashed rgba(10,132,255,.35);
  border-radius: 12px; padding: 20px; text-align: center; color: #7A9CC0;
  font-size: 13px; margin-bottom: 8px;
}

/* Metric cards */
.metric-card {
  background: rgba(255,255,255,.03); border: 1px solid #1B3050;
  border-radius: 10px; padding: 16px; text-align: center;
}
.metric-val  { font-size: 26px; font-weight: 900; color: white; line-height: 1; }
.metric-lbl  { font-size: 11px; color: #7A9CC0; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; margin-top: 5px; }

/* Observation cards */
.obs-card {
  background: rgba(255,255,255,.03); border: 1px solid #1B3050;
  border-radius: 10px; padding: 16px; margin-bottom: 12px;
}
.obs-card-title { font-size: 14px; font-weight: 700; color: white; margin-bottom: 8px; }

/* Severity badges */
.badge-critical { color:#FF3B30; background:rgba(255,59,48,.12); padding:3px 10px; border-radius:10px; font-size:11px; font-weight:800; border:1px solid rgba(255,59,48,.3); }
.badge-high     { color:#FF9500; background:rgba(255,149,0,.12); padding:3px 10px; border-radius:10px; font-size:11px; font-weight:800; border:1px solid rgba(255,149,0,.3); }
.badge-moderate { color:#FFCC02; background:rgba(255,204,2,.12); padding:3px 10px; border-radius:10px; font-size:11px; font-weight:800; border:1px solid rgba(255,204,2,.3); }
.badge-low      { color:#34C759; background:rgba(52,199,89,.12); padding:3px 10px; border-radius:10px; font-size:11px; font-weight:800; border:1px solid rgba(52,199,89,.3); }

/* Progress steps */
.prog-step { display:flex; align-items:center; gap:10px; padding:8px 0; font-size:13px; color:#7A9CC0; }
.prog-step.done   { color:#34C759; }
.prog-step.active { color:#60A5FA; font-weight:600; }

div[data-testid="stButton"] button {
  background: linear-gradient(135deg, #0A84FF, #0055CC) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; font-size: 15px !important;
  padding: 14px 28px !important; width: 100% !important;
  box-shadow: 0 4px 20px rgba(10,132,255,.4) !important;
  transition: all .2s !important;
}
div[data-testid="stDownloadButton"] button {
  background: rgba(10,132,255,.1) !important;
  border: 1px solid rgba(10,132,255,.4) !important;
  color: #0A84FF !important; border-radius: 8px !important;
  font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px; text-align:center'>
      <div style='font-size:40px'>🏗️</div>
      <div class='sidebar-title'>DDR AI System</div>
      <div class='sidebar-ver'>v2.0 · Advanced Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="AIza… (leave blank = rule-based)",
        help="Optional. Enables 3-stage Gemini reasoning. Without key, the advanced rule-based engine runs offline."
    )
    model_choice = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"])

    st.divider()
    st.markdown("**📋 Pipeline Steps**")
    for n, t in [("1","Extract PDF text + images"),("2","Parse 30 thermal readings"),
                 ("3","Generate 4 analytical charts"),("4","AI reasoning chain (3 stages)"),
                 ("5","Build self-contained HTML report")]:
        st.markdown(f"<div class='step-box'><span class='step-num'>{n}</span>{t}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#3A5A7A;text-align:center;line-height:1.7'>
    AI/ML Engineer Assignment<br>DDR Report Generation<br>Advanced Pipeline v2.0
    </div>""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-tag'><span class='dot'></span>AI-Powered · DDR v2.0 · Advanced Pipeline</div>
  <h1 class='hero-title'>DDR Report Generation AI System</h1>
  <p class='hero-sub'>Upload inspection + thermal documents → get a structured, chart-rich Detailed Diagnostic Report in seconds</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("**📄 Inspection Report**")
    inspection_file = st.file_uploader("Inspection PDF", type=["pdf"], label_visibility="collapsed", key="insp")
    if inspection_file:
        st.success(f"✅ {inspection_file.name}  ({inspection_file.size//1024} KB)")
with col2:
    st.markdown("**🌡️ Thermal Imaging Report**")
    thermal_file = st.file_uploader("Thermal PDF", type=["pdf"], label_visibility="collapsed", key="therm")
    if thermal_file:
        st.success(f"✅ {thermal_file.name}  ({thermal_file.size//1024} KB)")

st.divider()
can_run = bool(inspection_file and thermal_file)
run_btn = st.button("🚀 Generate DDR Report", disabled=not can_run)

# ── Pipeline execution ─────────────────────────────────────────────────────────
if run_btn and can_run:
    st.divider()

    # Progress display
    prog_col, stat_col = st.columns([2, 1])
    with prog_col:
        prog_bar   = st.progress(0)
        prog_label = st.empty()

    step_statuses = [
        "📄 Extracting inspection document…",
        "🌡️ Parsing thermal readings…",
        "📈 Generating analytical charts…",
        "🤖 Running AI reasoning chain…",
        "📝 Building HTML report…",
    ]

    def update(i, msg=None):
        prog_bar.progress((i + 1) / len(step_statuses))
        prog_label.markdown(f"**{step_statuses[i]}**")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            insp_path  = os.path.join(tmp, "inspection.pdf")
            therm_path = os.path.join(tmp, "thermal.pdf")
            html_path  = os.path.join(tmp, "DDR_Report.html")
            json_path  = os.path.join(tmp, "DDR_Report.json")

            Path(insp_path).write_bytes(inspection_file.getvalue())
            Path(therm_path).write_bytes(thermal_file.getvalue())

            # Step 1
            update(0)
            insp_doc = extract_document(insp_path, "inspection")

            # Step 2
            update(1)
            therm_doc = extract_document(therm_path, "thermal")
            readings  = parse_thermal_report(therm_doc.full_text, therm_doc.images)
            stats     = compute_thermal_statistics(readings)

            # Step 3
            update(2)
            charts_placeholder = generate_all_charts(readings, [])  # initial pass

            # Step 4
            update(3)
            engine = LLMEngine(api_key=api_key, model=model_choice)
            ddr    = engine.generate_ddr(insp_doc, readings)
            ddr.charts = generate_all_charts(readings, ddr.area_observations)
            if stats:
                ddr.thermal_stats = ThermalStats(**{k: v for k, v in stats.items() if k in ThermalStats.model_fields})

            # Step 5
            update(4)
            all_imgs  = insp_doc.images + therm_doc.images
            html_out  = build_html_report(ddr, all_imgs, html_path)
            json_out  = export_ddr_json(ddr, json_path)

            html_bytes = Path(html_out).read_bytes()
            json_text  = Path(json_out).read_text(encoding="utf-8")

        prog_bar.progress(1.0)
        prog_label.markdown("✅ **Report generated successfully!**")

        # ── Results ────────────────────────────────────────────────────────────
        st.success(f"🎉 DDR Report ready! Report ID: `{ddr.report_id}`")

        # KPI metrics
        sev_icon = {"Critical":"🔴","High":"🟠","Moderate":"🟡","Low":"🟢"}.get(ddr.overall_severity.value,"⚪")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Severity",    f"{sev_icon} {ddr.overall_severity.value}")
        m2.metric("Areas",       len(ddr.area_observations))
        m3.metric("Readings",    len(readings))
        m4.metric("Actions",     len(ddr.recommended_actions))
        m5.metric("AI Confidence", f"{ddr.overall_confidence*100:.0f}%")
        m6.metric("Gen Time",    f"{ddr.processing_time_seconds}s")

        # Downloads
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("📥 Download DDR Report (HTML · Self-contained)",
                               html_bytes, "DDR_Report.html", "text/html", use_container_width=True)
        with d2:
            st.download_button("📥 Download Structured Data (JSON)",
                               json_text, "DDR_Report.json", "application/json", use_container_width=True)

        st.divider()

        # Tabbed preview
        tab_sum, tab_areas, tab_thermal, tab_actions, tab_raw = st.tabs([
            "📋 Summary", "🏢 Area Observations", "🌡️ Thermal Analysis", "🔧 Actions", "🔍 Raw Data"
        ])

        # ── Summary tab ──────────────────────────────────────────────────────
        with tab_sum:
            st.markdown("### 1. Property Issue Summary")
            st.info(ddr.property_issue_summary)

            if ddr.estimated_repair_urgency and ddr.estimated_repair_urgency != "Not Available":
                st.error(f"⏰ **Repair Urgency:** {ddr.estimated_repair_urgency}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🏠 Property Details**")
                st.json({"Inspection Date": ddr.inspection_date,
                         "Inspected By": ddr.inspected_by,
                         "Property Type": ddr.property_type,
                         "Floors": ddr.floors,
                         "AI Engine": ddr.ai_model_used})
            with c2:
                st.markdown("**❌ Missing Information (Section 7)**")
                for item in ddr.missing_or_unclear_info:
                    st.markdown(f"- `N/A` &nbsp; {item}")

            if ddr.charts and ddr.charts.severity_distribution_chart:
                st.markdown("**📊 Severity Distribution**")
                import base64
                img_data = base64.b64decode(ddr.charts.severity_distribution_chart)
                st.image(img_data, use_container_width=False, width=380)

        # ── Areas tab ─────────────────────────────────────────────────────────
        with tab_areas:
            st.markdown("### 2. Area-wise Observations")
            sev_icon_map = {"Critical":"🔴","High":"🟠","Moderate":"🟡","Low":"🟢","Not Available":"⚪"}
            for obs in ddr.area_observations:
                icon = sev_icon_map.get(obs.severity.value, "⚪")
                conf_pct = f"{obs.confidence_score*100:.0f}%" if obs.confidence_score > 0 else "N/A"
                with st.expander(f"{icon} **{obs.area_name}** — {obs.severity.value} | Confidence: {conf_pct}"):
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown("**⬇ Negative Side (Affected)**")
                        st.write(obs.negative_side_description)
                    with cb:
                        st.markdown("**⬆ Positive Side (Source)**")
                        st.write(obs.positive_side_description)

                    st.markdown(f"**Root Cause:** {obs.root_cause}")
                    st.markdown(f"**Severity Reasoning:** {obs.severity_reasoning}")
                    if obs.chain_of_thought:
                        st.markdown("**🧠 AI Reasoning Trace:**")
                        st.code(obs.chain_of_thought, language=None)
                    if obs.recommended_actions:
                        st.markdown("**Recommended Actions:**")
                        for act in obs.recommended_actions:
                            st.markdown(f"→ {act}")

        # ── Thermal tab ────────────────────────────────────────────────────────
        with tab_thermal:
            st.markdown("### Thermal Imaging Analysis")
            st.write(ddr.thermal_analysis_summary)

            if stats:
                t1,t2,t3,t4,t5,t6 = st.columns(6)
                t1.metric("Max Hotspot",  f"{stats['max_hotspot']}°C")
                t2.metric("Min Coldspot", f"{stats['min_coldspot']}°C")
                t3.metric("Avg ΔT",       f"{stats['avg_delta']}°C")
                t4.metric("Max ΔT",       f"{stats['max_delta']}°C")
                t5.metric("Readings",     stats['total_readings'])
                t6.metric("Anomalies",    stats['anomaly_count'])

            import base64 as b64lib
            if ddr.charts:
                if ddr.charts.thermal_profile_chart:
                    st.markdown("**🌡️ Thermal Profile Chart**")
                    st.image(b64lib.b64decode(ddr.charts.thermal_profile_chart), use_container_width=True)
                if ddr.charts.delta_temp_chart:
                    st.markdown("**📈 ΔT Trend Chart**")
                    st.image(b64lib.b64decode(ddr.charts.delta_temp_chart), use_container_width=True)
                if ddr.charts.anomaly_heatmap:
                    st.markdown("**🎯 Confidence Scores**")
                    st.image(b64lib.b64decode(ddr.charts.anomaly_heatmap), use_container_width=True)

            if ddr.thermal_anomalies:
                st.warning(f"⚠ {len(ddr.thermal_anomalies)} thermal anomalies detected:")
                for a in ddr.thermal_anomalies[:10]:
                    st.markdown(f"- {a}")
            if ddr.conflicts_detected:
                for c in ddr.conflicts_detected:
                    st.error(f"⚡ Conflict: {c}")

        # ── Actions tab ────────────────────────────────────────────────────────
        with tab_actions:
            st.markdown("### 3. Root Causes")
            for i, c in enumerate(ddr.probable_root_causes, 1):
                st.markdown(f"**{i}.** {c}")
            st.divider()
            st.markdown("### 5. Recommended Actions")
            if ddr.priority_actions:
                st.error("**🚨 Immediate Priority Actions:**")
                for a in ddr.priority_actions:
                    st.markdown(f"⚡ {a}")
            st.markdown("**Full Action Plan:**")
            for a in ddr.recommended_actions:
                st.markdown(f"✓ {a}")
            st.divider()
            st.markdown("### 6. Additional Notes")
            for n in ddr.additional_notes:
                st.markdown(f"📌 {n}")

        # ── Raw data tab ───────────────────────────────────────────────────────
        with tab_raw:
            st.markdown("### Structured DDR Data (JSON)")
            st.caption("This is the machine-readable output of the AI pipeline. Useful for downstream integrations.")
            raw = json.loads(json_text)
            # Remove chart blobs from display
            if "charts" in raw:
                raw["charts"] = {k: "[base64 chart data]" for k in (raw["charts"] or {}) if raw["charts"][k]}
            st.json(raw)

    except Exception as exc:
        st.error(f"❌ Pipeline error: {exc}")
        st.exception(exc)

# ── Empty state ────────────────────────────────────────────────────────────────
elif not (inspection_file and thermal_file):
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#3A5A7A'>
      <div style='font-size:64px;margin-bottom:16px'>📂</div>
      <div style='font-size:20px;font-weight:700;color:#7A9CC0;margin-bottom:8px'>
        Upload Both Documents to Begin
      </div>
      <div style='font-size:14px'>Inspection Report (PDF) + Thermal Images Report (PDF)</div>
    </div>
    """, unsafe_allow_html=True)
