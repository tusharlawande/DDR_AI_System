"""
Chart Generator — creates matplotlib visualizations embedded in the DDR report.
Generates: thermal profile, ΔT trend, severity distribution, anomaly heatmap.
All charts output as base64 PNG strings for inline HTML embedding.
"""
from __future__ import annotations
import base64
import io
import math
from typing import List, Optional

from pipeline.models import ThermalReading, AreaObservation, ChartData, SeverityLevel

# ── matplotlib setup ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# ── Palette ───────────────────────────────────────────────────────────────────
DARK_BG   = "#0F1923"
CARD_BG   = "#161F2E"
BORDER    = "#1E3A5F"
HOT_CLR   = "#FF3B30"
COLD_CLR  = "#0077CC"
DELTA_CLR = "#FF9500"
TEXT_CLR  = "#E0EAF6"
MUTED_CLR = "#5A7A9A"
GRID_CLR  = "#1E3A5F"

SEV_COLORS = {
    "Critical": "#FF3B30",
    "High": "#FF9500",
    "Moderate": "#FFCC00",
    "Low": "#34C759",
    "Not Available": "#5A7A9A",
}


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
def generate_thermal_profile_chart(readings: List[ThermalReading]) -> str:
    """
    Grouped bar chart: Hotspot vs Coldspot temperature for every reading.
    Anomalous readings highlighted in orange.
    """
    n = len(readings)
    x = list(range(n))
    hotspots = [r.hotspot_temp for r in readings]
    coldspots = [r.coldspot_temp for r in readings]
    bar_colors = [HOT_CLR if r.anomaly_flag else "#CC3322" for r in readings]

    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)

    width = 0.38
    bars_h = ax.bar([i - width/2 for i in x], hotspots, width, color=HOT_CLR,
                     alpha=0.85, label="Hotspot (°C)", zorder=3)
    bars_c = ax.bar([i + width/2 for i in x], coldspots, width, color=COLD_CLR,
                     alpha=0.85, label="Coldspot (°C)", zorder=3)

    # Highlight anomalies with border
    for i, r in enumerate(readings):
        if r.anomaly_flag:
            ax.bar(i - width/2, hotspots[i], width, color="none",
                   edgecolor=DELTA_CLR, linewidth=2.5, zorder=4)

    ax.axhline(23, color="#FFFFFF", linewidth=1, linestyle="--", alpha=0.3,
               label="Ambient Temp (23°C)")

    ax.set_xlabel("Thermal Reading #", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("Temperature (°C)", color=TEXT_CLR, fontsize=10)
    ax.set_title("Thermal Profile — Hotspot vs Coldspot per Reading",
                 color=TEXT_CLR, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r.image_index) for r in readings],
                       color=MUTED_CLR, fontsize=8, rotation=45)
    ax.tick_params(colors=MUTED_CLR)
    ax.spines[:].set_color(BORDER)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f°"))
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.7, zorder=0)
    ax.set_ylim(18, 32)

    legend = ax.legend(facecolor=DARK_BG, edgecolor=BORDER,
                       labelcolor=TEXT_CLR, fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
def generate_delta_temp_chart(readings: List[ThermalReading]) -> str:
    """
    Line chart of ΔT (Hotspot − Coldspot) across all readings.
    Shaded bands indicate severity thresholds.
    """
    x = [r.image_index for r in readings]
    deltas = [r.delta_temp for r in readings]

    fig, ax = plt.subplots(figsize=(14, 3.8), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)

    # Threshold bands
    ax.axhspan(5.0, 7.0, alpha=0.12, color=HOT_CLR, label="High Risk Zone (ΔT>5°C)")
    ax.axhspan(3.0, 5.0, alpha=0.10, color=DELTA_CLR, label="Caution Zone (ΔT 3–5°C)")
    ax.axhspan(0.0, 3.0, alpha=0.06, color="#34C759", label="Normal Zone (ΔT<3°C)")

    ax.plot(x, deltas, color=DELTA_CLR, linewidth=2.2, zorder=4, marker="o",
            markersize=5, markerfacecolor=DARK_BG, markeredgecolor=DELTA_CLR, markeredgewidth=1.8)

    # Mark anomalies
    for r in readings:
        if r.anomaly_flag:
            ax.plot(r.image_index, r.delta_temp, "o", color=HOT_CLR,
                    markersize=9, zorder=5)
            ax.annotate(f"#{r.image_index}\n{r.delta_temp}°",
                        (r.image_index, r.delta_temp),
                        textcoords="offset points", xytext=(0, 10),
                        color=HOT_CLR, fontsize=7.5, ha="center", fontweight="bold")

    ax.set_xlabel("Reading #", color=TEXT_CLR, fontsize=10)
    ax.set_ylabel("ΔT (°C)", color=TEXT_CLR, fontsize=10)
    ax.set_title("Temperature Delta (ΔT) Trend — Moisture Indicator",
                 color=TEXT_CLR, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=MUTED_CLR)
    ax.spines[:].set_color(BORDER)
    ax.grid(color=GRID_CLR, linewidth=0.6, zorder=0)
    ax.set_ylim(-0.5, max(deltas) + 1.5)

    legend = ax.legend(facecolor=DARK_BG, edgecolor=BORDER,
                       labelcolor=TEXT_CLR, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
def generate_severity_distribution_chart(observations: List[AreaObservation]) -> str:
    """
    Donut chart showing distribution of severity levels across areas.
    """
    counts: dict[str, int] = {}
    for obs in observations:
        lvl = obs.severity.value
        counts[lvl] = counts.get(lvl, 0) + 1

    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [SEV_COLORS.get(l, "#5A7A9A") for l in labels]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.0f%%", startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.55, "edgecolor": DARK_BG, "linewidth": 3},
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontweight("bold")
        at.set_fontsize(11)

    # Centre label
    ax.text(0, 0, f"{len(observations)}\nAreas", ha="center", va="center",
            color=TEXT_CLR, fontsize=13, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=SEV_COLORS.get(l, "#5A7A9A"), label=f"{l} ({c})")
        for l, c in counts.items()
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              facecolor=DARK_BG, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=9)

    ax.set_title("Severity Distribution", color=TEXT_CLR,
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
def generate_confidence_chart(observations: List[AreaObservation]) -> str:
    """
    Horizontal bar chart of confidence score per area.
    """
    names = [obs.area_name.split("—")[0].strip() for obs in observations]
    scores = [obs.confidence_score * 100 for obs in observations]
    colors = [SEV_COLORS.get(obs.severity.value, "#5A7A9A") for obs in observations]

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)

    bars = ax.barh(names, scores, color=colors, alpha=0.85, height=0.55)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{score:.0f}%", va="center", color=TEXT_CLR, fontsize=9, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence Score (%)", color=TEXT_CLR, fontsize=9)
    ax.set_title("Observation Confidence Scores", color=TEXT_CLR,
                 fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED_CLR, axis="both")
    ax.spines[:].set_color(BORDER)
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.6)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
def generate_all_charts(
    readings: List[ThermalReading],
    observations: List[AreaObservation],
) -> ChartData:
    """Generate all charts and return as ChartData model."""
    thermal_profile = generate_thermal_profile_chart(readings) if readings else None
    delta_chart     = generate_delta_temp_chart(readings) if readings else None
    severity_chart  = generate_severity_distribution_chart(observations) if observations else None
    confidence_chart = generate_confidence_chart(observations) if observations else None

    return ChartData(
        thermal_profile_chart=thermal_profile,
        delta_temp_chart=delta_chart,
        severity_distribution_chart=severity_chart,
        anomaly_heatmap=confidence_chart,
    )
