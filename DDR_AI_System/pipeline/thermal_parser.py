"""
Thermal Parser v2 — structured parsing + anomaly detection + statistics.
"""
from __future__ import annotations
import re
from typing import List
from pipeline.models import ThermalReading, ExtractedImage, ThermalStats


def parse_thermal_report(text: str, images: List[ExtractedImage]) -> List[ThermalReading]:
    block_pattern = re.compile(
        r"Hotspot\s*:\s*([\d.]+)\s*°C\s*"
        r"Coldspot\s*:\s*([\d.]+)\s*°C\s*"
        r"Emissivity\s*:\s*([\d.]+)\s*"
        r"Reflected temperature\s*:\s*([\d.]+)\s*°C\s*"
        r"Thermal image\s*:\s*(\S+)\s*"
        r"Device\s*:\s*([^\n]+?)\s*"
        r"Serial Number\s*:\s*(\S+)\s*"
        r"(\d+)",
        re.IGNORECASE | re.DOTALL,
    )
    date_pattern = re.compile(r"(\d{2}/\d{2}/\d{2,4})")
    readings: List[ThermalReading] = []

    for match in block_pattern.finditer(text):
        block_start = match.start()
        preceding   = text[max(0, block_start - 60): block_start + 20]
        date_m      = date_pattern.search(preceding)
        readings.append(ThermalReading(
            image_index=int(match.group(8)),
            image_filename=match.group(5).strip(),
            capture_date=date_m.group(1) if date_m else "Not Available",
            hotspot_temp=float(match.group(1)),
            coldspot_temp=float(match.group(2)),
            emissivity=float(match.group(3)),
            reflected_temperature=float(match.group(4)),
            device=match.group(6).strip(),
            serial_number=match.group(7).strip(),
        ))
    return readings


def compute_thermal_statistics(readings: List[ThermalReading]) -> dict:
    if not readings:
        return {}
    hotspots     = [r.hotspot_temp for r in readings]
    coldspots    = [r.coldspot_temp for r in readings]
    deltas       = [r.delta_temp   for r in readings]
    anomaly_count = sum(1 for r in readings if r.anomaly_flag)
    max_delta_r  = max(readings, key=lambda r: r.delta_temp)

    return {
        "total_readings":  len(readings),
        "max_hotspot":     max(hotspots),
        "min_hotspot":     min(hotspots),
        "avg_hotspot":     round(sum(hotspots) / len(hotspots), 2),
        "max_coldspot":    max(coldspots),
        "min_coldspot":    min(coldspots),
        "avg_coldspot":    round(sum(coldspots) / len(coldspots), 2),
        "max_delta":       round(max(deltas), 2),
        "min_delta":       round(min(deltas), 2),
        "avg_delta":       round(sum(deltas) / len(deltas), 2),
        "anomaly_count":   anomaly_count,
        "max_delta_reading_id": max_delta_r.image_index,
        "device":          readings[0].device if readings else "Not Available",
        "capture_date":    readings[0].capture_date if readings else "Not Available",
    }


def detect_thermal_anomalies(readings: List[ThermalReading]) -> List[str]:
    return [r.anomaly_reason for r in readings if r.anomaly_flag and r.anomaly_reason]


def format_thermal_summary(readings: List[ThermalReading]) -> str:
    if not readings:
        return "No thermal data available."
    stats     = compute_thermal_statistics(readings)
    anomalies = [r for r in readings if r.anomaly_flag]

    summary = (
        f"Thermal inspection was conducted on {stats['capture_date']} using a "
        f"{stats['device']} (Serial: {readings[0].serial_number}). "
        f"A total of {stats['total_readings']} thermal readings were captured across the property. "
        f"Surface temperatures ranged from {stats['min_coldspot']}°C (coldspot) to "
        f"{stats['max_hotspot']}°C (hotspot). "
        f"The average temperature differential (ΔT) across all readings was "
        f"{stats['avg_delta']}°C, with a peak ΔT of {stats['max_delta']}°C recorded at "
        f"Reading #{stats['max_delta_reading_id']}. "
    )
    if anomalies:
        summary += (
            f"A total of {len(anomalies)} readings were flagged as anomalous based on elevated "
            f"temperature differentials (ΔT > 3°C threshold), strongly corroborating the "
            f"inspection findings of active moisture ingress across the property."
        )
    else:
        summary += "No significant thermal anomalies were detected."
    return summary
