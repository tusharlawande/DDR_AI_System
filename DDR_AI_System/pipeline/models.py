"""
Data models for the DDR AI pipeline — Advanced Version.
Includes confidence scoring, chain-of-thought fields, and vision analysis results.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from enum import Enum


class SeverityLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    NOT_AVAILABLE = "Not Available"


class ThermalReading(BaseModel):
    image_index: int
    image_filename: str
    capture_date: str
    hotspot_temp: float
    coldspot_temp: float
    emissivity: float
    reflected_temperature: float
    device: str
    serial_number: str
    delta_temp: float = 0.0
    anomaly_flag: bool = False
    anomaly_reason: str = ""

    @model_validator(mode="after")
    def compute_delta(self) -> "ThermalReading":
        self.delta_temp = round(self.hotspot_temp - self.coldspot_temp, 2)
        if self.delta_temp > 5.0:
            self.anomaly_flag = True
            self.anomaly_reason = f"High ΔT={self.delta_temp}°C — active moisture/leakage suspected"
        elif self.delta_temp > 3.0:
            self.anomaly_flag = True
            self.anomaly_reason = f"Elevated ΔT={self.delta_temp}°C — monitor for moisture ingress"
        if self.hotspot_temp > 27.5:
            self.anomaly_flag = True
            self.anomaly_reason += f" | Surface hotspot {self.hotspot_temp}°C — possible heat bridge"
        return self


class ExtractedImage(BaseModel):
    page_number: int
    image_index_on_page: int
    description: str = "Image Not Available"
    base64_data: Optional[str] = None
    width: int = 0
    height: int = 0
    format: str = "jpeg"
    vision_analysis: Optional[str] = None  # Gemini vision description


class DocumentExtraction(BaseModel):
    document_type: str
    total_pages: int
    full_text: str
    images: List[ExtractedImage] = []
    metadata: Dict[str, Any] = {}


class Evidence(BaseModel):
    """A single piece of evidence supporting an observation."""
    source: str           # "inspection" | "thermal" | "vision"
    description: str
    confidence: float     # 0.0 – 1.0
    photo_refs: List[str] = []
    thermal_reading_ids: List[int] = []


class AreaObservation(BaseModel):
    area_name: str
    flat_number: Optional[str] = None
    negative_side_description: str
    positive_side_description: str
    severity: SeverityLevel = SeverityLevel.NOT_AVAILABLE
    severity_reasoning: str = "Not Available"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="0-1 confidence in this observation based on evidence quality")
    evidence: List[Evidence] = []
    associated_thermal_readings: List[int] = []
    image_refs: List[str] = []
    root_cause: str = "Not Available"
    recommended_actions: List[str] = []
    chain_of_thought: str = ""  # AI reasoning trace


class ThermalStats(BaseModel):
    total_readings: int = 0
    max_hotspot: float = 0.0
    min_hotspot: float = 0.0
    avg_hotspot: float = 0.0
    max_coldspot: float = 0.0
    min_coldspot: float = 0.0
    avg_coldspot: float = 0.0
    max_delta: float = 0.0
    min_delta: float = 0.0
    avg_delta: float = 0.0
    anomaly_count: int = 0
    device: str = "Not Available"
    capture_date: str = "Not Available"


class ChartData(BaseModel):
    """Base64-encoded chart images for embedding in the report."""
    thermal_profile_chart: Optional[str] = None    # Hotspot/coldspot bar chart
    delta_temp_chart: Optional[str] = None         # ΔT trend chart
    severity_distribution_chart: Optional[str] = None  # Pie/donut chart
    anomaly_heatmap: Optional[str] = None          # Anomaly count heatmap


class DDRReport(BaseModel):
    # ── Header ────────────────────────────────────────────────────────────────
    report_title: str = "Detailed Diagnostic Report (DDR)"
    report_id: str = ""
    property_address: str = "Not Available"
    inspection_date: str = "Not Available"
    inspected_by: str = "Not Available"
    property_type: str = "Not Available"
    property_age: str = "Not Available"
    floors: str = "Not Available"

    # ── AI Metadata ───────────────────────────────────────────────────────────
    ai_model_used: str = "Rule-based engine"
    pipeline_version: str = "2.0.0"
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_seconds: float = 0.0

    # ── Section 1 ─────────────────────────────────────────────────────────────
    property_issue_summary: str

    # ── Section 2 ─────────────────────────────────────────────────────────────
    area_observations: List[AreaObservation]

    # ── Section 3 ─────────────────────────────────────────────────────────────
    probable_root_causes: List[str]

    # ── Section 4 ─────────────────────────────────────────────────────────────
    severity_assessment: Dict[str, Any]
    overall_severity: SeverityLevel

    # ── Section 5 ─────────────────────────────────────────────────────────────
    recommended_actions: List[str]
    priority_actions: List[str]
    estimated_repair_urgency: str = "Not Available"

    # ── Section 6 ─────────────────────────────────────────────────────────────
    additional_notes: List[str]

    # ── Section 7 ─────────────────────────────────────────────────────────────
    missing_or_unclear_info: List[str]

    # ── Thermal ───────────────────────────────────────────────────────────────
    thermal_stats: Optional[ThermalStats] = None
    thermal_analysis_summary: str = "Not Available"
    thermal_anomalies: List[str] = []

    # ── Conflict detection ────────────────────────────────────────────────────
    conflicts_detected: List[str] = []

    # ── Charts ────────────────────────────────────────────────────────────────
    charts: Optional[ChartData] = None

    # ── Images ────────────────────────────────────────────────────────────────
    embedded_images: List[ExtractedImage] = []
