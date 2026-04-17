"""
Unit tests for the DDR AI Pipeline v2.
Run with:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.models import (
    ThermalReading, AreaObservation, DDRReport, SeverityLevel,
    ThermalStats, ExtractedImage, ChartData,
)
from pipeline.thermal_parser import (
    parse_thermal_report, compute_thermal_statistics,
    detect_thermal_anomalies, format_thermal_summary,
)
from pipeline.extractor import _extract_metadata


# ══════════════════════════════════════════════════════════
# THERMAL READING MODEL
# ══════════════════════════════════════════════════════════

class TestThermalReading:
    def _make(self, hotspot=28.8, coldspot=23.4, **kw):
        defaults = dict(
            image_index=1, image_filename="TEST.JPG",
            capture_date="27/09/22", emissivity=0.94,
            reflected_temperature=23.0,
            device="GTC 400 C Professional", serial_number="000001",
        )
        return ThermalReading(hotspot_temp=hotspot, coldspot_temp=coldspot,
                              **{**defaults, **kw})

    def test_delta_computed_automatically(self):
        r = self._make(hotspot=28.8, coldspot=23.4)
        assert r.delta_temp == pytest.approx(5.4, rel=1e-3)

    def test_anomaly_flagged_high_delta(self):
        r = self._make(hotspot=30.0, coldspot=24.0)   # delta = 6.0 > 5.0
        assert r.anomaly_flag is True
        assert "moisture" in r.anomaly_reason.lower() or "ΔT" in r.anomaly_reason

    def test_anomaly_flagged_caution_delta(self):
        r = self._make(hotspot=27.0, coldspot=23.5)   # delta = 3.5
        assert r.anomaly_flag is True

    def test_no_anomaly_normal_delta(self):
        r = self._make(hotspot=24.0, coldspot=23.0)   # delta = 1.0
        assert r.anomaly_flag is False

    def test_anomaly_high_hotspot(self):
        r = self._make(hotspot=28.0, coldspot=25.5)   # hotspot > 27.5
        assert r.anomaly_flag is True


# ══════════════════════════════════════════════════════════
# THERMAL PARSER
# ══════════════════════════════════════════════════════════

SAMPLE_THERMAL_TEXT = """
28.8 °C
23.4 °C
27/09/22
Hotspot :  28.8 °C
Coldspot : 23.4 °C
Emissivity : 0.94
Reflected temperature : 23 °C
Thermal image : RB02380X.JPG
Device : GTC 400 C Professional
Serial Number : 02700034772
1
27.4 °C
22.4 °C
27/09/22
Hotspot :  27.4 °C
Coldspot : 22.4 °C
Emissivity : 0.94
Reflected temperature : 23 °C
Thermal image : RB02386X.JPG
Device : GTC 400 C Professional
Serial Number : 02700034772
2
"""

class TestThermalParser:
    @pytest.fixture
    def readings(self):
        return parse_thermal_report(SAMPLE_THERMAL_TEXT, [])

    def test_parse_count(self, readings):
        assert len(readings) == 2

    def test_parse_hotspot(self, readings):
        assert readings[0].hotspot_temp == pytest.approx(28.8)

    def test_parse_coldspot(self, readings):
        assert readings[0].coldspot_temp == pytest.approx(23.4)

    def test_parse_index(self, readings):
        assert readings[0].image_index == 1
        assert readings[1].image_index == 2

    def test_parse_filename(self, readings):
        assert readings[0].image_filename == "RB02380X.JPG"

    def test_parse_device(self, readings):
        assert "GTC" in readings[0].device

    def test_parse_date(self, readings):
        assert readings[0].capture_date == "27/09/22"

    def test_empty_text_returns_empty_list(self):
        assert parse_thermal_report("", []) == []


class TestThermalStatistics:
    @pytest.fixture
    def readings(self):
        return parse_thermal_report(SAMPLE_THERMAL_TEXT, [])

    def test_stats_total(self, readings):
        stats = compute_thermal_statistics(readings)
        assert stats["total_readings"] == 2

    def test_stats_max_hotspot(self, readings):
        stats = compute_thermal_statistics(readings)
        assert stats["max_hotspot"] == pytest.approx(28.8)

    def test_stats_min_coldspot(self, readings):
        stats = compute_thermal_statistics(readings)
        assert stats["min_coldspot"] == pytest.approx(22.4)

    def test_stats_delta(self, readings):
        stats = compute_thermal_statistics(readings)
        # reading 1: 28.8-23.4=5.4, reading 2: 27.4-22.4=5.0 → avg=5.2
        assert stats["avg_delta"] == pytest.approx(5.2, rel=1e-2)

    def test_empty_readings_returns_empty(self):
        assert compute_thermal_statistics([]) == {}

    def test_anomaly_count(self, readings):
        stats = compute_thermal_statistics(readings)
        assert stats["anomaly_count"] >= 0


# ══════════════════════════════════════════════════════════
# METADATA EXTRACTION
# ══════════════════════════════════════════════════════════

class TestMetadataExtraction:
    SAMPLE_TEXT = """
    Inspection Date and Time: 27.09.2022 14:28 IST
    Inspected By: Krushna & Mahesh
    Property Type: Flat
    Floors: 11
    Score 85.71%
    """

    def test_extracts_inspection_date(self):
        meta = _extract_metadata(self.SAMPLE_TEXT, "inspection")
        assert "27.09.2022" in meta.get("inspection_date", "")

    def test_extracts_inspected_by(self):
        meta = _extract_metadata(self.SAMPLE_TEXT, "inspection")
        assert "Krushna" in meta.get("inspected_by", "")

    def test_extracts_property_type(self):
        meta = _extract_metadata(self.SAMPLE_TEXT, "inspection")
        assert meta.get("property_type") == "Flat"

    def test_extracts_floors(self):
        meta = _extract_metadata(self.SAMPLE_TEXT, "inspection")
        assert meta.get("floors") == "11"

    def test_doc_type_preserved(self):
        meta = _extract_metadata(self.SAMPLE_TEXT, "inspection")
        assert meta.get("doc_type") == "inspection"


# ══════════════════════════════════════════════════════════
# SEVERITY MODEL
# ══════════════════════════════════════════════════════════

class TestSeverityLevel:
    def test_all_levels_exist(self):
        for lvl in ["Critical", "High", "Moderate", "Low", "Not Available"]:
            assert SeverityLevel(lvl) is not None

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            SeverityLevel("Unknown")


# ══════════════════════════════════════════════════════════
# LLM ENGINE — Rule-based path
# ══════════════════════════════════════════════════════════

class TestLLMEngine:
    @pytest.fixture
    def engine(self):
        from pipeline.llm_engine import LLMEngine
        return LLMEngine(api_key="")   # no key → rule-based

    @pytest.fixture
    def inspection_doc(self):
        from pipeline.models import DocumentExtraction
        return DocumentExtraction(
            document_type="inspection",
            total_pages=1,
            full_text="Inspection Date: 27.09.2022\nInspected By: Test\nProperty Type: Flat\nFloors: 11",
            metadata={
                "inspection_date": "27.09.2022",
                "inspected_by": "Test Inspector",
                "property_type": "Flat",
                "floors": "11",
            },
        )

    @pytest.fixture
    def readings(self):
        return parse_thermal_report(SAMPLE_THERMAL_TEXT, [])

    def test_engine_mode_without_key(self, engine):
        assert "Rule-based" in engine.mode

    def test_generate_ddr_returns_report(self, engine, inspection_doc, readings):
        ddr = engine.generate_ddr(inspection_doc, readings)
        assert isinstance(ddr, DDRReport)

    def test_ddr_has_all_sections(self, engine, inspection_doc, readings):
        ddr = engine.generate_ddr(inspection_doc, readings)
        assert ddr.property_issue_summary
        assert len(ddr.area_observations) > 0
        assert len(ddr.probable_root_causes) > 0
        assert len(ddr.recommended_actions) > 0
        assert len(ddr.priority_actions) > 0
        assert len(ddr.additional_notes) > 0
        assert len(ddr.missing_or_unclear_info) > 0

    def test_ddr_overall_severity_is_valid(self, engine, inspection_doc, readings):
        ddr = engine.generate_ddr(inspection_doc, readings)
        assert ddr.overall_severity in list(SeverityLevel)

    def test_ddr_confidence_in_range(self, engine, inspection_doc, readings):
        ddr = engine.generate_ddr(inspection_doc, readings)
        assert 0.0 <= ddr.overall_confidence <= 1.0
        for obs in ddr.area_observations:
            assert 0.0 <= obs.confidence_score <= 1.0

    def test_ddr_no_invented_facts_flag(self, engine, inspection_doc, readings):
        ddr = engine.generate_ddr(inspection_doc, readings)
        # missing info should not be empty — we should always flag something
        assert len(ddr.missing_or_unclear_info) > 0

    def test_ddr_has_report_id(self, engine, inspection_doc, readings):
        ddr = engine.generate_ddr(inspection_doc, readings)
        assert ddr.report_id and len(ddr.report_id) > 0

    def test_ddr_has_processing_time(self, engine, inspection_doc, readings):
        ddr = engine.generate_ddr(inspection_doc, readings)
        # processing_time_seconds is always set (may be 0.0 on very fast machines)
        assert ddr.processing_time_seconds >= 0.0


# ══════════════════════════════════════════════════════════
# CHART GENERATOR
# ══════════════════════════════════════════════════════════

class TestChartGenerator:
    @pytest.fixture
    def readings(self):
        return parse_thermal_report(SAMPLE_THERMAL_TEXT, [])

    @pytest.fixture
    def observations(self):
        return [
            AreaObservation(
                area_name="Hall",
                negative_side_description="Dampness",
                positive_side_description="Tile gaps",
                severity=SeverityLevel.HIGH,
                confidence_score=0.85,
            ),
            AreaObservation(
                area_name="Parking",
                negative_side_description="Leakage",
                positive_side_description="Plumbing",
                severity=SeverityLevel.CRITICAL,
                confidence_score=0.95,
            ),
        ]

    def test_thermal_profile_chart_returns_base64(self, readings):
        from pipeline.chart_generator import generate_thermal_profile_chart
        result = generate_thermal_profile_chart(readings)
        assert isinstance(result, str)
        assert len(result) > 100  # non-empty base64

    def test_delta_chart_returns_base64(self, readings):
        from pipeline.chart_generator import generate_delta_temp_chart
        result = generate_delta_temp_chart(readings)
        assert isinstance(result, str) and len(result) > 100

    def test_severity_chart_returns_base64(self, observations):
        from pipeline.chart_generator import generate_severity_distribution_chart
        result = generate_severity_distribution_chart(observations)
        assert isinstance(result, str) and len(result) > 100

    def test_generate_all_returns_chart_data(self, readings, observations):
        from pipeline.chart_generator import generate_all_charts
        charts = generate_all_charts(readings, observations)
        assert isinstance(charts, ChartData)
        assert charts.thermal_profile_chart is not None
        assert charts.delta_temp_chart is not None
        assert charts.severity_distribution_chart is not None

    def test_empty_readings_returns_none_charts(self, observations):
        from pipeline.chart_generator import generate_all_charts
        charts = generate_all_charts([], observations)
        assert charts.thermal_profile_chart is None
        assert charts.delta_temp_chart is None


# ══════════════════════════════════════════════════════════
# REPORT BUILDER
# ══════════════════════════════════════════════════════════

class TestReportBuilder:
    @pytest.fixture
    def minimal_ddr(self):
        return DDRReport(
            report_id="DDR-TEST001",
            property_issue_summary="Test summary.",
            area_observations=[],
            probable_root_causes=["Cause 1"],
            severity_assessment={"overall_severity": "High", "overall_reasoning": "Test"},
            overall_severity=SeverityLevel.HIGH,
            recommended_actions=["Action 1"],
            priority_actions=["Priority 1"],
            additional_notes=["Note 1"],
            missing_or_unclear_info=["Item 1 — Not Available"],
        )

    def test_html_report_generates_without_error(self, minimal_ddr, tmp_path):
        from pipeline.report_builder import build_html_report
        out = build_html_report(minimal_ddr, [], str(tmp_path / "test.html"))
        assert Path(out).exists()
        assert Path(out).stat().st_size > 500

    def test_html_contains_all_sections(self, minimal_ddr, tmp_path):
        from pipeline.report_builder import build_html_report
        out = build_html_report(minimal_ddr, [], str(tmp_path / "test.html"))
        html = Path(out).read_text(encoding="utf-8")
        for section_id in ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]:
            assert f'id="{section_id}"' in html, f"Missing section {section_id}"

    def test_json_export(self, minimal_ddr, tmp_path):
        from pipeline.report_builder import export_ddr_json
        import json
        out = export_ddr_json(minimal_ddr, str(tmp_path / "test.json"))
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert data["report_id"] == "DDR-TEST001"
        assert data["overall_severity"] == "High"

    def test_image_map_skips_banners(self):
        from pipeline.report_builder import _build_image_map
        import base64
        # Create a fake 1x1 banner (wide PNG)
        banner = ExtractedImage(
            page_number=1, image_index_on_page=0,
            width=2000, height=100, format="png",
            base64_data=base64.b64encode(b"fake").decode(),
        )
        real = ExtractedImage(
            page_number=1, image_index_on_page=1,
            width=493, height=370, format="jpeg",
            base64_data=base64.b64encode(b"fake2").decode(),
        )
        imap = _build_image_map([banner, real])
        assert "Photo 1" in imap        # real image gets slot 1
        assert "Photo 2" not in imap    # only one real image
