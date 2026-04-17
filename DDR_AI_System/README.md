# 🏗️ DDR AI Report Generation System

> **AI/ML Engineer Assignment** — Advanced 3-Stage AI Pipeline for converting raw property inspection and thermal imaging documents into a professional Detailed Diagnostic Report (DDR).

---

## 🎯 What It Does

Upload an **Inspection Report PDF** + **Thermal Imaging PDF** → Get a complete, professionally formatted DDR with:
- All 7 required sections (Property Issue Summary → Missing Info)
- 4 embedded analytical charts (thermal profile, ΔT trend, severity distribution, confidence scores)
- Site photographs placed under their relevant observation sections
- Per-observation AI confidence scores and reasoning traces
- Conflict detection between inspection and thermal data

---

## 🧠 Architecture

```
[Inspection PDF] ──► PDF Extractor (PyMuPDF) ──► Text + 150 Images
[Thermal PDF]    ──► PDF Extractor ──► Text ──► Thermal Parser ──► 30 ThermalReading objects
                                                      │
                             ┌──────────────────────────────────────────┐
                             │         Chart Generator (matplotlib)      │
                             │  ├── Thermal Profile Chart                │
                             │  ├── ΔT Trend Chart (anomaly bands)       │
                             │  ├── Severity Distribution Donut          │
                             │  └── Confidence Score Bar Chart           │
                             └──────────────────────────────────────────┘
                                                      │
                             ┌──────────────────────────────────────────┐
                             │         LLM Engine v2                     │
                             │  Stage 1: Structured Extraction           │
                             │  Stage 2: Thermal Integration + Analysis  │
                             │  Stage 3: Final DDR Synthesis             │
                             │  Fallback: Advanced Rule-based Engine     │
                             └──────────────────────────────────────────┘
                                                      │
                             ┌──────────────────────────────────────────┐
                             │         Report Builder (Jinja2)           │
                             │  ├── Self-contained HTML (1MB+)           │
                             │  └── Structured JSON export               │
                             └──────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option A — CLI (no setup required)
```bash
cd DDR_AI_System
pip install -r requirements.txt

# Run on the provided sample documents (auto-detected)
python generate_report.py

# With Gemini AI for enhanced analysis
python generate_report.py --api-key YOUR_GEMINI_KEY

# Custom document paths
python generate_report.py \
  --inspection path/to/inspection.pdf \
  --thermal    path/to/thermal.pdf \
  --output     ./output
```
📄 **Output:** `output/DDR_Report.html` (open in any browser — fully self-contained)

### Option B — Streamlit Web App
```bash
python -m streamlit run app.py
# → http://localhost:8501
```

### Option C — Python API
```python
from pipeline.extractor      import extract_document
from pipeline.thermal_parser import parse_thermal_report
from pipeline.llm_engine     import LLMEngine
from pipeline.chart_generator import generate_all_charts
from pipeline.report_builder import build_html_report
from pipeline.models          import ThermalStats

insp     = extract_document("inspection.pdf", "inspection")
therm    = extract_document("thermal.pdf",    "thermal")
readings = parse_thermal_report(therm.full_text, therm.images)

engine = LLMEngine(api_key="YOUR_KEY")   # or "" for offline mode
ddr    = engine.generate_ddr(insp, readings)
ddr.charts = generate_all_charts(readings, ddr.area_observations)

build_html_report(ddr, insp.images + therm.images, "DDR_Report.html")
```

---

## 📋 DDR Output Structure

| # | Section | Contents |
|---|---------|----------|
| 1 | **Property Issue Summary** | Executive 3-sentence overview |
| 2 | **Area-wise Observations** | 7 areas with neg/pos sides, photos, confidence, AI reasoning trace |
| 3 | **Probable Root Causes** | 6 evidenced, ranked causes |
| 4 | **Severity Assessment** | Per-area severity table with confidence bars + donut chart |
| 5 | **Recommended Actions** | Priority (immediate) + tiered full plan (11 actions) |
| 6 | **Additional Notes** | Conflicts, cross-flat issues, thermal methodology |
| 7 | **Missing/Unclear Info** | Explicit "Not Available" flags for every gap |

---

## ⚙️ Advanced Features

| Feature | Detail |
|---------|--------|
| **3-Stage LLM Chain** | Extract → Analyze+Merge → Synthesize (Gemini 1.5 Pro) |
| **Offline Mode** | Advanced rule-based engine — no API key needed |
| **Confidence Scoring** | 0–1 per observation, tracked from evidence sources |
| **Chain-of-Thought** | AI reasoning trace embedded per section |
| **Anomaly Detection** | ΔT > 3°C caution / > 5°C high-risk thresholds |
| **Conflict Detection** | Flags inspection vs thermal data mismatches |
| **4 Analytical Charts** | All generated with matplotlib, embedded as PNG |
| **Image Extraction** | 150 photos extracted, 37 placed in relevant sections |
| **Self-contained** | Single ~1MB HTML with zero external dependencies |
| **Unit Tests** | 35+ tests across all modules (pytest) |
| **Structured Logging** | Coloured console + file logging |
| **Generalizable** | Works on any similar inspection + thermal PDF format |

---

## 🧪 Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected: **35 tests, all passing** (~15s)

---

## 📁 Project Structure

```
DDR_AI_System/
├── app.py                    Streamlit web application
├── generate_report.py        CLI pipeline runner
├── requirements.txt          Python dependencies
├── .env.example              Environment config template
├── README.md
│
├── pipeline/
│   ├── models.py             Pydantic data models (DDRReport, ThermalReading, …)
│   ├── extractor.py          PDF text + image extraction (PyMuPDF)
│   ├── thermal_parser.py     Thermal readings parser + anomaly detection
│   ├── llm_engine.py         3-stage Gemini chain + rule-based fallback
│   ├── chart_generator.py    4 matplotlib analytical charts
│   ├── report_builder.py     Jinja2 HTML renderer + JSON exporter
│   └── logger.py             Structured coloured logging
│
├── templates/
│   └── ddr_template.html     Premium dark-theme report template
│
├── tests/
│   └── test_pipeline.py      35+ unit tests (pytest)
│
└── output/
    ├── DDR_Report.html        ← Open this
    ├── DDR_Report.json        ← Machine-readable
    └── pipeline.log           ← Execution log
```

---

## 🔑 Configuration

```bash
# Copy and edit
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (empty) | Gemini API key — system works offline without it |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Model to use for 3-stage chain |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 📊 Sample Results (Provided Documents)

```
Documents:       Sample Report.pdf (23 pages, 150 images)
                 Thermal Images.pdf (30 readings, GTC 400 C Professional)

Pipeline mode:   Advanced Rule-based Engine (offline)
Processing time: ~9.8 seconds

Extraction:      150 site photos extracted
Thermal:         30/30 readings parsed, all 30 anomalous (avg ΔT = 5.01°C)
Areas analysed:  7 (Hall, 2 Bedrooms, Kitchen, External Wall, Parking, Bathroom)
Overall severity: CRITICAL (active slab leakage in parking area)
AI confidence:   86%
Actions issued:  11 (3 immediate, 4 short-term, 2 medium, 2 long-term)
Report size:     1,032 KB self-contained HTML
```

---

## 📝 Important Rules Followed

- ✅ **No invented facts** — all findings based strictly on source documents
- ✅ **Conflicts flagged** — thermal vs inspection mismatches explicitly noted
- ✅ **Missing data** — 7 "Not Available" entries explicitly listed
- ✅ **Client-friendly language** — no unnecessary technical jargon
- ✅ **Generalizable** — works on any similar inspection + thermal PDF format
- ✅ **Images in context** — photos placed under their relevant area section

---

## 🛠️ Tech Stack

`Python 3.11` · `PyMuPDF` · `Google Gemini 1.5 Pro` · `Pydantic v2` · `Jinja2` · `matplotlib` · `Streamlit` · `pytest`
