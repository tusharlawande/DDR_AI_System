from pathlib import Path

# Use path relative to script to find output/DDR_Report.html
SCRIPT_DIR = Path(__file__).parent
report_path = SCRIPT_DIR / "output" / "DDR_Report.html"

if not report_path.exists():
    print(f"Error: Report not found at {report_path}")
    print("Please run generate_report.py first.")
    exit(1)

html = report_path.read_text(encoding='utf-8')
print("=== DDR Report v2 Quality Check ===")
items = [
    ("Cover page",          "cover-title"),
    ("KPI strip",           "kpi-strip"),
    ("Confidence bars",     "conf-bar-fill"),
    ("Evidence chips",      "evidence-chip"),
    ("Chain of thought",    "AI Reasoning Trace"),
    ("Section 1",           'id="s1"'),
    ("Section 2",           'id="s2"'),
    ("Section 3",           'id="s3"'),
    ("Section 4",           'id="s4"'),
    ("Section 5",           'id="s5"'),
    ("Section 6",           'id="s6"'),
    ("Section 7",           'id="s7"'),
    ("Thermal stats",       "thermal-stat-val"),
    ("Priority actions",    "priority-card"),
    ("Conflict detection",  "conflict-box"),
    ("Missing info",        "na-tag"),
    ("Severity table",      "sev-table"),
    ("Footer",              "report-footer"),
]
all_pass = True
for label, token in items:
    ok = token in html
    if not ok: all_pass = False
    print(f"  {'OK' if ok else 'FAIL'} {label}")

charts_count = html.count("data:image/png;base64,")
jpeg_count   = html.count("data:image/jpeg;base64,")
html_kb      = len(html) // 1024

print()
print(f"  Charts embedded  : {charts_count} PNG charts")
print(f"  Site photos      : {jpeg_count} JPEG images")
print(f"  Report size      : {html_kb} KB")
print(f"  Overall status   : {'ALL PASS' if all_pass else 'SOME FAILURES'}")
