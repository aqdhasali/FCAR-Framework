"""
FCAR Interactive Dashboard — Streamlit UI (Premium Edition).

Pages:
  1. Interactive Recourse (UC-101 / UC-105)
  2. Fairness Audit (UC-103)
  3. Evaluation Metrics
  4. Methodology & About
"""

import sys
from pathlib import Path
import json

import streamlit as st
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from src.config.config_loader import (
    load_dataset_config,
    get_mutable_numeric_cols,
    get_mutable_categorical_cols,
    get_numeric_cost_weights,
    get_categorical_step_weights,
    get_sensitive_attributes,
)
from src.recourse.generic_recourse_mip import solve_recourse_mip

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="FCAR — Fairness Constrained Actionable Recourse",
    page_icon="\u2696",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* ─── Force Poppins everywhere ─── */
*, html, body,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stHeader"],
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown td, .stMarkdown th,
button, input, select, textarea,
h1, h2, h3, h4, h5, h6,
label, .stRadio label, .stSelectbox label,
div, span, a, p, td, th {
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
/* Restore Material Symbols / Icons font for Streamlit's built-in icon spans */
[data-testid="stExpander"] svg,
[data-testid="stExpander"] span[class*="icon"],
[data-testid="stExpander"] span[data-testid],
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
.st-emotion-cache-p5msec,
[data-testid="stExpander"] details summary svg {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}
/* Hide any text-based icon fallback in expanders */
[data-testid="stExpanderToggleIcon"] {
    font-size: 0px !important;
    overflow: hidden !important;
    width: 24px !important;
    height: 24px !important;
}
[data-testid="stExpanderToggleIcon"] svg {
    font-size: initial !important;
    width: 24px !important;
    height: 24px !important;
}

/* ─── Color Palette ─── */
:root {
    --navy: #0c4466;
    --navy-light: #0e5a85;
    --teal: #14b8a6;
    --teal-light: #5eead4;
    --bg: #ffffff;
    --bg-alt: #f8fafc;
    --border: #e2e8f0;
    --text: #0c4466;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --green: #059669;
    --red: #dc2626;
    --amber: #d97706;
    --radius: 10px;
}

/* ─── Background ─── */
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ─── Sidebar always visible — hide collapse/expand controls ─── */
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebar"] button[kind="headerNoPadding"],
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
[data-testid="stSidebar"] {
    transform: none !important;
    width: 310px !important;
    min-width: 310px !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
}
/* Center sidebar logo */
[data-testid="stSidebar"] [data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
}

/* ─── Sidebar nav radio ─── */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 16px !important;
    border-radius: 8px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background: var(--bg-alt) !important;
}
/* Hide radio circle */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
/* Selected state */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(12, 68, 102, 0.08) !important;
    color: var(--navy) !important;
    font-weight: 600 !important;
}
/* Hide radio label header */
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}

/* ─── Headers ─── */
h1 { color: var(--navy) !important; font-weight: 700 !important; }
h2 { color: var(--navy) !important; font-weight: 600 !important; }
h3 { color: var(--navy) !important; font-weight: 600 !important; }

/* ─── Buttons ─── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--navy) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 24px !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--navy-light) !important;
    color: #ffffff !important;
}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: var(--navy) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--navy) !important;
    background: rgba(12, 68, 102, 0.04) !important;
}

/* ─── Selectbox / Inputs ─── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ─── Toggle ─── */
[data-testid="stSwitchWidget"] span[data-checked="true"] {
    background-color: var(--navy) !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 2px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--navy) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--navy) !important;
    background: transparent !important;
}

/* ─── DataFrame ─── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ─── Links ─── */
.stMarkdown a { color: var(--navy) !important; }

/* ─── Metrics ─── */
[data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-weight: 700 !important;
}

/* ─── Glass Card ─── */
.glass-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
}

/* ─── KPI Card ─── */
.kpi-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(135deg, var(--navy), var(--teal));
}
.kpi-card.green::before  { background: var(--green); }
.kpi-card.red::before    { background: var(--red); }
.kpi-card.orange::before { background: var(--amber); }
.kpi-card.purple::before { background: #7c3aed; }
.kpi-card.cyan::before   { background: var(--teal); }
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--navy);
    line-height: 1.1;
}
.kpi-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 6px;
}

/* ─── Badges ─── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
}
.badge-pass {
    background: #ecfdf5; color: var(--green); border: 1px solid #a7f3d0;
}
.badge-fail {
    background: #fef2f2; color: var(--red); border: 1px solid #fecaca;
}
.badge-fcar {
    background: #e0f2fe; color: var(--navy); border: 1px solid #7dd3fc;
}
.badge-ar {
    background: #fffbeb; color: var(--amber); border: 1px solid #fde68a;
}
.badge-info {
    background: #f0fdfa; color: #0d9488; border: 1px solid #99f6e4;
}

/* ─── Section Header ─── */
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: var(--navy);
    margin: 32px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header .sec-icon { display:inline-flex; width:22px; height:22px; flex-shrink:0; }
.sec-icon svg { width:100%; height:100%; }
.mi-svg { display:inline-flex; vertical-align:middle; width:20px; height:20px; flex-shrink:0; align-items:center; justify-content:center; }
.mi-svg svg { width:100%; height:100%; }

/* ─── Score Ring ─── */
.score-display { text-align: center; padding: 32px 16px; }
.score-ring {
    position: relative;
    width: 140px; height: 140px;
    margin: 0 auto 16px;
}
.score-ring svg { transform: rotate(-90deg); }
.score-ring .bg { stroke: #e2e8f0; }
.score-ring .fg { transition: stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1); }
.score-ring .value {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 28px;
    font-weight: 700;
}
.score-ring .value.denied  { color: var(--red); }
.score-ring .value.approved { color: var(--green); }

/* ─── Change Pills ─── */
.change-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
.change-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
}
.change-pill.up {
    background: #ecfdf5; border: 1px solid #a7f3d0; color: var(--green);
}
.change-pill.down {
    background: #fef2f2; border: 1px solid #fecaca; color: var(--red);
}
.change-pill.swap {
    background: #fffbeb; border: 1px solid #fde68a; color: var(--amber);
}
.change-pill .feat { font-weight: 700; }

/* ─── Narrative Box ─── */
.narrative-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid var(--navy);
    border-radius: var(--radius);
    padding: 20px 24px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text);
    margin: 20px 0;
}
.narrative-box b { color: var(--navy); }

/* ─── Audit Card ─── */
.audit-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    position: relative;
    overflow: hidden;
}
.audit-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
}
.audit-card.pass::before { background: var(--green); }
.audit-card.fail::before { background: var(--red); }
.audit-card .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.audit-card table { width: 100%; border-collapse: collapse; }
.audit-card table td {
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
}
.audit-card table td:first-child { color: var(--text-secondary); font-weight: 500; }
.audit-card table td:last-child {
    text-align: right; font-weight: 700; color: var(--navy);
    font-variant-numeric: tabular-nums;
}
.audit-card table tr:last-child td { border-bottom: none; }

/* ─── Status Banner ─── */
.status-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    margin: 12px 0;
}
.status-banner.fcar {
    background: #e0f2fe;
    border: 1px solid #7dd3fc;
    color: var(--navy);
}
.status-banner.ar {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: var(--amber);
}

/* ─── Profile Table ─── */
.profile-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    font-size: 13px;
}
.profile-table th {
    background: var(--bg-alt);
    color: var(--text-secondary);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 11px;
    padding: 12px 16px;
    text-align: left;
}
.profile-table td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.profile-table tr:last-child td { border-bottom: none; }
.profile-table tr:hover td { background: var(--bg-alt); }
.feat-mutable { color: var(--teal); font-weight: 600; }
.feat-immutable { color: var(--text-muted); }

/* ─── About Hero ─── */
.about-hero {
    background: linear-gradient(135deg, #f0f9ff 0%, #f0fdfa 100%);
    border: 1px solid #bae6fd;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
}
.about-hero h2 {
    font-size: 32px;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 12px;
    -webkit-text-fill-color: unset;
    background: none;
}
.about-hero p {
    color: var(--text-secondary);
    font-size: 16px;
    line-height: 1.7;
    max-width: 680px;
}

/* ─── Sparkline Bars ─── */
.spark-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; }
.spark-label { font-size: 12px; color: var(--text-secondary); width: 80px; flex-shrink: 0; text-align: right; }
.spark-track {
    flex: 1; height: 8px; background: #f1f5f9;
    border-radius: 4px; overflow: hidden;
}
.spark-fill { height: 100%; border-radius: 4px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }
.spark-fill.indigo  { background: linear-gradient(90deg, var(--navy), var(--teal)); }
.spark-fill.amber   { background: linear-gradient(90deg, #d97706, #f59e0b); }
.spark-fill.emerald { background: linear-gradient(90deg, #059669, #10b981); }
.spark-fill.rose    { background: linear-gradient(90deg, #dc2626, #f43f5e); }
.spark-val {
    font-size: 11px; font-weight: 700; color: var(--navy);
    width: 55px; text-align: right; font-variant-numeric: tabular-nums;
}

/* ─── Comparison Arrow ─── */
.comparison-arrow {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 20px 0;
}
.comparison-arrow .arrow { font-size: 32px; color: var(--teal); }
.comparison-arrow .label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--teal); margin-top: 4px;
}

/* ─── Flow Steps ─── */
.flow-container { display: flex; align-items: center; gap: 0; flex-wrap: wrap; margin: 16px 0; }
.flow-step {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    text-align: center;
    flex: 1; min-width: 130px;
}
.flow-step .num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px;
    background: var(--navy); border-radius: 50%;
    font-size: 12px; font-weight: 700; color: white; margin-bottom: 8px;
}
.flow-step .title { font-size: 12px; font-weight: 700; color: var(--navy); margin-bottom: 4px; }
.flow-step .desc { font-size: 10px; color: var(--text-muted); line-height: 1.4; }
.flow-arrow { font-size: 18px; color: var(--text-muted); padding: 0 4px; flex-shrink: 0; }

/* ─── Highlight Card ─── */
.highlight-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    position: relative; overflow: hidden;
}
.highlight-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
}
.highlight-card.best::after { background: var(--green); }
.highlight-card.sig::after  { background: var(--navy); }
.highlight-card .big { font-size: 32px; font-weight: 700; line-height: 1; }
.highlight-card .big.emerald { color: var(--green); }
.highlight-card .big.indigo  { color: var(--navy); }
.highlight-card .big.amber   { color: var(--amber); }
.highlight-card .lbl {
    font-size: 11px; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px; margin-top: 8px;
}

/* ─── Before/After Table ─── */
.ba-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    border-radius: 8px; overflow: hidden;
    border: 1px solid var(--border); font-size: 13px; margin: 12px 0;
}
.ba-table th {
    background: var(--bg-alt); color: var(--text-secondary);
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;
    font-size: 11px; padding: 12px 14px; text-align: left;
}
.ba-table td {
    padding: 10px 14px; border-bottom: 1px solid var(--border); color: var(--text);
}
.ba-table tr:last-child td { border-bottom: none; }
.ba-table tr:hover td { background: var(--bg-alt); }
.ba-table .delta-up   { color: var(--green); font-weight: 700; }
.ba-table .delta-down { color: var(--red); font-weight: 700; }
.ba-table .delta-swap { color: var(--amber); font-weight: 700; }

/* ─── Success Banner ─── */
.success-banner {
    background: #ecfdf5; border: 1px solid #a7f3d0;
    border-radius: var(--radius); padding: 20px 28px;
    display: flex; align-items: center; gap: 16px; margin: 16px 0;
}
.success-banner .icon { font-size: 28px; }
.success-banner .text { font-size: 15px; font-weight: 600; color: var(--green); }
.success-banner .sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

/* ─── Page Title ─── */
.page-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 28px;
    line-height: 1.6;
}

/* ─── Divider ─── */
.premium-divider {
    height: 1px;
    background: var(--border);
    margin: 28px 0;
    border: none;
}

/* ─── Footer ─── */
.app-footer {
    margin-top: 48px; padding: 20px 0;
    border-top: 1px solid var(--border);
    text-align: center; font-size: 11px; color: var(--text-muted);
}
.app-footer a { color: var(--navy) !important; text-decoration: none; }
.app-footer .sep { margin: 0 8px; opacity: 0.3; }

/* ─── Hide decoration ─── */
[data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cached Loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def load_model_and_data(dataset_name: str):
    model_path = ART / "models" / f"{dataset_name}_logreg.joblib"
    if not model_path.exists():
        return None, None, None, None, None
    pipe = joblib.load(model_path)
    config = load_dataset_config(dataset_name)
    X = pd.read_csv(PROC / dataset_name / "X.csv")
    A = pd.read_csv(PROC / dataset_name / "A.csv")
    test_idx = np.load(SPLITS / dataset_name / "test_idx.npy")
    train_idx = np.load(SPLITS / dataset_name / "train_idx.npy")
    A_test = A.iloc[test_idx].reset_index(drop=True)
    if "age" in A_test.columns:
        A_test["age_bucket"] = pd.cut(
            A_test["age"], bins=[0, 25, 40, 60, 120],
            labels=["<=25", "26-40", "41-60", "60+"], include_lowest=True,
        ).astype(str)
    if "AGE" in A_test.columns:
        A_test["AGE_bucket"] = pd.cut(
            A_test["AGE"], bins=[0, 25, 40, 60, 120],
            labels=["<=25", "26-40", "41-60", "60+"], include_lowest=True,
        ).astype(str)
    return (
        pipe, config,
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        A_test,
    )


@st.cache_data
def load_eval_summaries():
    bench_dir = ART / "reports" / "benchmarks"
    summaries = {}
    if bench_dir.exists():
        for f in bench_dir.glob("*_ab_summary.json"):
            with open(f) as fh:
                summaries[f.name] = json.load(fh)
    return summaries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _is_rejected(prob, target_cls):
    return (prob < 0.5) if target_cls == 1 else (prob >= 0.5)


def _kpi(label, value, color="", sub=""):
    cls = f"kpi-card {color}" if color else "kpi-card"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


_TEAL = "#14b8a6"

_IC = {
    "person":        f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>',
    "gavel":         f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M1 21h12v2H1v-2zm8.75-7.75L3 20l1.41 1.41 6.75-6.75-1.41-1.41zM22 5.72l-4.6-3.86-1.29 1.53 4.6 3.86L22 5.72zM7.88 8.3l2.95 2.95 1.41-1.41-2.95-2.95L7.88 8.3zm6.59-2.12l-1.41-1.41-3.54 3.54 1.41 1.41 3.54-3.54zm1.57 4.45l1.41-1.41-5.66-5.66-1.41 1.41 5.66 5.66z"/></svg>',
    "trending_up":   f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>',
    "build":         f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z"/></svg>',
    "lock":          '<svg viewBox="0 0 24 24" fill="#94a3b8"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z"/></svg>',
    "compare_arrows":f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M9.01 14H2v2h7.01v3L13 15l-3.99-4v3zm5.98-1v-3H22V8h-7.01V5L11 9l3.99 4z"/></svg>',
    "bar_chart":     f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M5 9.2h3V19H5V9.2zM10.6 5h2.8v14h-2.8V5zm5.6 8H19v6h-2.8v-6z"/></svg>',
    "search":        f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>',
    "science":       f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M13 11.33L18 18H6l5-6.67V6h2v5.33M15.96 4H8.04C7.62 4 7.28 4.34 7.28 4.76c0 .42.34.76.76.76H9v4.67L3.2 18.4c-.49.66-.02 1.6.8 1.6h16c.82 0 1.29-.94.8-1.6L15 10.43V5.52c.42 0 .76-.34.76-.76 0-.42-.34-.76-.8-.76z"/></svg>',
    "summarize":     f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 9H7v-2h6v2zm3 4H7v-2h9v2zm-3-8V3.5L18.5 9H13z"/></svg>',
    "target":        f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M12 2C6.49 2 2 6.49 2 12s4.49 10 10 10 10-4.49 10-10S17.51 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3-8c0 1.66-1.34 3-3 3s-3-1.34-3-3 1.34-3 3-3 3 1.34 3 3z"/></svg>',
    "sync":          f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"/></svg>',
    "architecture":  f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M6.36 18.78L6.61 21l1.62-1.54 2.77-7.6c-.68-.17-1.28-.51-1.77-.98l-3.27 8.9zM14.77 10.88c-.49.47-1.1.81-1.77.98l2.77 7.6L17.39 21l.25-2.22-3.27-8.9zm3.09-5.73L15.33 2H8.67L6.14 5.15l1.86.93 1.67-2.08h4.66l1.67 2.08 1.86-.93zM12 7c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>',
    "menu_book":     f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M21 5c-1.11-.35-2.33-.5-3.5-.5-1.95 0-4.05.4-5.5 1.5-1.45-1.1-3.55-1.5-5.5-1.5S2.45 4.9 1 6v14.65c0 .25.25.5.5.5.1 0 .15-.05.25-.05C3.1 20.45 5.05 20 6.5 20c1.95 0 4.05.4 5.5 1.5 1.35-.85 3.8-1.5 5.5-1.5 1.65 0 3.35.3 4.75 1.05.1.05.15.05.25.05.25 0 .5-.25.5-.5V6c-.6-.45-1.25-.75-2-1zm0 13.5c-1.1-.35-2.3-.5-3.5-.5-1.7 0-4.15.65-5.5 1.5V8c1.35-.85 3.8-1.5 5.5-1.5 1.2 0 2.4.15 3.5.5v11.5z"/></svg>',
    "auto_stories":  f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M19 1l-5 5v11l5-4.5V1zM1 6v14.65c0 .25.25.5.5.5.1 0 .15-.05.25-.05C3.1 20.45 5.05 20 6.5 20c1.95 0 4.05.4 5.5 1.5V8c-1.45-1.1-3.55-1.5-5.5-1.5S2.45 4.9 1 6zm20-1c-.7.25-1.4.55-2 1v12.5c.6.45 1.25.75 2 1V5z"/></svg>',
    "link":          f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z"/></svg>',
    "balance":       f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M12 3L1 9l4 2.18v6L12 21l7-3.82v-6l2-1.09V17h2V9L12 3zm6.82 6L12 12.72 5.18 9 12 5.28 18.82 9zM17 15.99l-5 2.73-5-2.73v-3.72L12 15l5-2.73v3.72z"/></svg>',
    "straighten":    f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M21 6H3c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 10H3V8h2v4h2V8h2v4h2V8h2v4h2V8h2v4h2V8h2v4z"/></svg>',
    "check_circle":  '<svg viewBox="0 0 24 24" fill="#10b981"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>',
    "warning":       '<svg viewBox="0 0 24 24" fill="#fbbf24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>',
    "info":          '<svg viewBox="0 0 24 24" fill="#f59e0b"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>',
    "arrow_forward": f'<svg viewBox="0 0 24 24" fill="{_TEAL}"><path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"/></svg>',
}

def _icon(name, size=20):
    """Return an inline SVG icon. Always renders — no external font needed."""
    svg = _IC.get(name, "")
    return f'<span class="mi-svg" style="width:{size}px;height:{size}px;">{svg}</span>'


def _section(icon, title):
    st.markdown(
        f'<div class="section-header">'
        f'<span class="sec-icon">{_IC.get(icon, "")}</span> {title}</div>',
        unsafe_allow_html=True,
    )


# ── Human-readable labels for coded dataset values ──────────────────
GERMAN_VALUE_LABELS = {
    # Checking account status
    "A11": "Overdrawn (< 0 DM)",
    "A12": "Balance 0 \u2013 200 DM",
    "A13": "Balance \u2265 200 DM",
    "A14": "No Checking Account",
    # Savings account status
    "A61": "Savings < 100 DM",
    "A62": "Savings 100 \u2013 500 DM",
    "A63": "Savings 500 \u2013 1,000 DM",
    "A64": "Savings \u2265 1,000 DM",
    "A65": "No Savings Account",
}

GERMAN_FEATURE_LABELS = {
    "checking_status": "Checking Account Status",
    "savings_status": "Savings Account Status",
    "duration": "Loan Duration (months)",
    "credit_amount": "Loan Amount (DM)",
    "installment_commitment": "Installment Rate (% of income)",
    "existing_credits": "Existing Credits at Bank",
    "residence_since": "Years at Residence",
}


def _human_value(code: str) -> str:
    """Translate a coded value like 'A12' to 'A12 (Balance 0\u2013200 DM)'."""
    label = GERMAN_VALUE_LABELS.get(code)
    return f"{code} ({label})" if label else code


def _human_feature(name: str) -> str:
    """Translate a raw feature name like 'checking_status' to a readable label."""
    return GERMAN_FEATURE_LABELS.get(name, name.replace("_", " ").title())


def _divider():
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)


def _score_ring(score, is_denied=True):
    """Render a circular score gauge."""
    pct = min(score * 100, 100)
    circum = 2 * 3.14159 * 54
    offset = circum * (1 - pct / 100)
    color = "#f43f5e" if is_denied else "#10b981"
    css_class = "denied" if is_denied else "approved"
    status = "DENIED" if is_denied else "APPROVED"
    badge_cls = "badge-fail" if is_denied else "badge-pass"
    return (
        '<div class="score-display">'
        '<div class="score-ring">'
        '<svg width="140" height="140" viewBox="0 0 120 120">'
        f'<circle class="bg" cx="60" cy="60" r="54" fill="none" stroke-width="8"/>'
        f'<circle class="fg" cx="60" cy="60" r="54" fill="none" '
        f'stroke="{color}" stroke-width="8" stroke-linecap="round" '
        f'stroke-dasharray="{circum}" stroke-dashoffset="{offset}"/>'
        '</svg>'
        f'<div class="value {css_class}">{score:.4f}</div>'
        '</div>'
        f'<div style="margin-bottom:6px;"><span class="badge {badge_cls}">{status}</span></div>'
        '<div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Approval Score</div>'
        '</div>'
    )


def _spark_bar(label, value, max_val, color="indigo"):
    """Horizontal sparkline bar."""
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return (
        '<div class="spark-row">'
        f'<div class="spark-label">{label}</div>'
        f'<div class="spark-track"><div class="spark-fill {color}" style="width:{pct:.1f}%"></div></div>'
        f'<div class="spark-val">{value:.4f}</div>'
        '</div>'
    )


def _footer():
    """Render the app footer."""
    st.markdown(
        '<div class="app-footer">'
        'FCAR Framework v1.0'
        '<span class="sep">|</span>'
        'Built with Streamlit + Pyomo + HiGHS'
        '<span class="sep">|</span>'
        '<a href="https://github.com/aqdhasali/FCAR-Framework">GitHub</a>'
        '<span class="sep">|</span>'
        '\u00a9 2026 Aqdhas Ali \u2014 IIT / University of Westminster'
        '</div>',
        unsafe_allow_html=True,
    )


def _methodology_flow():
    """Render the FCAR pipeline methodology flowchart."""
    steps = [
        ("1", "Data Prep", "Load & encode features"),
        ("2", "Train Model", "LogReg + ColumnTransformer"),
        ("3", "Compute Burden", "MISOB per group"),
        ("4", "Auto-Tune", "Asymmetric weight adj."),
        ("5", "Solve MIP", "Pyomo + HiGHS"),
        ("6", "Audit", "Gap, ratio, p-value"),
    ]
    html = '<div class="flow-container">'
    for i, (num, title, desc) in enumerate(steps):
        if i > 0:
            html += '<div class="flow-arrow">\u2192</div>'
        html += (
            '<div class="flow-step">'
            f'<div class="num">{num}</div>'
            f'<div class="title">{title}</div>'
            f'<div class="desc">{desc}</div>'
            '</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _before_after_table(changes):
    """Render a styled HTML before/after comparison table."""
    rows = ""
    for c in changes:
        d = c["_dir"]
        delta_cls = {"up": "delta-up", "down": "delta-down", "swap": "delta-swap"}[d]
        arrow = {"up": "\u2191", "down": "\u2193", "swap": "\u2194"}[d]
        delta_key = "\u0394 Amount"
        delta_val = c[delta_key]
        rows += (
            f'<tr>'
            f'<td style="font-weight:600;">{c["Feature"]}</td>'
            f'<td>{c["Before"]}</td>'
            f'<td>{c["After"]}</td>'
            f'<td class="{delta_cls}">{arrow} {delta_val}</td>'
            f'<td style="color:var(--text-muted);">{c["Cost Weight"]}</td>'
            f'</tr>'
        )
    st.markdown(
        '<table class="ba-table">'
        '<thead><tr><th>Feature</th><th>Before</th><th>After</th>'
        '<th>\u0394 Change</th><th>Weight</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── Sidebar: Logo ──
    logo_path = ROOT / "assets" / "logo.png"
    if logo_path.exists():
        try:
            st.sidebar.image(str(logo_path), width=180)
        except Exception:
            # Fallback if image is invalid
            st.sidebar.markdown(
                '<div style="text-align:center;padding:20px 0 8px;">'
                '<div style="font-size:24px;font-weight:700;color:#0c4466;">FCAR</div>'
                '<div style="font-size:10px;color:#94a3b8;letter-spacing:1.2px;'
                'text-transform:uppercase;">Fairness Constrained<br/>Actionable Recourse</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.markdown(
            '<div style="text-align:center;padding:20px 0 8px;">'
            '<div style="font-size:24px;font-weight:700;color:#0c4466;">FCAR</div>'
            '<div style="font-size:10px;color:#94a3b8;letter-spacing:1.2px;'
            'text-transform:uppercase;">Fairness Constrained<br/>Actionable Recourse</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")

    # ── Sidebar: Navigation ──
    page = st.sidebar.radio(
        "Navigation",
        [
            "Interactive Recourse",
            "Fairness Audit",
            "Evaluation Metrics",
            "About & Methodology",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # ── Sidebar: Dataset ──
    st.sidebar.markdown(
        '<div style="font-size:12px;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Dataset</div>',
        unsafe_allow_html=True,
    )
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["german", "adult", "default_credit"],
        format_func=lambda x: {
            "german": "German Credit",
            "adult": "Adult Income",
            "default_credit": "Default Credit",
        }[x],
        label_visibility="collapsed",
    )

    pipe, config, X_train, X_test, A_test = load_model_and_data(dataset)
    if pipe is None:
        st.error(f"Models for **{dataset}** not found. Train first: `python scripts/train_baseline.py`")
        st.stop()

    target_cls = int(config.get("label_positive", 1))
    proba = pipe.predict_proba(X_test)[:, 1]
    rejected_mask = np.array([_is_rejected(p, target_cls) for p in proba])
    rejected_indices = np.where(rejected_mask)[0]
    rej_pct = len(rejected_indices) / len(X_test) if len(X_test) > 0 else 0

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:10px;padding:16px;text-align:center;">'
        '<div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;'
        'color:#94a3b8;margin-bottom:8px;font-weight:600;">Test Set Stats</div>'
        f'<div style="font-size:28px;font-weight:700;color:#0c4466;">{len(rejected_indices)}</div>'
        f'<div style="font-size:12px;color:#94a3b8;">rejected of {len(X_test)} ({rej_pct:.0%})</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="text-align:center;font-size:10px;color:#94a3b8;padding:8px;">'
        'Aqdhas Ali \u00b7 IIT / UoW \u00b7 2026</div>',
        unsafe_allow_html=True,
    )

    # ── Page Router ──
    if "Interactive" in page:
        page_recourse(dataset, pipe, config, X_train, X_test, A_test,
                      rejected_indices, proba, target_cls)
    elif "Audit" in page:
        page_audit(dataset)
    elif "Evaluation" in page:
        page_evaluation(pipe, config)
    elif "About" in page:
        page_about()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1 — Interactive Recourse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_recourse(dataset, pipe, config, X_train, X_test, A_test,
                  rejected_indices, proba, target_cls):
    ds_label = dataset.replace("_", " ").title()

    st.markdown(
        f'<div class="page-title">'
        f'Individual Recourse \u2014 {ds_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Select a denied applicant and generate '
        'an optimal action plan to overturn the decision. Toggle FCAR to '
        'compare fairness-constrained vs unconstrained recommendations.</div>',
        unsafe_allow_html=True,
    )

    if len(rejected_indices) == 0:
        st.warning("No rejected applicants in the test set.")
        return

    # ── Discover available FCAR benchmarks ──
    bench_dir = ART / "reports" / "benchmarks"
    available_gcols = []
    if bench_dir.exists():
        for f in bench_dir.glob(f"{dataset}_*_ab_summary.json"):
            with open(f) as fh:
                s = json.load(fh)
            available_gcols.append(s["group_col"])
    available_gcols = sorted(set(available_gcols))

    # ── Controls ──
    c1, c2, c3 = st.columns([1.2, 1, 1.5])
    with c1:
        selected_idx = int(
            st.selectbox(
                "Select Applicant",
                options=rejected_indices,
                format_func=lambda x: f"Applicant #{x}",
            )
        )
    with c2:
        use_fcar = st.toggle(
            "Enable FCAR",
            value=False,
            help="Apply fairness-adjusted cost weights from Auto-FCAR tuning.",
        )
    with c3:
        if use_fcar and available_gcols:
            selected_gcol = st.selectbox(
                "FCAR Group Attribute",
                options=available_gcols,
                format_func=lambda x: x.replace("_", " ").title(),
                help="Sensitive attribute for FCAR weight adjustment",
            )
        else:
            selected_gcol = available_gcols[0] if available_gcols else None
            if use_fcar:
                st.warning("No FCAR benchmarks available.")

    if use_fcar:
        st.markdown(
            '<div class="status-banner fcar">'
            f'{_icon("balance", 18)} FCAR Mode \u2014 '
            'Fairness-constrained cost weights active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-banner ar">'
            f'{_icon("straighten", 18)} Unconstrained AR \u2014 '
            'Standard cost minimization (no fairness adjustment)</div>',
            unsafe_allow_html=True,
        )
    _divider()

    # ── Applicant Profile ──
    applicant = X_test.iloc[selected_idx]
    sensitive_info = A_test.iloc[selected_idx]
    current_prob = proba[selected_idx]
    group_cols = get_sensitive_attributes(config)

    col_profile, col_score = st.columns([3, 1])
    with col_profile:
        _section("person", "Applicant Profile")
        badges = ""
        for g in group_cols:
            if g in sensitive_info.index:
                badges += (
                    f'<span class="badge badge-info" style="margin-right:8px;">'
                    f'{g.replace("_"," ").title()}: {sensitive_info[g]}</span>'
                )
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("")

        mutable_num = set(get_mutable_numeric_cols(config))
        mutable_cat = set(get_mutable_categorical_cols(config))
        profile = applicant.to_dict()

        table_rows = ""
        for feat, val in profile.items():
            if feat in mutable_num:
                cls, tag, ic = "feat-mutable", "Mutable (Numeric)", _icon("build", 14)
            elif feat in mutable_cat:
                cls, tag, ic = "feat-mutable", "Mutable (Categorical)", _icon("build", 14)
            else:
                cls, tag, ic = "feat-immutable", "Immutable", _icon("lock", 14)
            table_rows += (
                f'<tr><td class="{cls}">{feat}</td><td>{val}</td>'
                f'<td><span style="font-size:11px;">{ic} {tag}</span></td></tr>'
            )

        st.markdown(
            '<div style="max-height:380px;overflow-y:auto;border-radius:var(--radius-md);">'
            '<table class="profile-table">'
            '<thead><tr><th>Feature</th><th>Value</th><th>Type</th></tr></thead>'
            f'<tbody>{table_rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    with col_score:
        _section("gavel", "Model Decision")
        denied = _is_rejected(current_prob, target_cls)
        st.markdown(_score_ring(current_prob, denied), unsafe_allow_html=True)

    _divider()

    # ── Custom Constraints (UC-106) ──
    _section("build", "Custom Constraints")
    st.markdown(
        '<div style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;">'
        'Optionally lock features, limit change amounts, or restrict '
        'categorical changes before generating the recourse plan.</div>',
        unsafe_allow_html=True,
    )
    custom_constraints = {"locked": [], "max_change": {}, "max_cat_changes": None}
    mutable_num_list = get_mutable_numeric_cols(config)
    mutable_cat_list = get_mutable_categorical_cols(config)

    with st.expander("Configure Constraints", expanded=False):
        # ── Lock features ──
        st.markdown("**Lock Features** — prevent specific features from changing")
        all_mutable = mutable_num_list + mutable_cat_list
        locked = st.multiselect(
            "Features to lock (will not change)",
            options=all_mutable,
            default=[],
            format_func=lambda x: _human_feature(x),
            help="Select features that the solver must keep at their current value.",
        )
        custom_constraints["locked"] = locked

        st.markdown("---")

        # ── Max change limits for numeric features ──
        st.markdown("**Max Change Limits** — cap how much each numeric feature can change")
        unlocked_num = [c for c in mutable_num_list if c not in locked]
        if unlocked_num:
            for feat in unlocked_num:
                x0_val = float(applicant.get(feat, 0))
                train_range = float(X_train[feat].max() - X_train[feat].min())
                if train_range < 1:
                    train_range = 1.0
                default_max = train_range
                spec = config.get("mutable_numeric", {}).get(feat, {})
                # sensible ceiling from config
                if "max_decrease" in spec:
                    default_max = min(default_max, float(spec["max_decrease"]))
                elif "max_rel_decrease" in spec:
                    default_max = min(default_max, abs(x0_val * float(spec["max_rel_decrease"])))
                user_max = st.slider(
                    f"{_human_feature(feat)} — max change",
                    min_value=0.0,
                    max_value=float(round(train_range, 2)),
                    value=float(round(default_max, 2)),
                    step=max(0.01, round(train_range / 100, 2)),
                    key=f"cc_{feat}",
                )
                if user_max < default_max - 0.01:
                    custom_constraints["max_change"][feat] = user_max
        else:
            st.caption("All numeric features are locked.")

        st.markdown("---")

        # ── Max categorical changes ──
        unlocked_cat = [c for c in mutable_cat_list if c not in locked]
        default_mcc = config.get("max_cat_changes", len(unlocked_cat))
        if unlocked_cat:
            mcc = st.number_input(
                "Max categorical features allowed to change",
                min_value=0,
                max_value=len(unlocked_cat),
                value=min(default_mcc, len(unlocked_cat)),
                step=1,
                help="Limit how many categorical features the solver may alter.",
            )
            custom_constraints["max_cat_changes"] = int(mcc)
        else:
            st.caption("No unlocked categorical features.")

    if st.button(
        "Generate Optimal Recourse Plan",
        type="primary",
        use_container_width=True,
    ):
        fcar_gcol = selected_gcol if use_fcar else None
        _solve_and_render(
            selected_idx, applicant, current_prob, pipe, config,
            X_train, target_cls, use_fcar, dataset, sensitive_info,
            group_cols, fcar_gcol, custom_constraints,
        )


# ──────────────────────────────────────────────────────────────

def _solve_and_render(test_idx, x0, p0, pipe, config, X_train,
                      target_cls, use_fcar, dataset, sensitive_info,
                      group_cols, fcar_gcol=None, custom_constraints=None):
    import copy
    config = copy.deepcopy(config)           # isolate from cached object

    # ── Apply custom constraints from UI ──
    cc = custom_constraints or {}
    locked_feats = cc.get("locked", [])
    user_max_change = cc.get("max_change", {})
    user_max_cat = cc.get("max_cat_changes", None)

    # Lock features: remove them from mutable sections
    for feat in locked_feats:
        if feat in config.get("mutable_numeric", {}):
            del config["mutable_numeric"][feat]
        if feat in config.get("mutable_categorical", {}):
            del config["mutable_categorical"][feat]

    # Apply user max-change limits on numeric features
    for feat, max_chg in user_max_change.items():
        if feat in config.get("mutable_numeric", {}):
            spec = config["mutable_numeric"][feat]
            direction = spec.get("direction", "auto")
            if direction == "decrease":
                spec["max_decrease"] = min(
                    spec.get("max_decrease", max_chg), max_chg
                )
                spec.pop("max_rel_decrease", None)
            elif direction == "increase":
                spec["max_increase"] = min(
                    spec.get("max_increase", max_chg), max_chg
                )
                spec.pop("max_rel_increase", None)
            else:
                # auto direction — cap both ways
                spec["max_decrease"] = min(
                    spec.get("max_decrease", max_chg), max_chg
                )
                spec["max_increase"] = min(
                    spec.get("max_increase", max_chg), max_chg
                )
                spec.pop("max_rel_decrease", None)
                spec.pop("max_rel_increase", None)

    # Apply user max categorical changes
    if user_max_cat is not None:
        config["max_cat_changes"] = user_max_cat

    # Show constraint banner if any user constraints are active
    _active_cc = []
    if locked_feats:
        _active_cc.append(f"Locked: {', '.join(_human_feature(f) for f in locked_feats)}")
    if user_max_change:
        _mc = "; ".join(f"{_human_feature(f)} \u2264 {v:.1f}" for f, v in user_max_change.items())
        _active_cc.append(f"Max change: {_mc}")
    if user_max_cat is not None:
        _active_cc.append(f"Max cat changes: {user_max_cat}")
    if _active_cc:
        st.markdown(
            '<div style="background:rgba(99,102,241,0.08);'
            'border:1px solid rgba(99,102,241,0.25);'
            'border-radius:var(--radius-md);padding:12px 18px;margin:8px 0;'
            'font-size:12px;color:#a5b4fc;line-height:1.7;">'
            '<b>⚙ Custom Constraints Active</b><br/>'
            + "<br/>".join(_active_cc)
            + '</div>',
            unsafe_allow_html=True,
        )

    num_w = dict(get_numeric_cost_weights(config))
    cat_w = dict(get_categorical_step_weights(config))
    applied_overrides = False
    active_group = None

    if use_fcar and fcar_gcol:
        gcol = fcar_gcol
        gval = str(sensitive_info.get(gcol, ""))
        summary_path = (
            ART / "reports" / "benchmarks" / f"{dataset}_{gcol}_ab_summary.json"
        )
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            fcar_weights = summary.get("fcar_overrides", {})
            if gval in fcar_weights:
                num_w.update(
                    {k: float(v) for k, v in fcar_weights[gval].get("num", {}).items()}
                )
                cat_w.update(
                    {k: float(v) for k, v in fcar_weights[gval].get("cat", {}).items()}
                )
                applied_overrides = True
                active_group = f"{gcol} = {gval}"
            else:
                # Build informative explanation
                avail = list(fcar_weights.keys())
                avail_badges = " ".join(
                    f'<span class="badge badge-fcar">{g}</span>' for g in avail
                )
                # Show per-group burden from the benchmark
                ar_burden = summary.get("unconstrained_ar", {}).get("social_burden", {})
                group_burden = ar_burden.get(gval, None)
                burden_note = ""
                if group_burden is not None and ar_burden:
                    avg_b = sum(ar_burden.values()) / len(ar_burden)
                    if group_burden <= avg_b:
                        burden_note = (
                            f'<div style="margin-top:8px;font-size:12px;color:var(--text-muted);">'
                            f'This group\'s AR burden ({group_burden:.4f}) is at or below '
                            f'the average ({avg_b:.4f}), so the Auto-FCAR tuner did not '
                            f'adjust weights for it \u2014 it was not disadvantaged.</div>'
                        )
                    else:
                        burden_note = (
                            f'<div style="margin-top:8px;font-size:12px;color:var(--text-muted);">'
                            f'This group\'s AR burden is {group_burden:.4f} '
                            f'(avg: {avg_b:.4f}).</div>'
                        )
                st.markdown(
                    '<div style="background:rgba(245,158,11,0.08);'
                    'border:1px solid rgba(245,158,11,0.25);'
                    'border-radius:var(--radius-lg);padding:20px 24px;margin:12px 0;">'
                    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                    f'{_icon("warning", 20)}'
                    '<span style="font-size:14px;font-weight:700;color:#fbbf24;">'
                    f'No FCAR overrides for group: {gcol} = {gval}</span></div>'
                    '<div style="font-size:13px;color:var(--text-secondary);line-height:1.6;">'
                    'The Auto-FCAR tuner only adjusts weights for groups that are '
                    'disproportionately burdened. This group has <b>no weight overrides</b>, '
                    'so the output below is <b>identical to unconstrained AR</b>.</div>'
                    f'{burden_note}'
                    '<div style="margin-top:10px;font-size:12px;color:var(--text-muted);">'
                    f'Groups with FCAR overrides: {avail_badges}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    inst_config = copy.deepcopy(config)
    for col in num_w:
        if col in inst_config.get("mutable_numeric", {}):
            inst_config["mutable_numeric"][col]["cost_weight"] = num_w[col]
    for col in cat_w:
        if col in inst_config.get("mutable_categorical", {}):
            inst_config["mutable_categorical"][col]["step_weight"] = cat_w[col]

    # Also solve AR baseline when FCAR is active (for comparison)
    x_cf_ar, slack_ar, p1_ar = None, None, None
    if applied_overrides:
        with st.spinner("\u23f3 Solving AR baseline for comparison\u2026"):
            try:
                x_cf_ar, slack_ar = solve_recourse_mip(pipe, X_train, x0, config)
                p1_ar = float(pipe.predict_proba(pd.DataFrame([x_cf_ar]))[:, 1][0])
            except Exception:
                pass  # Non-critical; comparison just won't show

    with st.spinner("\u23f3 Solving Mixed-Integer Program\u2026"):
        try:
            x_cf, slack = solve_recourse_mip(pipe, X_train, x0, inst_config)
        except Exception as e:
            st.error(f"Solver error: {e}")
            return

    p1 = float(pipe.predict_proba(pd.DataFrame([x_cf]))[:, 1][0])
    flipped = (
        ((p0 < 0.5) and (p1 >= 0.5) and (slack == 0.0))
        if target_cls == 1
        else ((p0 >= 0.5) and (p1 < 0.5) and (slack == 0.0))
    )

    if applied_overrides:
        st.markdown(
            f'<div class="status-banner fcar">'
            f'{_icon("balance", 18)} FCAR weights '
            f'applied for: <b>{active_group}</b></div>',
            unsafe_allow_html=True,
        )
    elif use_fcar:
        st.markdown(
            '<div style="background:rgba(245,158,11,0.06);'
            'border:1px solid rgba(245,158,11,0.2);'
            'border-radius:var(--radius-md);padding:12px 18px;margin:8px 0;'
            'font-size:13px;color:#fcd34d;display:flex;align-items:center;gap:10px;">'
            f'{_icon("info", 16)}'
            'No FCAR overrides for this group \u2014 result is identical to unconstrained AR.'
            '</div>',
            unsafe_allow_html=True,
        )

    _section("trending_up", "Recourse Plan (UC-105)")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        _kpi("Score Before", f"{p0:.4f}", "red")
    with s2:
        _kpi("Score After", f"{p1:.4f}", "green" if flipped else "orange")
    with s3:
        _kpi(
            "Decision Flipped",
            "YES" if flipped else "NO",
            "green" if flipped else "red",
        )
    with s4:
        _kpi(
            "Slack (Infeasibility)",
            f"{slack:.4f}",
            "green" if slack == 0 else "red",
        )

    if not flipped:
        st.error(
            "The solver could not find a valid recourse path within "
            "the plausibility constraints."
        )
        _footer()
        return

    # ── Success banner ──
    score_delta = abs(p1 - p0)
    st.markdown(
        '<div class="success-banner">'
        f'<div class="icon">{_icon("check_circle", 28)}</div>'
        '<div><div class="text">Decision Successfully Overturned</div>'
        f'<div class="sub">Score improved by {score_delta:.4f} \u2014 '
        f'from {p0:.4f} to {p1:.4f}</div></div></div>',
        unsafe_allow_html=True,
    )

    # ── Dual score rings ──
    ring_before = _score_ring(p0, True)
    ring_after = _score_ring(p1, False)
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(
            '<div style="text-align:center;font-size:11px;font-weight:700;'
            'text-transform:uppercase;letter-spacing:1.5px;color:var(--text-muted);'
            'margin-bottom:-8px;">Before</div>',
            unsafe_allow_html=True,
        )
        st.markdown(ring_before, unsafe_allow_html=True)
    with rc2:
        st.markdown(
            '<div style="text-align:center;font-size:11px;font-weight:700;'
            'text-transform:uppercase;letter-spacing:1.5px;color:var(--text-muted);'
            'margin-bottom:-8px;">After</div>',
            unsafe_allow_html=True,
        )
        st.markdown(ring_after, unsafe_allow_html=True)

    _divider()

    # ── Build changes ──
    changes = []
    for c in get_mutable_numeric_cols(config):
        v0, v1 = float(x0.get(c, np.nan)), float(x_cf.get(c, np.nan))
        if pd.notna(v0) and pd.notna(v1) and abs(v1 - v0) > 1e-4:
            changes.append({
                "Feature": _human_feature(c),
                "_raw_feature": c,
                "Action": "\u2191 Increase" if v1 > v0 else "\u2193 Decrease",
                "Before": f"{v0:.2f}",
                "After": f"{v1:.2f}",
                "\u0394 Amount": f"{abs(v1 - v0):.2f}",
                "Cost Weight": f"{num_w.get(c, 1.0):.3f}",
                "_dir": "up" if v1 > v0 else "down",
            })
    for c in get_mutable_categorical_cols(config):
        v0, v1 = str(x0.get(c, "")), str(x_cf.get(c, ""))
        if v0 != v1:
            changes.append({
                "Feature": _human_feature(c),
                "_raw_feature": c,
                "Action": "Change",
                "Before": _human_value(v0),
                "After": _human_value(v1),
                "\u0394 Amount": "\u2014",
                "Cost Weight": f"{cat_w.get(c, 0.25):.3f}",
                "_dir": "swap",
            })

    if not changes:
        st.warning("Decision flipped without any feature changes (edge case).")
        _footer()
        return

    # ── Change pills ──
    _section("build", "Required Changes")
    pills = '<div class="change-pills">'
    for c in changes:
        d = c["_dir"]
        arrow = {"up": "\u2191", "down": "\u2193", "swap": "\u2194"}[d]
        pills += (
            f'<div class="change-pill {d}">'
            f'<span class="feat">{c["Feature"]}</span> '
            f'{arrow} {c["Before"]} \u2192 {c["After"]}</div>'
        )
    pills += '</div>'
    st.markdown(pills, unsafe_allow_html=True)

    # ── Before/After comparison table ──
    _before_after_table(changes)

    # ── Narrative ──
    delta_key = "\u0394 Amount"
    dash = "\u2014"
    parts = []
    for c in changes:
        feat = c["Feature"]
        bef = c["Before"]
        aft = c["After"]
        if c[delta_key] != dash:
            action = c["Action"].split(" ", 1)[1]
            amt = c[delta_key]
            parts.append(f"{action} {feat} by {amt} (from {bef} to {aft})")
        else:
            parts.append(f"Change {feat} from {bef} to {aft}")

    action_text = (
        "To overturn the negative decision, the applicant must: "
        + "; ".join(parts) + "."
    )

    # ── Build FCAR-aware explanation ──
    if applied_overrides:
        # Check if AR baseline produced the same actions
        ar_actions_same = False
        if x_cf_ar is not None:
            ar_changes_set = set()
            fcar_changes_set = set()
            for c_col in get_mutable_numeric_cols(config):
                v0_val = float(x0.get(c_col, 0))
                v_ar = float(x_cf_ar.get(c_col, 0)) if x_cf_ar is not None else v0_val
                v_fc = float(x_cf.get(c_col, 0))
                if abs(v_ar - v0_val) > 1e-4:
                    ar_changes_set.add((c_col, round(v_ar, 2)))
                if abs(v_fc - v0_val) > 1e-4:
                    fcar_changes_set.add((c_col, round(v_fc, 2)))
            for c_col in get_mutable_categorical_cols(config):
                v0_val = str(x0.get(c_col, ""))
                v_ar = str(x_cf_ar.get(c_col, "")) if x_cf_ar is not None else v0_val
                v_fc = str(x_cf.get(c_col, ""))
                if v_ar != v0_val:
                    ar_changes_set.add((c_col, v_ar))
                if v_fc != v0_val:
                    fcar_changes_set.add((c_col, v_fc))
            ar_actions_same = (ar_changes_set == fcar_changes_set)

        # Get default (AR) weights for comparison
        default_num_w_narr = dict(get_numeric_cost_weights(config))
        default_cat_w_narr = dict(get_categorical_step_weights(config))

        # Build weight comparison details
        weight_diffs = []
        for c in changes:
            feat = c["Feature"]
            raw_feat = c.get("_raw_feature", feat)
            fcar_w = float(c["Cost Weight"])
            ar_w = default_num_w_narr.get(raw_feat, default_cat_w_narr.get(raw_feat, fcar_w))
            if abs(fcar_w - ar_w) > 1e-4:
                pct = ((ar_w - fcar_w) / ar_w) * 100 if ar_w > 0 else 0
                weight_diffs.append(
                    f"<b>{feat}</b>: {ar_w:.3f} &rarr; {fcar_w:.3f} "
                    f"({pct:+.0f}%)"
                )

        if ar_actions_same and weight_diffs:
            # Same actions, different weights — explain the academic rationale
            weight_lines = "<br/>".join(weight_diffs)
            narrative_html = (
                f'<div class="narrative-box">'
                f'<b>Explanation (FCAR Mode):</b><br/>'
                f'{action_text}'
                f'<br/><br/>'
                f'<b>Why are the actions the same as standard AR?</b><br/>'
                f'For this applicant, only <b>one feasible path</b> exists to '
                f'overturn the decision given the model coefficients and '
                f'plausibility constraints (e.g., maximum loan reduction, '
                f'monotonic category improvements). No weight adjustment can '
                f'create an alternative path that does not exist.'
                f'<br/><br/>'
                f'<b>What did FCAR change?</b><br/>'
                f'FCAR adjusted the <b>cost weights</b> used to measure how '
                f'burdensome this action is for <b>{active_group}</b>:<br/>'
                f'{weight_lines}'
                f'<br/><br/>'
                f'<b>How does this help fairness?</b><br/>'
                f'FCAR operates on the <i>Social Burden</i> metric: '
                f'<code>SB(g) = rejection_rate(g) &times; avg_recourse_cost(g)</code>. '
                f'By lowering the cost weight, the <b>same physical action</b> '
                f'is measured as less costly for this group. Across all '
                f'applicants in <b>{active_group}</b>, this systematically '
                f'reduces the group\'s aggregate recourse burden, narrowing the '
                f'disparity gap between demographic groups. '
                f'The action itself remains unchanged because it is the only '
                f'mathematically feasible solution &mdash; but its <b>measured '
                f'cost contribution</b> to the group\'s Social Burden is reduced.'
                f'</div>'
            )
        elif weight_diffs:
            # Different actions AND different weights
            weight_lines = "<br/>".join(weight_diffs)
            narrative_html = (
                f'<div class="narrative-box">'
                f'<b>Explanation (FCAR Mode):</b><br/>'
                f'{action_text}'
                f'<br/><br/>'
                f'<b>How does this differ from standard AR?</b><br/>'
                f'FCAR adjusted the cost weights for <b>{active_group}</b>, '
                f'which changed the solver\'s optimization objective:<br/>'
                f'{weight_lines}'
                f'<br/><br/>'
                f'With lower weights on certain features, those changes become '
                f'<b>cheaper in the objective function</b>, causing the MIP '
                f'solver to choose a <b>different recourse path</b> than '
                f'standard AR. This redistributes effort toward actions that '
                f'are less burdensome for this demographic group.'
                f'<br/><br/>'
                f'<b>Fairness impact:</b> Across all applicants in '
                f'<b>{active_group}</b>, these re-weighted paths reduce the '
                f'group\'s aggregate Social Burden '
                f'(<code>SB = rejection_rate &times; avg_cost</code>), '
                f'narrowing the disparity gap between demographic groups.'
                f'</div>'
            )
        else:
            # FCAR active but no weight differences on changed features
            narrative_html = (
                f'<div class="narrative-box">'
                f'<b>Explanation (FCAR Mode):</b><br/>'
                f'{action_text}'
                f'<br/><br/>'
                f'<i>The FCAR weight adjustments for <b>{active_group}</b> '
                f'did not affect the features involved in this recourse plan. '
                f'The result is equivalent to standard AR for this applicant.</i>'
                f'</div>'
            )
    else:
        # Standard AR mode
        narrative_html = (
            f'<div class="narrative-box">'
            f'<b>Explanation:</b><br/>'
            f'{action_text}'
            f'<br/><br/>'
            f'<i>This is a standard Algorithmic Recourse (AR) recommendation '
            f'using uniform cost weights. Enable FCAR above to see how '
            f'fairness-constrained weights adjust the recourse cost for '
            f'specific demographic groups.</i>'
            f'</div>'
        )

    st.markdown(narrative_html, unsafe_allow_html=True)

    # ── Downloadable Recourse Report ──
    import io as _io
    _recourse_rows = []
    for _c in changes:
        _recourse_rows.append({
            "Feature": _c["Feature"],
            "Before": _c["Before"],
            "After": _c["After"],
            "Change": _c["\u0394 Amount"],
            "Action": _c["Action"],
            "Cost Weight": _c["Cost Weight"],
        })
    _df_recourse = pd.DataFrame(_recourse_rows)
    _meta = pd.DataFrame([{
        "Applicant": f"#{test_idx}",
        "Dataset": dataset,
        "Mode": "FCAR" if use_fcar else "Unconstrained AR",
        "FCAR Group": active_group or "-",
        "Score Before": f"{p0:.4f}",
        "Score After": f"{p1:.4f}",
        "Decision Flipped": "Yes" if flipped else "No",
        "Slack": f"{slack:.4f}",
    }])
    _buf = _io.StringIO()
    _buf.write("# Recourse Summary\n")
    _meta.to_csv(_buf, index=False)
    _buf.write("\n# Required Changes\n")
    _df_recourse.to_csv(_buf, index=False)
    st.download_button(
        label="\u2b07 Download Recourse Report (CSV)",
        data=_buf.getvalue(),
        file_name=f"recourse_report_{dataset}_applicant{test_idx}.csv",
        mime="text/csv",
    )

    # ── AR vs FCAR side-by-side comparison (only when FCAR overrides applied) ──
    if applied_overrides and x_cf_ar is not None:
        _divider()
        _section("compare_arrows", "AR vs FCAR Comparison")
        st.markdown(
            '<div style="font-size:13px;color:var(--text-secondary);margin-bottom:16px;">'
            'Side-by-side view showing how FCAR redistributes effort across features '
            'compared to unconstrained Algorithmic Recourse.</div>',
            unsafe_allow_html=True,
        )

        default_num_w = dict(get_numeric_cost_weights(config))
        default_cat_w = dict(get_categorical_step_weights(config))

        cmp_rows = ""
        mutable_num_set = set(get_mutable_numeric_cols(config))
        mutable_cat_set = set(get_mutable_categorical_cols(config))
        all_mutable = list(get_mutable_numeric_cols(config)) + list(get_mutable_categorical_cols(config))
        has_diff = False

        for c in all_mutable:
            v0 = x0.get(c, "")
            is_num = c in mutable_num_set
            if is_num:
                v_ar = float(x_cf_ar.get(c, v0))
                v_fc = float(x_cf.get(c, v0))
                v0f = float(v0)
                ar_changed = abs(v_ar - v0f) > 1e-4
                fc_changed = abs(v_fc - v0f) > 1e-4
                if not ar_changed and not fc_changed:
                    continue
                ar_str = f"{v_ar:.0f}" if ar_changed else "\u2014"
                fc_str = f"{v_fc:.0f}" if fc_changed else "\u2014"
                orig_str = f"{v0f:.0f}"
                w_ar = f"{default_num_w.get(c, 1.0):.3f}"
                w_fc = f"{num_w.get(c, 1.0):.3f}"
            else:
                v_ar = str(x_cf_ar.get(c, v0))
                v_fc = str(x_cf.get(c, v0))
                v0s = str(v0)
                ar_changed = v_ar != v0s
                fc_changed = v_fc != v0s
                if not ar_changed and not fc_changed:
                    continue
                ar_str = v_ar if ar_changed else "\u2014"
                fc_str = v_fc if fc_changed else "\u2014"
                orig_str = v0s
                w_ar = f"{default_cat_w.get(c, 0.25):.3f}"
                w_fc = f"{cat_w.get(c, 0.25):.3f}"

            diff = ar_str != fc_str or w_ar != w_fc
            if diff:
                has_diff = True
            highlight = ' style="background:rgba(16,185,129,0.06);"' if diff else ""
            diff_badge = (
                '<span style="color:#10b981;font-size:10px;font-weight:700;">● DIFFERENT</span>'
                if diff else
                '<span style="color:var(--text-muted);font-size:10px;">same</span>'
            )
            cmp_rows += (
                f'<tr{highlight}>'
                f'<td style="font-weight:600;">{c}</td>'
                f'<td>{orig_str}</td>'
                f'<td>{ar_str}</td><td style="font-size:11px;">{w_ar}</td>'
                f'<td style="font-weight:600;">{fc_str}</td><td style="font-size:11px;">{w_fc}</td>'
                f'<td>{diff_badge}</td>'
                f'</tr>'
            )

        if cmp_rows and has_diff:
            st.markdown(
                '<div style="overflow-x:auto;">'
                '<table class="profile-table" style="font-size:13px;">'
                '<thead><tr>'
                '<th>Feature</th><th>Original</th>'
                '<th style="color:#f87171;">AR Value</th>'
                '<th style="color:#f87171;">AR Wt</th>'
                '<th style="color:#34d399;">FCAR Value</th>'
                '<th style="color:#34d399;">FCAR Wt</th>'
                '<th>Status</th>'
                '</tr></thead>'
                f'<tbody>{cmp_rows}</tbody>'
                '</table></div>',
                unsafe_allow_html=True,
            )

            # Summary insight
            ar_feat_count = sum(
                1 for c in all_mutable
                if (c in mutable_num_set and abs(float(x_cf_ar.get(c, x0.get(c, 0))) - float(x0.get(c, 0))) > 1e-4)
                or (c in mutable_cat_set and str(x_cf_ar.get(c, "")) != str(x0.get(c, "")))
            )
            fc_feat_count = sum(
                1 for c in all_mutable
                if (c in mutable_num_set and abs(float(x_cf.get(c, x0.get(c, 0))) - float(x0.get(c, 0))) > 1e-4)
                or (c in mutable_cat_set and str(x_cf.get(c, "")) != str(x0.get(c, "")))
            )
            st.markdown(
                '<div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);'
                'border-radius:var(--radius-md);padding:14px 20px;margin-top:12px;font-size:13px;'
                'color:var(--text-secondary);line-height:1.7;">'
                f'{_icon("bar_chart", 16)} <b>FCAR redistributes the recourse burden</b> by adjusting cost weights. '
                f'AR changes <b>{ar_feat_count}</b> feature(s), FCAR changes <b>{fc_feat_count}</b>. '
                f'With lower weights on features like <i>duration</i> ({num_w.get("duration", 1.0):.3f}) '
                f'and <i>credit_amount</i> ({num_w.get("credit_amount", 1.0):.3f}), '
                f'FCAR may push these features harder while avoiding costlier categorical changes, '
                f'resulting in a fairer distribution of effort across demographic groups.'
                '</div>',
                unsafe_allow_html=True,
            )
        elif cmp_rows:
            st.markdown(
                '<div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);'
                'border-radius:var(--radius-md);padding:14px 20px;margin-top:4px;font-size:13px;'
                'color:#fcd34d;">'
                f'{_icon("info", 16)} For this specific applicant, AR and FCAR produce the same recourse path. '
                'This can happen when the decision boundary constraint dominates the solution — '
                'the applicant\'s score is far enough from 0.5 that the solver must push features to '
                'their plausibility limits regardless of weights. '
                '<b>FCAR still reduces group-level burden disparity across the full population.</b>'
                '</div>',
                unsafe_allow_html=True,
            )

    _footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2 — Fairness Audit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_audit(dataset):
    ds_label = dataset.replace("_", " ").title()

    st.markdown(
        f'<div class="page-title">'
        f'Fairness Audit \u2014 {ds_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">MISOB Social Burden audit comparing '
        'Unconstrained AR vs FCAR across demographic groups. '
        'Implements Use Case UC-103.</div>',
        unsafe_allow_html=True,
    )

    summaries = load_eval_summaries()
    ds_summaries = {k: v for k, v in summaries.items() if dataset in k}
    if not ds_summaries:
        st.warning(
            f"No benchmark results for **{ds_label}**. "
            "Run `python scripts/benchmark_ab.py` first."
        )
        return

    selected_key = st.selectbox(
        "Sensitive Attribute",
        list(ds_summaries.keys()),
        format_func=lambda x: ds_summaries[x]["group_col"].replace("_", " ").title(),
    )
    data = ds_summaries[selected_key]
    ar = data["unconstrained_ar"]
    fc = data["fcar"]
    stats = data.get("statistical_tests", {})
    reduction = stats.get("disparity_reduction", {}).get("gap_reduction_pct", 0)

    _divider()

    # ── KPIs ──
    _section("bar_chart", "Key Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi("Evaluated", str(data["n_evaluated"]), "cyan")
    with k2:
        _kpi("AR Feasibility", f"{ar['feasible_rate']:.0%}", "orange")
    with k3:
        _kpi("FCAR Feasibility", f"{fc['feasible_rate']:.0%}", "green")
    with k4:
        _kpi(
            "Gap Reduction",
            f"\u2193{reduction:.1f}%",
            "green" if reduction > 0 else "red",
        )
    with k5:
        wil = stats.get("overall_wilcoxon", {})
        p = wil.get("p_value", 1.0)
        _kpi(
            "p-value",
            "< 0.001" if p < 0.001 else f"{p:.4f}",
            "green" if p < 0.05 else "red",
        )

    # ── Downloadable Audit Report ──
    import io
    import pandas as pd
    # Prepare audit metrics for CSV export
    audit_rows = []
    for label, d in [("Unconstrained AR", ar), ("FCAR (Auto-Tuned)", fc)]:
        audit = d["audit"]
        disp = d["disparity"]
        audit_rows.append({
            "Audit Type": label,
            "Audit Score": audit["audit_score"],
            "Disparity Gap": disp["gap"],
            "Burden Ratio": disp["ratio"],
            "Worst Group": disp.get("worst_group", "-"),
            "Best Group": disp.get("best_group", "-"),
            "Avg Burden": d["avg_burden"],
            "Feasible Rate": d["feasible_rate"],
        })
    # Add statistical validation row
    wil = stats.get("overall_wilcoxon", {})
    audit_rows.append({
        "Audit Type": "Statistical Validation",
        "Audit Score": "-",
        "Disparity Gap": "-",
        "Burden Ratio": "-",
        "Worst Group": "-",
        "Best Group": "-",
        "Avg Burden": wil.get("mean_diff", "-"),
        "Feasible Rate": "-",
        "p-value": wil.get("p_value", "-"),
        "Significant (alpha=0.05)": wil.get("significant_at_005", "-"),
    })
    df_audit = pd.DataFrame(audit_rows)
    # Reorder columns for clarity
    cols = [
        "Audit Type", "Audit Score", "Disparity Gap", "Burden Ratio", "Worst Group", "Best Group", "Avg Burden", "Feasible Rate", "p-value", "Significant (alpha=0.05)"
    ]
    for c in cols:
        if c not in df_audit.columns:
            df_audit[c] = "-"
    df_audit = df_audit[cols]
    csv = df_audit.to_csv(index=False)
    st.download_button(
        label="Download Audit Report (CSV)",
        data=csv,
        file_name=f"audit_report_{dataset}.csv",
        mime="text/csv",
    )
    _divider()

    # ── Side-by-side Audit Cards ──
    _section("search", "MISOB Audit Comparison (UC-103)")
    col_ar, col_arrow, col_fc = st.columns([5, 1, 5])

    for col, label, d, badge_cls in [
        (col_ar, "Unconstrained AR", ar, "badge-ar"),
        (col_fc, "FCAR (Auto-Tuned)", fc, "badge-fcar"),
    ]:
        with col:
            passed = d["audit"].get("passed", False)
            card_cls = "pass" if passed else "fail"
            status_badge = (
                '<span class="badge badge-pass">PASS</span>'
                if passed
                else '<span class="badge badge-fail">FAIL</span>'
            )
            dash = "\u2014"
            audit_score = d["audit"]["audit_score"]
            gap = d["disparity"]["gap"]
            ratio = d["disparity"]["ratio"]
            worst_g = d["disparity"].get("worst_group", dash)
            best_g = d["disparity"].get("best_group", dash)
            avg_b = d["avg_burden"]
            st.markdown(
                f'<div class="audit-card {card_cls}">'
                f'<div class="header"><span class="badge {badge_cls}">{label}</span>'
                f'{status_badge}</div>'
                '<table>'
                f'<tr><td>Audit Score</td><td>{audit_score:.3f}</td></tr>'
                f'<tr><td>Disparity Gap</td><td>{gap:.4f}</td></tr>'
                f'<tr><td>Burden Ratio</td><td>{ratio:.2f}x</td></tr>'
                f'<tr><td>Worst Group</td><td>{worst_g}</td></tr>'
                f'<tr><td>Best Group</td><td>{best_g}</td></tr>'
                f'<tr><td>Avg Burden</td><td>{avg_b:.4f}</td></tr>'
                '</table></div>',
                unsafe_allow_html=True,
            )

            # Sparkline burden bars per group
            sb = d["social_burden"]
            max_burden = max(sb.values()) if sb else 1
            bar_color = "amber" if label.startswith("Uncon") else "indigo"
            bars_html = ""
            for grp in sorted(sb, key=sb.get, reverse=True):
                bars_html += _spark_bar(grp, sb[grp], max_burden, bar_color)
            st.markdown(bars_html, unsafe_allow_html=True)

    # Center arrow between the two cards
    with col_arrow:
        red_str = f"{reduction:.0f}%" if reduction > 0 else "\u2014"
        st.markdown(
            '<div class="comparison-arrow">'
            f'<div class="arrow">{_icon("arrow_forward", 32)}</div>'
            f'<div class="label">\u2193 {red_str}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Statistical Validation ──
    wil = stats.get("overall_wilcoxon", {})
    if wil:
        _divider()
        _section("science", "Statistical Validation")
        t1, t2, t3 = st.columns(3)
        with t1:
            p_val = wil.get("p_value", 1.0)
            _kpi(
                "Wilcoxon p-value",
                "< 0.001" if p_val < 0.001 else f"{p_val:.4f}",
                "green" if p_val < 0.05 else "red",
                sub="Paired signed-rank test",
            )
        with t2:
            _kpi(
                "Mean Burden Diff",
                f"{wil.get('mean_diff', 0):.4f}",
                sub="AR \u2212 FCAR (per individual)",
            )
        with t3:
            sig = wil.get("significant_at_005", False)
            _kpi(
                "Significant (\u03b1=0.05)",
                "Yes" if sig else "No",
                "green" if sig else "red",
            )

    # ── Chart ──
    _divider()
    chart_path = (
        ART / "reports" / "figures"
        / f"{dataset}_{data['group_col']}_social_burden.png"
    )
    if chart_path.exists():
        st.image(
            str(chart_path),
            caption="Social Burden Comparison (AR vs FCAR)",
            use_container_width=True,
        )

    _footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3 — Evaluation Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_evaluation(pipe=None, config=None):
    st.markdown(
        '<div class="page-title">'
        'Evaluation Metrics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Comprehensive comparison of model quality, '
        'pre-recourse fairness gaps, and FCAR effectiveness across all datasets '
        'and sensitive attributes.</div>',
        unsafe_allow_html=True,
    )

    # ── Headline benchmark highlights ──
    bench_dir = ART / "reports" / "benchmarks"
    all_summaries = []
    if bench_dir.exists():
        for bf in sorted(bench_dir.glob("*_ab_summary.json")):
            with open(bf) as bfh:
                all_summaries.append(json.load(bfh))
    if all_summaries:
        reductions = []
        for s in all_summaries:
            ag = float(s["unconstrained_ar"]["disparity"]["gap"])
            fg = float(s["fcar"]["disparity"]["gap"])
            reductions.append(100 * (ag - fg) / ag if ag > 1e-9 else 0)
        best_red = max(reductions)
        avg_red = sum(reductions) / len(reductions)
        all_p = [
            s.get("statistical_tests", {}).get("overall_wilcoxon", {}).get("p_value", 1)
            for s in all_summaries
        ]
        all_sig = all(p < 0.05 for p in all_p)

        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            st.markdown(
                '<div class="highlight-card best">'
                f'<div class="big emerald">{best_red:.0f}%</div>'
                '<div class="lbl">Best Gap Reduction</div></div>',
                unsafe_allow_html=True,
            )
        with hc2:
            st.markdown(
                '<div class="highlight-card">'
                f'<div class="big indigo">{avg_red:.0f}%</div>'
                '<div class="lbl">Avg Gap Reduction</div></div>',
                unsafe_allow_html=True,
            )
        with hc3:
            st.markdown(
                '<div class="highlight-card">'
                f'<div class="big amber">{len(all_summaries)}</div>'
                '<div class="lbl">Benchmarks Run</div></div>',
                unsafe_allow_html=True,
            )
        with hc4:
            sig_label = "ALL" if all_sig else "PARTIAL"
            sig_color = "emerald" if all_sig else "amber"
            st.markdown(
                f'<div class="highlight-card sig">'
                f'<div class="big {sig_color}">{sig_label}</div>'
                '<div class="lbl">Significant (p&lt;0.05)</div></div>',
                unsafe_allow_html=True,
            )
        _divider()

    fig_dir = ART / "reports" / "figures"
    charts = [
        (
            "Model Quality",
            "eval_model_quality.png",
            "Accuracy, F1, ROC-AUC, and PR-AUC for each dataset\u2019s logistic regression baseline.",
        ),
        (
            "Fairness Gaps",
            "eval_fairness_gaps.png",
            "Demographic Parity and Equalized Odds differences before any recourse intervention.",
        ),
        (
            "Social Burden",
            "eval_social_burden_all.png",
            "Per-group Social Burden comparison showing how FCAR narrows the disparity gap.",
        ),
        (
            "Gap Reduction",
            "eval_gap_reduction.png",
            "Absolute disparity gap before/after FCAR, and burden ratio improvement towards parity.",
        ),
        (
            "Significance",
            "eval_statistical_summary.png",
            "Wilcoxon p-values (log scale) and MISOB audit scores confirming statistical significance.",
        ),
        (
            "Trade-offs",
            "eval_performance_dashboard.png",
            "Feasibility, flip rate, and average burden \u2014 FCAR maintains full utility with zero trade-off.",
        ),
    ]

    tabs = st.tabs([c[0] for c in charts])
    for tab, (title, filename, desc) in zip(tabs, charts):
        with tab:
            st.caption(desc)
            img_path = fig_dir / filename
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(
                    f"Chart `{filename}` not found. "
                    "Run `python scripts/generate_full_evaluation.py`."
                )

    _divider()
    _section("summarize", "Results Summary")

    bench_dir = ART / "reports" / "benchmarks"
    if bench_dir.exists():
        rows = []
        for f in sorted(bench_dir.glob("*_ab_summary.json")):
            with open(f) as fh:
                s = json.load(fh)
            ag = float(s["unconstrained_ar"]["disparity"]["gap"])
            fg = float(s["fcar"]["disparity"]["gap"])
            pct = 100 * (ag - fg) / ag if ag > 1e-9 else 0
            wil = s.get("statistical_tests", {}).get("overall_wilcoxon", {})
            p = wil.get("p_value", 1.0)
            rows.append({
                "Dataset": s["dataset"].replace("_", " ").title(),
                "Group": s["group_col"].replace("_", " ").title(),
                "AR Gap": f"{ag:.4f}",
                "FCAR Gap": f"{fg:.4f}",
                "\u2193 Reduction": f"{pct:.1f}%",
                "p-value": "< 0.001" if p < 0.001 else f"{p:.4f}",
                "Feasibility": f"{s['fcar']['feasible_rate']:.0%}",
                "Audit Score": f"{s['fcar']['audit']['audit_score']:.3f}",
            })
        if rows:
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, use_container_width=True,
            )

    # ── Feature Importance Visualization ──
    if pipe is not None:
        _divider()
        _section("bar_chart", "Feature Importance")
        st.markdown(
            '<div style="font-size:13px;color:var(--text-secondary);margin-bottom:16px;">'
            'Logistic regression coefficient magnitudes (standardised) show which '
            'features have the greatest influence on the model\'s approval decision. '
            'Positive coefficients push toward approval; negative toward denial.</div>',
            unsafe_allow_html=True,
        )

        try:
            pre = pipe.named_steps["pre"]
            clf = pipe.named_steps["clf"]
            feat_names = list(pre.get_feature_names_out())
            coefs = clf.coef_.ravel()

            # Build readable feature name -> coefficient mapping
            feat_coef = []
            for fname, w in zip(feat_names, coefs):
                # Clean up ColumnTransformer prefixes
                readable = fname
                if readable.startswith("num__"):
                    readable = readable[5:]
                elif readable.startswith("cat__"):
                    readable = readable[5:]
                feat_coef.append((readable, float(w)))

            # Sort by absolute magnitude
            feat_coef.sort(key=lambda x: abs(x[1]), reverse=True)

            # Show top 20
            top_n = feat_coef[:20]
            max_abs = max(abs(c) for _, c in top_n) if top_n else 1.0

            # Render as horizontal bars using HTML
            bars_html = ""
            for feat, coef in top_n:
                pct = abs(coef) / max_abs * 100
                color = "#10b981" if coef > 0 else "#f43f5e"
                direction = "+" if coef > 0 else "\u2212"
                label = _human_feature(feat)
                bars_html += (
                    '<div style="display:flex;align-items:center;gap:10px;'
                    'padding:5px 0;border-bottom:1px solid #f1f5f9;">'
                    f'<div style="width:180px;font-size:12px;color:var(--text-secondary);'
                    f'text-align:right;flex-shrink:0;overflow:hidden;'
                    f'text-overflow:ellipsis;white-space:nowrap;">{label}</div>'
                    f'<div style="flex:1;height:10px;background:#f1f5f9;'
                    f'border-radius:5px;overflow:hidden;">'
                    f'<div style="width:{pct:.1f}%;height:100%;'
                    f'background:{color};border-radius:5px;"></div></div>'
                    f'<div style="width:70px;font-size:11px;font-weight:700;'
                    f'color:{color};text-align:right;font-variant-numeric:tabular-nums;">'
                    f'{direction}{abs(coef):.4f}</div>'
                    '</div>'
                )

            # Legend
            legend = (
                '<div style="display:flex;gap:20px;margin-bottom:12px;font-size:12px;">'
                '<div style="display:flex;align-items:center;gap:6px;">'
                '<div style="width:12px;height:12px;background:#10b981;border-radius:3px;"></div>'
                '<span style="color:var(--text-muted);">Positive (pushes toward approval)</span></div>'
                '<div style="display:flex;align-items:center;gap:6px;">'
                '<div style="width:12px;height:12px;background:#f43f5e;border-radius:3px;"></div>'
                '<span style="color:var(--text-muted);">Negative (pushes toward denial)</span></div>'
                '</div>'
            )

            st.markdown(legend, unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:var(--bg);border:1px solid var(--border);'
                f'border-radius:var(--radius);padding:20px 24px;">'
                f'{bars_html}</div>',
                unsafe_allow_html=True,
            )

            # Download coefficients
            _df_coef = pd.DataFrame(feat_coef, columns=["Feature", "Coefficient"])
            csv_coef = _df_coef.to_csv(index=False)
            st.download_button(
                label="Download Feature Coefficients (CSV)",
                data=csv_coef,
                file_name="feature_importance.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.warning(f"Could not extract feature importance: {e}")

    _footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4 — About & Methodology
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_about():
    st.markdown(
        '<div class="about-hero">'
        '<h2>Fairness Constrained Actionable Recourse</h2>'
        '<p>FCAR is a research framework that addresses the systemic unfairness '
        'in algorithmic recourse. When AI systems deny financial services, the '
        'recommended corrective actions often impose a disproportionately higher '
        'burden on protected demographic groups. FCAR integrates the Social Burden '
        'metric into a constrained MIP solver to ensure equitable recourse across '
        'all groups.</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        _section("target", "Research Questions & Answers")
        for rq, q, a, status in [
            (
                "RQ1",
                "Can Social Burden be integrated into constrained optimization?",
                "Yes \u2014 MISOB metric drives asymmetric weight tuning in the MIP solver",
                _icon("check_circle", 22),
            ),
            (
                "RQ2",
                "Does FCAR reduce disparity across groups?",
                "Yes \u2014 45\u201364% gap reduction across all benchmarks, all p < 0.01",
                _icon("check_circle", 22),
            ),
            (
                "RQ3",
                "Does FCAR preserve utility?",
                "Yes \u2014 100% feasibility, 100% flip rate, lower average burden",
                _icon("check_circle", 22),
            ),
        ]:
            st.markdown(
                '<div class="glass-card" style="padding:20px;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<span class="badge badge-fcar">{rq}</span>'
                f'<span style="font-size:18px;">{status}</span></div>'
                f'<div style="font-size:14px;font-weight:600;color:var(--text-primary);margin-bottom:6px;">{q}</div>'
                f'<div style="font-size:13px;color:var(--text-secondary);">{a}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        _section("sync", "FCAR Pipeline")
        _methodology_flow()

        _section("architecture", "System Architecture")
        arch_data = [
            ("Model", "Scikit-learn Logistic Regression + ColumnTransformer"),
            ("Solver", "Pyomo MIP with HiGHS (Mixed-Integer Linear Programming)"),
            ("Metrics", "MISOB Social Burden (Barrainkua et al., 2025)"),
            ("Tuning", "Asymmetric feature-level weight adjustment"),
            ("Dashboard", "Streamlit (this UI)"),
            ("API", "FastAPI REST endpoint (NFR04)"),
            ("Datasets", "Adult Income \u00b7 German Credit \u00b7 Default Credit"),
        ]
        arch_rows = "".join(
            f'<tr><td style="font-weight:600;color:var(--accent-indigo);width:120px;">{k}</td>'
            f'<td>{v}</td></tr>'
            for k, v in arch_data
        )
        st.markdown(
            '<div class="glass-card">'
            f'<table style="width:100%;border-collapse:collapse;">{arch_rows}</table>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        _section("menu_book", "Use Cases")
        for uc, name in [
            ("UC-101", "Generate Fair Recourse"),
            ("UC-102", "Set Fairness Constraint"),
            ("UC-103", "Retrieve Audit Score"),
            ("UC-104", "Benchmark Performance"),
            ("UC-105", "Individual Explanation"),
        ]:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:12px;padding:10px 0;'
                'border-bottom:1px solid var(--border-subtle);">'
                f'<span class="badge badge-info">{uc}</span>'
                f'<span style="font-size:14px;color:var(--text-secondary);">{name}</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        _section("auto_stories", "Key References")
        refs = [
            ("Barrainkua et al., 2025", "Who Pays for Fairness? Rethinking Recourse under Social Burden"),
            ("Wang et al., 2024", "Achieving Fairness via Actionable Recourse"),
            ("Ustun et al., 2019", "Actionable Recourse in Linear Classification"),
            ("Yetukuri, 2024", "Towards Socially Acceptable Algorithmic Models"),
        ]
        for author, title in refs:
            st.markdown(
                '<div style="padding:8px 0;border-bottom:1px solid var(--border-subtle);">'
                f'<div style="font-size:13px;font-weight:600;color:var(--text-primary);">{author}</div>'
                f'<div style="font-size:12px;color:var(--text-muted);font-style:italic;">{title}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        _section("person", "Author")
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:28px;">'
            '<div style="font-size:20px;font-weight:700;color:var(--text-primary);margin-bottom:4px;">Aqdhas Ali</div>'
            '<div style="font-size:13px;color:var(--text-muted);margin-bottom:12px;">w1954000 / 20210860</div>'
            '<div style="font-size:12px;color:var(--text-secondary);">Supervised by <b>Ms. Suvetha Suvendran</b></div>'
            '<div style="font-size:11px;color:var(--text-muted);margin-top:4px;">IIT / University of Westminster \u00b7 2025\u20132026</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        _section("link", "Quick Links")
        st.markdown(
            '<div class="glass-card">'
            '<div style="padding:6px 0;"><span style="color:var(--text-muted);font-size:13px;">API Docs:</span> '
            '<code>http://localhost:8000/docs</code></div>'
            '<div style="padding:6px 0;"><span style="color:var(--text-muted);font-size:13px;">GitHub:</span> '
            '<a href="https://github.com/aqdhasali/FCAR-Framework">FCAR-Framework</a></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    _footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    main()
