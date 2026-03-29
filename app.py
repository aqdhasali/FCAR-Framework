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
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Premium CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
/* ─── Google Font import ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ─── Root variables ─── */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --bg-glass: rgba(255, 255, 255, 0.03);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-accent: rgba(99, 102, 241, 0.3);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --accent-cyan: #06b6d4;
    --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    --gradient-success: linear-gradient(135deg, #059669 0%, #10b981 100%);
    --gradient-danger: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%);
    --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
}

/* ─── Global ─── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99, 102, 241, 0.1) !important;
}

/* ─── Headers ─── */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}
h1 { font-weight: 800 !important; }

/* ─── Glass Card ─── */
.glass-card {
    background: var(--gradient-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
}

/* ─── KPI Cards ─── */
.kpi-card {
    background: var(--gradient-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-primary);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.kpi-card.green::before  { background: var(--gradient-success); }
.kpi-card.red::before    { background: var(--gradient-danger); }
.kpi-card.orange::before { background: linear-gradient(135deg, #d97706, #f59e0b); }
.kpi-card.purple::before { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
.kpi-card.cyan::before   { background: linear-gradient(135deg, #0891b2, #06b6d4); }
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
}
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
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.02em;
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
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}
.badge-pass {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.badge-fail {
    background: rgba(244, 63, 94, 0.12);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.2);
}
.badge-fcar {
    background: rgba(99, 102, 241, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
}
.badge-ar {
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.2);
}
.badge-info {
    background: rgba(6, 182, 212, 0.12);
    color: #67e8f9;
    border: 1px solid rgba(6, 182, 212, 0.2);
}

/* ─── Section Header ─── */
.section-header {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 32px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ─── Score Gauge ─── */
.score-display {
    text-align: center;
    padding: 32px 16px;
}
.score-ring {
    position: relative;
    width: 140px;
    height: 140px;
    margin: 0 auto 16px;
}
.score-ring svg { transform: rotate(-90deg); }
.score-ring .bg { stroke: rgba(255,255,255,0.06); }
.score-ring .fg { transition: stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1); }
.score-ring .value {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.score-ring .value.denied  { color: var(--accent-rose); }
.score-ring .value.approved { color: var(--accent-emerald); }

/* ─── Change Pills ─── */
.change-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 16px 0;
}
.change-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border-radius: var(--radius-md);
    font-size: 13px;
    font-weight: 500;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}
.change-pill:hover { transform: translateY(-1px); }
.change-pill.up {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
}
.change-pill.down {
    background: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.2);
    color: #fda4af;
}
.change-pill.swap {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    color: #fcd34d;
}
.change-pill .feat { font-weight: 700; }

/* ─── Narrative Box ─── */
.narrative-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
    margin: 20px 0;
    position: relative;
}
.narrative-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    background: var(--gradient-primary);
    border-radius: var(--radius-lg) 0 0 var(--radius-lg);
}
.narrative-box b { color: #c4b5fd; }

/* ─── Audit Card ─── */
.audit-card {
    background: var(--gradient-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 28px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.audit-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
}
.audit-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
}
.audit-card.pass::before { background: var(--gradient-success); }
.audit-card.fail::before { background: var(--gradient-danger); }
.audit-card .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.audit-card table {
    width: 100%;
    border-collapse: collapse;
}
.audit-card table td {
    padding: 10px 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 14px;
}
.audit-card table td:first-child {
    color: var(--text-muted);
    font-weight: 500;
}
.audit-card table td:last-child {
    text-align: right;
    font-weight: 700;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
}
.audit-card table tr:last-child td { border-bottom: none; }

/* ─── Status Banner ─── */
.status-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    border-radius: var(--radius-md);
    font-size: 13px;
    font-weight: 600;
    margin: 12px 0;
}
.status-banner.fcar {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
}
.status-banner.ar {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    color: #fcd34d;
}

/* ─── Profile Table ─── */
.profile-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    font-size: 13px;
}
.profile-table th {
    background: rgba(99, 102, 241, 0.08);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 11px;
    padding: 12px 16px;
    text-align: left;
}
.profile-table td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-primary);
}
.profile-table tr:last-child td { border-bottom: none; }
.profile-table tr:hover td { background: rgba(255,255,255,0.02); }
.feat-mutable { color: var(--accent-emerald); font-weight: 600; }
.feat-immutable { color: var(--text-muted); }

/* ─── About Hero ─── */
.about-hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: var(--radius-xl);
    padding: 40px 48px;
    margin-bottom: 32px;
}
.about-hero h2 {
    font-size: 32px;
    font-weight: 800;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.about-hero p {
    color: var(--text-secondary);
    font-size: 16px;
    line-height: 1.7;
    max-width: 680px;
}

/* ─── Streamlit overrides ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-glass);
    border-radius: var(--radius-md);
    padding: 4px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #a5b4fc !important;
}
.stButton > button[kind="primary"] {
    background: var(--gradient-primary) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px;
    padding: 14px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.45) !important;
}
.stSelectbox > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
}
.stDataFrame {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    overflow: hidden;
}
hr { border-color: var(--border-subtle) !important; }
.stMarkdown a { color: var(--accent-indigo) !important; }

/* ─── Page Title ─── */
.page-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
}
.page-title .icon { margin-right: 12px; }
.page-subtitle {
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 28px;
    line-height: 1.6;
}

/* ─── Divider ─── */
.premium-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
    margin: 28px 0;
    border: none;
}

/* ─── Animated Background Orb ─── */
@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -20px) scale(1.05); }
    66% { transform: translate(-20px, 15px) scale(0.95); }
}
.bg-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.08;
    pointer-events: none;
    z-index: 0;
    animation: float 20s ease-in-out infinite;
}
.bg-orb-1 {
    width: 400px; height: 400px;
    background: var(--accent-indigo);
    top: -100px; right: -100px;
}
.bg-orb-2 {
    width: 300px; height: 300px;
    background: var(--accent-violet);
    bottom: -50px; left: -50px;
    animation-delay: -7s;
}

/* ─── Success Banner ─── */
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}
.success-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(52,211,153,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: var(--radius-lg);
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 16px 0;
    animation: slideDown 0.4s ease-out;
}
.success-banner .icon { font-size: 28px; }
.success-banner .text {
    font-size: 15px;
    font-weight: 600;
    color: #6ee7b7;
}
.success-banner .sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ─── Sparkline Bar ─── */
.spark-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
}
.spark-label {
    font-size: 12px;
    color: var(--text-secondary);
    width: 80px;
    flex-shrink: 0;
    text-align: right;
}
.spark-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.spark-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.spark-fill.indigo { background: var(--gradient-primary); }
.spark-fill.amber  { background: linear-gradient(90deg, #d97706, #f59e0b); }
.spark-fill.emerald { background: var(--gradient-success); }
.spark-fill.rose   { background: var(--gradient-danger); }
.spark-val {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-primary);
    width: 55px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* ─── Comparison Arrow ─── */
.comparison-arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px 0;
}
.comparison-arrow .arrow {
    font-size: 32px;
    color: var(--accent-emerald);
    animation: float 3s ease-in-out infinite;
}
.comparison-arrow .label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent-emerald);
    margin-top: 4px;
}

/* ─── Methodology Flow ─── */
.flow-container {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
    margin: 16px 0;
}
.flow-step {
    background: var(--gradient-glass);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    text-align: center;
    flex: 1;
    min-width: 130px;
    transition: all 0.3s ease;
}
.flow-step:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.flow-step .num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: var(--gradient-primary);
    border-radius: 50%;
    font-size: 12px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}
.flow-step .title {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.flow-step .desc {
    font-size: 10px;
    color: var(--text-muted);
    line-height: 1.4;
}
.flow-arrow {
    font-size: 18px;
    color: var(--text-muted);
    padding: 0 4px;
    flex-shrink: 0;
}

/* ─── Highlight Card (Benchmarks) ─── */
.highlight-card {
    background: var(--gradient-glass);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.highlight-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.highlight-card.best::after  { background: var(--gradient-success); }
.highlight-card.sig::after   { background: var(--gradient-primary); }
.highlight-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
}
.highlight-card .big {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
}
.highlight-card .big.emerald { color: var(--accent-emerald); }
.highlight-card .big.indigo  { color: var(--accent-indigo); }
.highlight-card .big.amber   { color: var(--accent-amber); }
.highlight-card .lbl {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

/* ─── Before/After Table ─── */
.ba-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    font-size: 13px;
    margin: 12px 0;
}
.ba-table th {
    background: rgba(99, 102, 241, 0.08);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 11px;
    padding: 12px 14px;
    text-align: left;
}
.ba-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-primary);
}
.ba-table tr:last-child td { border-bottom: none; }
.ba-table tr:hover td { background: rgba(255,255,255,0.02); }
.ba-table .delta-up   { color: #34d399; font-weight: 700; }
.ba-table .delta-down { color: #fb7185; font-weight: 700; }
.ba-table .delta-swap { color: #fbbf24; font-weight: 700; }

/* ─── Footer ─── */
.app-footer {
    margin-top: 48px;
    padding: 20px 0;
    border-top: 1px solid var(--border-subtle);
    text-align: center;
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.3px;
}
.app-footer a { color: var(--accent-indigo) !important; text-decoration: none; }
.app-footer .sep { margin: 0 8px; opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

# Floating background orbs
st.markdown(
    '<div class="bg-orb bg-orb-1"></div><div class="bg-orb bg-orb-2"></div>',
    unsafe_allow_html=True,
)


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


def _section(icon, title):
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


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
        arrow = {"up": "\u2191", "down": "\u2193", "swap": "\U0001f504"}[d]
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
    # ── Sidebar branding ──
    st.sidebar.markdown(
        '<div style="text-align:center;padding:16px 0 8px;">'
        '<div style="font-size:36px;">\u2696\ufe0f</div>'
        '<div style="font-size:22px;font-weight:800;letter-spacing:-0.02em;color:#fff !important;margin-top:4px;">FCAR</div>'
        '<div style="font-size:11px;color:#94a3b8 !important;letter-spacing:1.5px;text-transform:uppercase;">Fairness Constrained<br/>Actionable Recourse</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "\U0001f50d Interactive Recourse",
            "\U0001f4cb Fairness Audit",
            "\U0001f4ca Evaluation Metrics",
            "\u2139\ufe0f About & Methodology",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### \u2699\ufe0f Dataset")
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["german", "adult", "default_credit"],
        format_func=lambda x: {
            "german": "\U0001f1e9\U0001f1ea German Credit",
            "adult": "\U0001f1fa\U0001f1f8 Adult Income",
            "default_credit": "\U0001f1f9\U0001f1fc Default Credit",
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
        '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
        'border-radius:12px;padding:16px;text-align:center;">'
        '<div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#64748b !important;margin-bottom:8px;">Test Set Stats</div>'
        f'<div style="font-size:24px;font-weight:800;color:#f1f5f9 !important;">{len(rejected_indices)}</div>'
        f'<div style="font-size:12px;color:#94a3b8 !important;">rejected of {len(X_test)} ({rej_pct:.0%})</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="text-align:center;font-size:10px;color:#475569 !important;padding:8px;">'
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
        page_evaluation()
    elif "About" in page:
        page_about()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1 — Interactive Recourse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_recourse(dataset, pipe, config, X_train, X_test, A_test,
                  rejected_indices, proba, target_cls):
    ds_label = dataset.replace("_", " ").title()

    st.markdown(
        f'<div class="page-title"><span class="icon">\U0001f50d</span>'
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

    # ── Controls card ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1.5])
    with c1:
        selected_idx = int(
            st.selectbox(
                "\U0001f464 Select Applicant",
                options=rejected_indices,
                format_func=lambda x: f"Applicant #{x}",
            )
        )
    with c2:
        use_fcar = st.toggle(
            "\u2696\ufe0f Enable FCAR",
            value=False,
            help="Apply fairness-adjusted cost weights from Auto-FCAR tuning.",
        )
    with c3:
        if use_fcar and available_gcols:
            selected_gcol = st.selectbox(
                "\U0001f3af FCAR Group Attribute",
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
            '<div class="status-banner fcar">\u2696\ufe0f FCAR Mode \u2014 '
            'Fairness-constrained cost weights active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-banner ar">\U0001f4d0 Unconstrained AR \u2014 '
            'Standard cost minimization (no fairness adjustment)</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    _divider()

    # ── Applicant Profile ──
    applicant = X_test.iloc[selected_idx]
    sensitive_info = A_test.iloc[selected_idx]
    current_prob = proba[selected_idx]
    group_cols = get_sensitive_attributes(config)

    col_profile, col_score = st.columns([3, 1])
    with col_profile:
        _section("\U0001f464", "Applicant Profile")
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
                cls, tag, ic = "feat-mutable", "Mutable (Numeric)", "\U0001f527"
            elif feat in mutable_cat:
                cls, tag, ic = "feat-mutable", "Mutable (Categorical)", "\U0001f527"
            else:
                cls, tag, ic = "feat-immutable", "Immutable", "\U0001f512"
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
        _section("\U0001f4c9", "Model Decision")
        denied = _is_rejected(current_prob, target_cls)
        st.markdown(_score_ring(current_prob, denied), unsafe_allow_html=True)

    _divider()
    if st.button(
        "\U0001f680 Generate Optimal Recourse Plan",
        type="primary",
        use_container_width=True,
    ):
        fcar_gcol = selected_gcol if use_fcar else None
        _solve_and_render(
            selected_idx, applicant, current_prob, pipe, config,
            X_train, target_cls, use_fcar, dataset, sensitive_info,
            group_cols, fcar_gcol,
        )


# ──────────────────────────────────────────────────────────────

def _solve_and_render(test_idx, x0, p0, pipe, config, X_train,
                      target_cls, use_fcar, dataset, sensitive_info,
                      group_cols, fcar_gcol=None):
    import copy
    config = copy.deepcopy(config)           # isolate from cached object
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
                    '<span style="font-size:20px;">\u26a0\ufe0f</span>'
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
            f'<div class="status-banner fcar">\u2696\ufe0f FCAR weights '
            f'applied for: <b>{active_group}</b></div>',
            unsafe_allow_html=True,
        )
    elif use_fcar:
        st.markdown(
            '<div style="background:rgba(245,158,11,0.06);'
            'border:1px solid rgba(245,158,11,0.2);'
            'border-radius:var(--radius-md);padding:12px 18px;margin:8px 0;'
            'font-size:13px;color:#fcd34d;display:flex;align-items:center;gap:10px;">'
            '<span style="font-size:16px;">\u26a0\ufe0f</span>'
            'No FCAR overrides for this group \u2014 result is identical to unconstrained AR.'
            '</div>',
            unsafe_allow_html=True,
        )

    _section("\U0001f4c8", "Recourse Plan (UC-105)")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        _kpi("Score Before", f"{p0:.4f}", "red")
    with s2:
        _kpi("Score After", f"{p1:.4f}", "green" if flipped else "orange")
    with s3:
        _kpi(
            "Decision Flipped",
            "\u2705 YES" if flipped else "\u274c NO",
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
            "\u274c The solver could not find a valid recourse path within "
            "the plausibility constraints."
        )
        _footer()
        return

    # ── Success banner ──
    score_delta = abs(p1 - p0)
    st.markdown(
        '<div class="success-banner">'
        '<div class="icon">\U0001f389</div>'
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
                "Feature": c,
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
                "Feature": c,
                "Action": "\U0001f504 Change",
                "Before": v0,
                "After": v1,
                "\u0394 Amount": "\u2014",
                "Cost Weight": f"{cat_w.get(c, 0.25):.3f}",
                "_dir": "swap",
            })

    if not changes:
        st.warning("Decision flipped without any feature changes (edge case).")
        _footer()
        return

    # ── Change pills ──
    _section("\U0001f527", "Required Changes")
    pills = '<div class="change-pills">'
    for c in changes:
        d = c["_dir"]
        arrow = {"up": "\u2191", "down": "\u2193", "swap": "\U0001f504"}[d]
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
        if c[delta_key] != dash:
            action = c["Action"].split(" ", 1)[1]
            feat = c["Feature"]
            amt = c[delta_key]
            bef = c["Before"]
            aft = c["After"]
            parts.append(f"{action} {feat} by {amt} (from {bef} to {aft})")
        else:
            feat = c["Feature"]
            bef = c["Before"]
            aft = c["After"]
            parts.append(f"Change {feat} from {bef} to {aft}")
    narrative = (
        "To overturn the negative decision, the applicant must: "
        + "; ".join(parts) + "."
    )
    st.markdown(
        f'<div class="narrative-box">\U0001f4a1 <b>Explanation:</b><br/>'
        f'{narrative}</div>',
        unsafe_allow_html=True,
    )

    # ── AR vs FCAR side-by-side comparison (only when FCAR overrides applied) ──
    if applied_overrides and x_cf_ar is not None:
        _divider()
        _section("\U0001f50d", "AR vs FCAR Comparison")
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
                f'\U0001f4ca <b>FCAR redistributes the recourse burden</b> by adjusting cost weights. '
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
                '\u26a0\ufe0f For this specific applicant, AR and FCAR produce the same recourse path. '
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
        f'<div class="page-title"><span class="icon">\U0001f4cb</span>'
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
        "\U0001f3af Sensitive Attribute",
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
    _section("\U0001f4ca", "Key Metrics")
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

    _divider()

    # ── Side-by-side Audit Cards ──
    _section("\U0001f50e", "MISOB Audit Comparison (UC-103)")
    col_ar, col_arrow, col_fc = st.columns([5, 1, 5])

    for col, label, d, badge_cls in [
        (col_ar, "Unconstrained AR", ar, "badge-ar"),
        (col_fc, "FCAR (Auto-Tuned)", fc, "badge-fcar"),
    ]:
        with col:
            passed = d["audit"].get("passed", False)
            card_cls = "pass" if passed else "fail"
            status_badge = (
                '<span class="badge badge-pass">PASS \u2705</span>'
                if passed
                else '<span class="badge badge-fail">FAIL \u274c</span>'
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
            '<div class="arrow">\u27a1</div>'
            f'<div class="label">\u2193 {red_str}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Statistical Validation ──
    wil = stats.get("overall_wilcoxon", {})
    if wil:
        _divider()
        _section("\U0001f4d0", "Statistical Validation")
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
                "\u2705 Yes" if sig else "\u274c No",
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

def page_evaluation():
    st.markdown(
        '<div class="page-title"><span class="icon">\U0001f4ca</span>'
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
            sig_label = "\u2705 ALL" if all_sig else "\u26a0\ufe0f PARTIAL"
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
    _section("\U0001f4cb", "Results Summary")

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

    _footer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4 — About & Methodology
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_about():
    st.markdown(
        '<div class="about-hero">'
        '<h2>\u2696\ufe0f Fairness Constrained Actionable Recourse</h2>'
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
        _section("\U0001f3af", "Research Questions & Answers")
        for rq, q, a, status in [
            (
                "RQ1",
                "Can Social Burden be integrated into constrained optimization?",
                "Yes \u2014 MISOB metric drives asymmetric weight tuning in the MIP solver",
                "\u2705",
            ),
            (
                "RQ2",
                "Does FCAR reduce disparity across groups?",
                "Yes \u2014 45\u201364% gap reduction across all benchmarks, all p < 0.01",
                "\u2705",
            ),
            (
                "RQ3",
                "Does FCAR preserve utility?",
                "Yes \u2014 100% feasibility, 100% flip rate, lower average burden",
                "\u2705",
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

        _section("\U0001f504", "FCAR Pipeline")
        _methodology_flow()

        _section("\U0001f3d7\ufe0f", "System Architecture")
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
        _section("\U0001f4da", "Use Cases")
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
        _section("\U0001f4d6", "Key References")
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
        _section("\U0001f464", "Author")
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
        _section("\U0001f517", "Quick Links")
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
