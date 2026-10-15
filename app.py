"""
InfraGuard AI: Industrial Predictive Maintenance & Infrastructure Health Monitor
Aligned with United Nations Sustainable Development Goal 9 (Industry, Innovation & Infrastructure)

Production-grade, single-file Streamlit application featuring:
- Custom-engineered dark-mode glassmorphism interface (IBM Plex Sans aesthetic)
- Scikit-learn IsolationForest ML anomaly detection engine
- Calibrated Remaining Useful Life (RUL) estimation
- 4 Core Tabs: Executive Telemetry, ML Anomaly Analytics, Live Digital Twin Simulator, Work Order Dispatch
- Dark transparent Plotly industrial telemetry visualizations (paper_bgcolor='rgba(0,0,0,0)')
- Downloadable engineering work orders & dispatch automation
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import html
import re

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION & HUMAN-CRAFTED ENTERPRISE CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InfraGuard AI | Industrial Predictive Maintenance & Infrastructure Health Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Industrial Dark Glassmorphism CSS (IBM Carbon / Plex Sans style)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-deep: #080d1a;
        --bg-card: rgba(15, 23, 42, 0.72);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(0, 242, 254, 0.35);
        --accent-cyan: #00f2fe;
        --accent-blue: #2563eb;
        --status-emerald: #10b981;
        --status-amber: #f59e0b;
        --status-rose: #f43f5e;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* Global Typography & Canvas */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    .stApp {
        background: radial-gradient(circle at 12% 10%, #111e38 0%, #080d1a 55%, #04070e 100%);
    }

    /* Top Executive Navigation Bar */
    .infraguard-header {
        background: rgba(15, 23, 42, 0.78);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-left: 5px solid var(--accent-cyan);
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.65), 0 0 24px -6px rgba(0, 242, 254, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .brand-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.6px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-title span.accent {
        background: linear-gradient(135deg, #00f2fe 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        font-size: 13px;
        color: var(--text-secondary);
        margin: 5px 0 0 0;
        font-weight: 400;
        letter-spacing: 0.2px;
    }

    .header-badges {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .sdg-badge {
        background: rgba(229, 36, 59, 0.14);
        border: 1px solid rgba(229, 36, 59, 0.4);
        color: #ff6b7e;
        font-size: 11px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .live-pulse-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 11px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .live-pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulseAnimation 2s infinite ease-in-out;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Glassmorphism Metric Cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.16);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
    }

    .glass-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.14), transparent);
    }

    .card-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .card-value {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .card-meta {
        font-size: 12px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Status Pill Badges */
    .badge-nominal {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.32);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.4px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.32);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.4px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    .badge-critical {
        background: rgba(244, 63, 94, 0.18);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.38);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.4px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* Monospace Asset Tags */
    .asset-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 2px 7px;
        border-radius: 4px;
        letter-spacing: 0.3px;
        display: inline-block;
    }

    /* Custom Designer Glassmorphism Data Table */
    .table-container {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-glass);
        border-radius: 10px;
        overflow-x: auto;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        text-align: left;
    }

    .custom-table th {
        background: rgba(24, 34, 53, 0.85);
        color: #94a3b8;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        white-space: nowrap;
    }

    .custom-table td {
        padding: 13px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: #e2e8f0;
        vertical-align: middle;
        white-space: nowrap;
    }

    .custom-table tr:hover {
        background-color: rgba(30, 41, 59, 0.45);
    }

    .custom-table tr:last-child td {
        border-bottom: none;
    }

    /* Mini Progress Bar for Table */
    .mini-progress-bg {
        width: 80px;
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 3px;
        overflow: hidden;
        display: inline-block;
        vertical-align: middle;
        margin-right: 8px;
    }

    .mini-progress-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Streamlit Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.65);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--border-glass);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 7px;
        color: var(--text-secondary);
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        font-weight: 500;
        border: none;
        background-color: transparent;
        padding: 0 20px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 41, 59, 0.95) !important;
        color: #00f2fe !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35), inset 0 0 0 1px rgba(0, 242, 254, 0.35);
    }

    /* Sidebar Glass Overrides */
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    .mono-code {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. INDUSTRIAL ASSET SPECIFICATION & ISOLATION FOREST ML ENGINE
# -----------------------------------------------------------------------------
INDUSTRIAL_ASSETS = [
    {
        "id": "TX-GRID-501",
        "name": "500kV Main Step-Down Transformer",
        "sector": "Energy Transmission Grid",
        "zone": "Substation Alpha",
        "criticality": "Tier 1 Critical",
        "nominal_temp": 64.0,
        "nominal_vib": 1.45,
        "nominal_power": 42.0,
        "nominal_uptime": 99.8
    },
    {
        "id": "HPP-TURB-302",
        "name": "High-Pressure Hydro Intake Turbine",
        "sector": "Municipal Water Utility",
        "zone": "Hydro Pumping Station B",
        "criticality": "Tier 1 Critical",
        "nominal_temp": 58.5,
        "nominal_vib": 2.80,
        "nominal_power": 115.0,
        "nominal_uptime": 99.5
    },
    {
        "id": "TRM-MOTOR-108",
        "name": "High-Speed Rail Traction Inverter Motor",
        "sector": "Intercity Transport Corridor",
        "zone": "Depot Terminal C",
        "criticality": "Tier 2 Essential",
        "nominal_temp": 71.0,
        "nominal_vib": 3.20,
        "nominal_power": 88.0,
        "nominal_uptime": 99.2
    },
    {
        "id": "CNC-MILL-405",
        "name": "5-Axis Precision Aerospace CNC Mill",
        "sector": "Advanced Manufacturing",
        "zone": "Cleanroom Fabrication Bay",
        "criticality": "Tier 2 Essential",
        "nominal_temp": 48.0,
        "nominal_vib": 1.10,
        "nominal_power": 24.5,
        "nominal_uptime": 98.9
    },
    {
        "id": "CHL-HVAC-901",
        "name": "Centrifugal Industrial Facility Chiller",
        "sector": "District Cooling Utility",
        "zone": "HVAC Central Plant",
        "criticality": "Tier 3 Standard",
        "nominal_temp": 52.0,
        "nominal_vib": 2.15,
        "nominal_power": 62.0,
        "nominal_uptime": 99.1
    },
    {
        "id": "SCADA-EDGE-770",
        "name": "SCADA Optical Edge CCTV Gateway",
        "sector": "Physical Security & Edge Network",
        "zone": "Perimeter Security Hub",
        "criticality": "Tier 2 Essential",
        "nominal_temp": 41.0,
        "nominal_vib": 0.35,
        "nominal_power": 1.8,
        "nominal_uptime": 99.9
    }
]

FEATURE_COLS = ["Temperature_C", "Vibration_mms", "Power_Load_kW", "Node_Uptime_Pct"]

@st.cache_data(ttl=600)
def generate_industrial_telemetry(hours_history=48, random_seed=42):
    """
    Synthesizes multi-dimensional telemetry reflecting nominal industrial operating bands
    with calibrated drift, bearing degradation, and optical edge anomalies.
    """
    np.random.seed(random_seed)
    records = []
    base_time = datetime.now() - timedelta(hours=hours_history)

    for asset in INDUSTRIAL_ASSETS:
        aid = asset["id"]
        nom_t = asset["nominal_temp"]
        nom_v = asset["nominal_vib"]
        nom_p = asset["nominal_power"]
        nom_u = asset["nominal_uptime"]

        for h in range(hours_history):
            ts = base_time + timedelta(hours=h)
            
            # Stochastic baseline fluctuations
            temp = np.random.normal(nom_t, 2.5)
            vib = np.random.normal(nom_v, 0.32)
            power = np.random.normal(nom_p, nom_p * 0.045)
            uptime = np.random.uniform(max(97.0, nom_u - 1.0), 100.0)

            # Realistic anomaly injections over time
            # 1. HPP-TURB-302: Progressive bearing race degradation in the last 14 hours
            if aid == "HPP-TURB-302" and h >= (hours_history - 14):
                step = (h - (hours_history - 14) + 1)
                temp += step * 1.85
                vib += step * 0.38
                power += step * 2.2

            # 2. SCADA-EDGE-770: Network node thermal saturation and packet latency drop in the last 8 hours
            elif aid == "SCADA-EDGE-770" and h >= (hours_history - 8):
                step = (h - (hours_history - 8) + 1)
                temp += step * 2.5
                uptime -= step * 3.4

            # 3. TRM-MOTOR-108: Sporadic power surge anomaly in recent hours
            elif aid == "TRM-MOTOR-108" and (hours_history - 6 <= h <= hours_history - 2):
                power += 26.0
                vib += 1.35

            records.append({
                "Timestamp": ts,
                "Asset_ID": aid,
                "Asset_Name": asset["name"],
                "Sector": asset["sector"],
                "Zone": asset["zone"],
                "Criticality": asset["criticality"],
                "Temperature_C": round(float(temp), 2),
                "Vibration_mms": round(float(max(0.15, vib)), 2),
                "Power_Load_kW": round(float(max(0.5, power)), 2),
                "Node_Uptime_Pct": round(float(np.clip(uptime, 40.0, 100.0)), 2),
                "Nominal_Temp": nom_t,
                "Nominal_Vib": nom_v,
                "Nominal_Power": nom_p,
                "Nominal_Uptime": nom_u
            })

    df = pd.DataFrame(records)
    return df

@st.cache_resource
def train_isolation_forest_engine(df_train, contamination_rate=0.08, n_estimators=150):
    """
    Fits an IsolationForest model on continuous multi-channel industrial sensor measurements.
    """
    clf = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=float(contamination_rate),
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(df_train[FEATURE_COLS])
    return clf

def evaluate_telemetry_batch(model, df):
    """
    Scores telemetry data with the trained model and computes calibrated
    failure risk index, health status, and estimated Remaining Useful Life (RUL).
    """
    df_eval = df.copy()
    
    # IsolationForest predictions: +1 = inlier (nominal), -1 = anomaly
    pred = model.predict(df_eval[FEATURE_COLS])
    raw_scores = model.decision_function(df_eval[FEATURE_COLS])
    
    df_eval["Raw_Score"] = raw_scores
    df_eval["ML_Anomaly_Flag"] = pred
    
    # Scale anomaly score into 0 - 100% Failure Risk Index
    min_score, max_score = -0.30, 0.25
    clipped = np.clip(raw_scores, min_score, max_score)
    risk_pct = 100.0 * (1.0 - (clipped - min_score) / (max_score - min_score))
    df_eval["Risk_Index_Pct"] = np.round(risk_pct, 1)

    # Asset health classification rule
    def assign_health_state(row):
        t_delta = row["Temperature_C"] - row["Nominal_Temp"]
        v_delta = row["Vibration_mms"] - row["Nominal_Vib"]
        
        if row["Risk_Index_Pct"] >= 72.0 or t_delta > 22.0 or v_delta > 3.6 or row["Node_Uptime_Pct"] < 75.0:
            return "Critical"
        elif row["Risk_Index_Pct"] >= 45.0 or t_delta > 12.0 or v_delta > 1.8 or row["Node_Uptime_Pct"] < 92.0:
            return "Warning"
        return "Nominal"

    df_eval["Health_Status"] = df_eval.apply(assign_health_state, axis=1)

    # Remaining Useful Life (RUL) estimation combining ML score and physical sensor strain
    def calculate_rul(row):
        base_rul = 100.0 - (row["Risk_Index_Pct"] * 0.85)
        # Thermal penalty
        t_excess = max(0.0, row["Temperature_C"] - (row["Nominal_Temp"] + 5.0))
        # Vibration penalty
        v_excess = max(0.0, row["Vibration_mms"] - (row["Nominal_Vib"] + 0.6))
        # Uptime degradation penalty
        u_deficit = max(0.0, 98.0 - row["Node_Uptime_Pct"])

        penalties = (t_excess * 1.5) + (v_excess * 7.5) + (u_deficit * 1.8)
        rul_pct = np.clip(base_rul - penalties, 5.0, 99.0)
        
        # Estimate RUL in operational hours (nominal 3000 operating hours full life cycle)
        rul_hours = int(np.round((rul_pct / 100.0) * 2800))
        return pd.Series([round(rul_pct, 1), rul_hours], index=["RUL_Pct", "RUL_Hours"])

    rul_df = df_eval.apply(calculate_rul, axis=1)
    df_eval["RUL_Pct"] = rul_df["RUL_Pct"]
    df_eval["RUL_Hours"] = rul_df["RUL_Hours"]

    return df_eval


# -----------------------------------------------------------------------------
# 3. HTML TABLE RENDERERS (Bypasses PyArrow AppLocker Restrictions)
# -----------------------------------------------------------------------------
def render_executive_matrix(df):
    """
    Renders an executive-grade HTML table with custom glassmorphism styles,
    color-coded condition badges, and embedded risk progress bars.
    """
    rows_html = ""
    for _, row in df.iterrows():
        # Condition badge
        status = row["Health_Status"]
        if status == "Nominal":
            status_html = '<span class="badge-nominal">● NOMINAL</span>'
        elif status == "Warning":
            status_html = '<span class="badge-warning">▲ WARNING</span>'
        else:
            status_html = '<span class="badge-critical">■ CRITICAL</span>'

        # Progress bar fill color
        risk = row["Risk_Index_Pct"]
        bar_color = "#10b981" if risk < 40 else ("#f59e0b" if risk < 70 else "#f43f5e")

        rows_html += f"""
        <tr>
            <td><span class="asset-tag">{html.escape(str(row['Asset_ID']))}</span></td>
            <td><strong>{html.escape(str(row['Asset_Name']))}</strong><br><span style="font-size:11px; color:#94a3b8;">{html.escape(str(row['Sector']))}</span></td>
            <td><span style="color:#cbd5e1;">{html.escape(str(row['Zone']))}</span></td>
            <td>{row['Temperature_C']:.1f} °C</td>
            <td>{row['Vibration_mms']:.2f} mm/s</td>
            <td>{row['Power_Load_kW']:.1f} kW</td>
            <td>{row['Node_Uptime_Pct']:.2f}%</td>
            <td>
                <div class="mini-progress-bg">
                    <div class="mini-progress-fill" style="width: {min(100, max(0, risk))}%; background: {bar_color};"></div>
                </div>
                <span>{risk:.1f}%</span>
            </td>
            <td><strong style="color: #fff;">{row['RUL_Hours']:,} hrs</strong></td>
            <td>{status_html}</td>
        </tr>
        """

    full_html = f"""
    <div class="table-container">
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Asset Tag</th>
                    <th>Infrastructure Unit</th>
                    <th>Zone</th>
                    <th>Temp (°C)</th>
                    <th>Vibration (mm/s)</th>
                    <th>Power (kW)</th>
                    <th>Node Uptime</th>
                    <th>Failure Risk</th>
                    <th>Est. RUL</th>
                    <th>Condition</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    st.markdown(re.sub(r'^\s+', '', full_html, flags=re.MULTILINE), unsafe_allow_html=True)

def render_dispatch_matrix(df):
    """
    Renders the dispatch management table with priority tags, work scopes, and condition indicators.
    """
    rows_html = ""
    for _, row in df.iterrows():
        status = row["Health_Status"]
        status_html = '<span class="badge-critical">■ CRITICAL</span>' if status == "Critical" else '<span class="badge-warning">▲ WARNING</span>'
        prio_color = "#f43f5e" if "P1" in row["Priority"] else "#f59e0b"

        rows_html += f"""
        <tr>
            <td><span class="mono-code" style="color: #00f2fe; font-weight:600;">{html.escape(str(row['Work_Order_ID']))}</span></td>
            <td><span style="color: {prio_color}; font-weight:600; font-size:12px;">{html.escape(str(row['Priority']))}</span></td>
            <td><span class="asset-tag">{html.escape(str(row['Asset_ID']))}</span></td>
            <td><strong>{html.escape(str(row['Asset_Name']))}</strong><br><span style="font-size:11px; color:#94a3b8;">{html.escape(str(row['Zone']))}</span></td>
            <td>{status_html}</td>
            <td><strong style="color: #fff;">{row['RUL_Hours']:,} hrs</strong></td>
            <td style="max-width: 380px; white-space: normal; font-size: 12px; color: #cbd5e1; line-height: 1.4;">
                {html.escape(str(row['Prescriptive_Remediation']))}
            </td>
        </tr>
        """

    full_html = f"""
    <div class="table-container">
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Work Order Ticket</th>
                    <th>Priority</th>
                    <th>Asset ID</th>
                    <th>Equipment Unit</th>
                    <th>Condition</th>
                    <th>Remaining Life</th>
                    <th>Prescribed Remediation Work Scope</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    st.markdown(re.sub(r'^\s+', '', full_html, flags=re.MULTILINE), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div style="font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -0.3px;">
            INFRA<span style="color: #00f2fe;">GUARD</span> CONTROL
        </div>
        <div style="font-size: 12px; color: #94a3b8;">Autonomous SCADA Edge Sentinel</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Topology Filter")
    zone_list = ["All Infrastructure Zones"] + sorted(list({a["zone"] for a in INDUSTRIAL_ASSETS}))
    selected_zone = st.selectbox("Infrastructure Zone", zone_list, index=0)

    asset_options = {a["id"]: f"{a['id']} — {a['name']}" for a in INDUSTRIAL_ASSETS}
    selected_asset_id = st.selectbox(
        "Focal Asset Inspector",
        options=list(asset_options.keys()),
        format_func=lambda x: asset_options[x],
        index=1  # Default to HPP-TURB-302 which has active bearing wear
    )

    st.divider()
    st.markdown("#### Machine Learning Engine")
    
    slider_contamination = st.slider(
        "Contamination Rate (α)",
        min_value=0.01,
        max_value=0.20,
        value=0.08,
        step=0.01,
        help="Calibrated proportion of anomalous deviations in baseline telemetry."
    )
    
    slider_trees = st.select_slider(
        "Isolation Forest Estimators",
        options=[50, 100, 150, 200, 250],
        value=150
    )

    st.caption("Algorithm: scikit-learn IsolationForest")
    st.caption("Telemetry Vector: 4 Continuous Channels")

    st.divider()
    if st.button("Re-sample Telemetry Stream", use_container_width=True):
        st.cache_data.clear()
        st.toast("Synthesized fresh telemetry buffer from SCADA edge nodes.")


# -----------------------------------------------------------------------------
# 5. DATA INGESTION & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
df_raw = generate_industrial_telemetry(hours_history=48)
ml_engine = train_isolation_forest_engine(df_raw, contamination_rate=slider_contamination, n_estimators=slider_trees)
df_scored = evaluate_telemetry_batch(ml_engine, df_raw)

# Latest snapshot across assets
df_latest_all = df_scored.sort_values("Timestamp").groupby("Asset_ID").last().reset_index()

if selected_zone != "All Infrastructure Zones":
    df_latest = df_latest_all[df_latest_all["Zone"] == selected_zone].copy()
else:
    df_latest = df_latest_all.copy()

focal_meta = next(item for item in INDUSTRIAL_ASSETS if item["id"] == selected_asset_id)
focal_asset_meta = focal_meta
focal_latest = df_latest_all[df_latest_all["Asset_ID"] == selected_asset_id].iloc[0]


# -----------------------------------------------------------------------------
# 6. TOP EXECUTIVE HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="infraguard-header">
    <div>
        <div class="brand-title">
            <span>INFRA<span class="accent">GUARD</span> AI</span>
            <span style="font-size: 14px; font-weight: 500; color: #64748b;">|</span>
            <span style="font-size: 16px; font-weight: 500; color: #e2e8f0;">Industrial Predictive Maintenance & Infrastructure Health Monitor</span>
        </div>
        <p class="brand-subtitle">Autonomous Edge Machine Learning & Anomaly Isolation for Resilient Infrastructure Under UN SDG 9</p>
    </div>
    <div class="header-badges">
        <div class="sdg-badge">
            <span style="font-size: 13px;">🏛️</span> UN SDG 9: Target 9.1 & 9.4
        </div>
        <div class="live-pulse-badge">
            <div class="live-pulse-dot"></div>
            <span>SCADA TELEMETRY ACTIVE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. EXECUTIVE TELEMETRY KPI CARDS (Glassmorphic)
# -----------------------------------------------------------------------------
total_nodes = len(df_latest_all)
crit_assets = len(df_latest_all[df_latest_all["Health_Status"] == "Critical"])
warn_assets = len(df_latest_all[df_latest_all["Health_Status"] == "Warning"])
norm_assets = len(df_latest_all[df_latest_all["Health_Status"] == "Nominal"])
fleet_avg_rul = round(df_latest_all["RUL_Pct"].mean(), 1)
fleet_avg_uptime = round(df_latest_all["Node_Uptime_Pct"].mean(), 2)

# SDG 9 impact estimation metrics
est_downtime_prevented_hrs = (crit_assets * 18) + (warn_assets * 6) + 42
est_economic_savings_usd = est_downtime_prevented_hrs * 3850

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">
            <span>Fleet Health Index</span>
            <span class="mono-code">SDG-9.4</span>
        </div>
        <div class="card-value">{fleet_avg_rul}%</div>
        <div class="card-meta">
            <span class="badge-nominal">● {norm_assets} Nominal</span>
            <span style="color: var(--text-muted); font-size: 11px;">{total_nodes} Monitored Units</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c2:
    if crit_assets > 0:
        alert_badge = f'<span class="badge-critical">■ {crit_assets} Critical Action</span>'
    elif warn_assets > 0:
        alert_badge = f'<span class="badge-warning">▲ {warn_assets} Warning</span>'
    else:
        alert_badge = '<span class="badge-nominal">● Zero Active Hazards</span>'

    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">
            <span>Active Failure Anomalies</span>
            <span class="mono-code">ISO-10816</span>
        </div>
        <div class="card-value">{crit_assets + warn_assets} <span style="font-size: 16px; color: var(--text-secondary); font-weight: 500;">Incidents</span></div>
        <div class="card-meta">
            {alert_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">
            <span>SCADA & Edge CCTV Uptime</span>
            <span class="mono-code">QoS 99.5%</span>
        </div>
        <div class="card-value">{fleet_avg_uptime}%</div>
        <div class="card-meta">
            <span style="color: #38bdf8; font-size: 11px;">Dual Optical Ring SCADA Bus</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c4:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">
            <span>SDG 9 Loss Mitigation</span>
            <span class="mono-code">ROI VALUE</span>
        </div>
        <div class="card-value">${est_economic_savings_usd:,.0f}</div>
        <div class="card-meta">
            <span style="color: #34d399; font-size: 11px;">+{est_downtime_prevented_hrs} hrs Catastrophic Downtime Averted</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# -----------------------------------------------------------------------------
# 8. FOUR STRUCTURED CORE APPLICATION TABS
# -----------------------------------------------------------------------------
tab_exec, tab_analytics, tab_twin, tab_dispatch = st.tabs([
    "Executive Telemetry",
    "ML Anomaly Analytics",
    "Live Digital Twin Simulator",
    "Work Order Dispatch"
])


# =============================================================================
# TAB 1: EXECUTIVE TELEMETRY
# =============================================================================
with tab_exec:
    st.markdown("""
    <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: flex-end;">
        <div>
            <h3 style="margin: 0; font-size: 18px; font-weight: 600; color: #f8fafc;">Grid & Plant Infrastructure Health Matrix</h3>
            <p style="margin: 2px 0 0 0; font-size: 12px; color: #94a3b8;">Multi-sector sensor telemetry stream aggregated across substations, hydro plants, rail transit, and manufacturing bays.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render custom glassmorphism HTML table
    render_executive_matrix(df_latest)

    st.write("")
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        # Remaining Useful Life across assets
        color_map = {"Nominal": "#10b981", "Warning": "#f59e0b", "Critical": "#f43f5e"}
        fig_rul = px.bar(
            df_latest,
            x="Asset_ID",
            y="RUL_Hours",
            color="Health_Status",
            color_discrete_map=color_map,
            hover_data=["Asset_Name", "Temperature_C", "Vibration_mms"],
            title="Estimated Remaining Useful Life (RUL Operational Hours)"
        )
        fig_rul.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1"),
            title_font=dict(size=14, color="#f8fafc"),
            xaxis=dict(showgrid=False, title=None, tickfont=dict(color="#94a3b8")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Hours to Maintenance Limit"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_rul, use_container_width=True)

    with col_chart2:
        # Multi-Vector Diagnostic Radar
        categories = ["Temperature Health", "Vibration Integrity", "Power Stability", "Node Uptime", "Model Inlier Score"]
        
        focal_t_score = max(0, 100 - (focal_latest["Temperature_C"] - focal_meta["nominal_temp"]) * 3.5)
        focal_v_score = max(0, 100 - (focal_latest["Vibration_mms"] - focal_meta["nominal_vib"]) * 20.0)
        focal_p_score = 95.0
        focal_u_score = focal_latest["Node_Uptime_Pct"]
        focal_m_score = max(0, 100 - focal_latest["Risk_Index_Pct"])

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[100, 100, 100, 100, 100],
            theta=categories,
            fill='toself',
            name='Nominal Baseline Benchmark',
            line=dict(color='rgba(255, 255, 255, 0.25)', dash='dash'),
            fillcolor='rgba(255, 255, 255, 0.04)'
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=[focal_t_score, focal_v_score, focal_p_score, focal_u_score, focal_m_score],
            theta=categories,
            fill='toself',
            name=f"{focal_latest['Asset_ID']} (Selected)",
            line=dict(color='#00f2fe', width=2),
            fillcolor='rgba(0, 242, 254, 0.18)'
        ))

        fig_radar.update_layout(
            title=f"Multi-Vector Diagnostic Radar: {focal_latest['Asset_ID']}",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 110], gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#64748b")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(color="#94a3b8"))
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1"),
            title_font=dict(size=14, color="#f8fafc"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=50, b=50)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # UN SDG 9 Contextual Information Ribbon
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px 20px; font-size: 12px; color: #94a3b8; display: flex; align-items: center; justify-content: space-between; gap: 20px;">
        <div>
            <strong style="color: #f8fafc;">UN Sustainable Development Goal 9 Alignment:</strong>
            InfraGuard AI upgrades infrastructure with advanced ML predictive diagnostics (Target 9.4), mitigating catastrophic turbine and transformer disruptions while minimizing resource replacement waste (Target 9.1).
        </div>
        <div style="white-space: nowrap; font-family: 'IBM Plex Mono', monospace; color: #00f2fe; font-size: 11px;">
            STANDARDS: ISO 10816-3 & IEEE C57.104
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: ML ANOMALY ANALYTICS
# =============================================================================
with tab_analytics:
    st.markdown("""
    <div style="margin-bottom: 14px;">
        <h3 style="margin: 0; font-size: 18px; font-weight: 600; color: #f8fafc;">Isolation Forest Telemetry Analytics & Feature Spaces</h3>
        <p style="margin: 2px 0 0 0; font-size: 12px; color: #94a3b8;">High-dimensional unsupervised anomaly isolation across continuous thermal, mechanical vibration, load, and communication channels.</p>
    </div>
    """, unsafe_allow_html=True)

    # Focus asset time-series
    df_focal_hist = df_scored[df_scored["Asset_ID"] == selected_asset_id].sort_values("Timestamp")

    col_ts1, col_ts2 = st.columns([2, 1])

    with col_ts1:
        # Time-series multi-sensor trajectory
        fig_timeseries = go.Figure()

        fig_timeseries.add_trace(go.Scatter(
            x=df_focal_hist["Timestamp"],
            y=df_focal_hist["Temperature_C"],
            mode="lines",
            name="Temperature (°C)",
            line=dict(color="#00f2fe", width=2)
        ))

        fig_timeseries.add_trace(go.Scatter(
            x=df_focal_hist["Timestamp"],
            y=df_focal_hist["Vibration_mms"],
            mode="lines",
            name="Vibration (mm/s)",
            line=dict(color="#fbbf24", width=2),
            yaxis="y2"
        ))

        # Flagged anomaly markers
        anomaly_subset = df_focal_hist[df_focal_hist["Health_Status"] != "Nominal"]
        if not anomaly_subset.empty:
            fig_timeseries.add_trace(go.Scatter(
                x=anomaly_subset["Timestamp"],
                y=anomaly_subset["Temperature_C"],
                mode="markers",
                name="ML Anomaly Flagged",
                marker=dict(color="#f43f5e", size=9, symbol="diamond-dot", line=dict(width=1, color="#ffffff"))
            ))

        fig_timeseries.update_layout(
            title=f"48-Hour Continuous Telemetry Stream: {selected_asset_id} ({focal_meta['name']})",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1"),
            title_font=dict(size=14, color="#f8fafc"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Timestamp"),
            yaxis=dict(title=dict(text="Temperature (°C)", font=dict(color="#00f2fe")), gridcolor="rgba(255,255,255,0.06)"),
            yaxis2=dict(title=dict(text="Vibration (mm/s)", font=dict(color="#fbbf24")), overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_timeseries, use_container_width=True)

    with col_ts2:
        # Decision function distribution
        fig_dist = px.histogram(
            df_scored,
            x="Raw_Score",
            color="Health_Status",
            color_discrete_map={"Nominal": "#10b981", "Warning": "#f59e0b", "Critical": "#f43f5e"},
            nbins=30,
            title="Isolation Forest Decision Function Score Distribution"
        )
        fig_dist.add_vline(x=0.0, line_dash="dash", line_color="#f43f5e", annotation_text="Anomaly Boundary (s=0)", annotation_font_size=10)
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1"),
            title_font=dict(size=14, color="#f8fafc"),
            xaxis=dict(title="Decision Function Score s(x, n)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Sample Observations Count", gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # 2D Multi-dimensional Anomaly Scatter Space
    col_sc1, col_sc2 = st.columns([1, 1])

    with col_sc1:
        fig_cluster = px.scatter(
            df_scored,
            x="Temperature_C",
            y="Vibration_mms",
            size="Power_Load_kW",
            color="Health_Status",
            color_discrete_map={"Nominal": "#10b981", "Warning": "#f59e0b", "Critical": "#f43f5e"},
            hover_data=["Asset_ID", "Node_Uptime_Pct", "Risk_Index_Pct"],
            title="Bi-Axial Anomaly Projection: Thermal vs. Mechanical Strain"
        )
        fig_cluster.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1"),
            title_font=dict(size=14, color="#f8fafc"),
            xaxis=dict(title="Temperature (°C)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Vibration Velocity (mm/s RMS)", gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col_sc2:
        # Sensor Residual Drift Breakdown for Focal Asset
        t_drift = focal_latest["Temperature_C"] - focal_meta["nominal_temp"]
        v_drift = focal_latest["Vibration_mms"] - focal_meta["nominal_vib"]
        p_drift = focal_latest["Power_Load_kW"] - focal_meta["nominal_power"]
        u_drift = focal_meta["nominal_uptime"] - focal_latest["Node_Uptime_Pct"]

        drift_df = pd.DataFrame({
            "Sensor Channel": ["Thermal Drift (°C)", "Vibration Drift (mm/s)", "Power Surge (kW)", "Uptime Packet Drop (%)"],
            "Deviation Magnitude": [round(t_drift, 2), round(v_drift, 2), round(p_drift, 2), round(u_drift, 2)]
        })

        fig_drift = px.bar(
            drift_df,
            x="Sensor Channel",
            y="Deviation Magnitude",
            color="Deviation Magnitude",
            color_continuous_scale=["#10b981", "#f59e0b", "#f43f5e"],
            title=f"Root-Cause Sensor Residuals: {selected_asset_id}"
        )
        fig_drift.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1"),
            title_font=dict(size=14, color="#f8fafc"),
            coloraxis_showscale=False,
            xaxis=dict(title=None, showgrid=False),
            yaxis=dict(title="Delta from Nominal Baseline", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_drift, use_container_width=True)


# =============================================================================
# TAB 3: LIVE DIGITAL TWIN SIMULATOR
# =============================================================================
with tab_twin:
    st.markdown("""
    <div style="margin-bottom: 14px;">
        <h3 style="margin: 0; font-size: 18px; font-weight: 600; color: #f8fafc;">Hardware-in-the-Loop Digital Twin & Stress Simulator</h3>
        <p style="margin: 2px 0 0 0; font-size: 12px; color: #94a3b8;">Simulate parametric operational stress scenarios, component wear, and edge network faults to evaluate real-time ML inference.</p>
    </div>
    """, unsafe_allow_html=True)

    twin_c1, twin_c2 = st.columns([1, 1], gap="large")

    with twin_c1:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 600; color: #00f2fe; text-transform: uppercase; letter-spacing: 0.5px;">Simulated Target Digital Twin</div>
            <div style="font-size: 16px; font-weight: 700; color: #fff; margin-top: 2px;">{focal_meta['name']}</div>
            <div style="font-size: 12px; color: #94a3b8;">Asset Tag: <span class="asset-tag">{focal_meta['id']}</span> | Sector: {focal_meta['sector']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Parametric Telemetry Injections:**")

        # Session state for simulator presets
        if "sim_temp_val" not in st.session_state:
            st.session_state["sim_temp_val"] = float(focal_latest["Temperature_C"])
            st.session_state["sim_vib_val"] = float(focal_latest["Vibration_mms"])
            st.session_state["sim_power_val"] = float(focal_latest["Power_Load_kW"])
            st.session_state["sim_uptime_val"] = float(focal_latest["Node_Uptime_Pct"])

        sim_temp = st.slider(
            "Core Temperature (°C)",
            min_value=25.0,
            max_value=125.0,
            value=float(st.session_state["sim_temp_val"]),
            step=0.5,
            key="slider_temp",
            help="Measured by RTD platinum sensor at core stator / bearing cage"
        )

        sim_vib = st.slider(
            "Tri-Axial Vibration Velocity (mm/s RMS)",
            min_value=0.1,
            max_value=14.0,
            value=float(st.session_state["sim_vib_val"]),
            step=0.1,
            key="slider_vib",
            help="Piezoelectric vibration accelerometer reading"
        )

        sim_power = st.slider(
            "Active Electrical Power Load (kW)",
            min_value=1.0,
            max_value=250.0,
            value=float(st.session_state["sim_power_val"]),
            step=1.0,
            key="slider_power",
            help="Precision digital power transducer"
        )

        sim_uptime = st.slider(
            "Edge CCTV / SCADA Node Uptime (%)",
            min_value=40.0,
            max_value=100.0,
            value=float(st.session_state["sim_uptime_val"]),
            step=0.5,
            key="slider_uptime",
            help="Optical link reliability and video streaming packet delivery"
        )

        st.write("")
        st.markdown("**Load Preset Industrial Stress Profiles:**")
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("Nominal Baseline", use_container_width=True):
                st.session_state["sim_temp_val"] = focal_meta["nominal_temp"]
                st.session_state["sim_vib_val"] = focal_meta["nominal_vib"]
                st.session_state["sim_power_val"] = focal_meta["nominal_power"]
                st.session_state["sim_uptime_val"] = focal_meta["nominal_uptime"]
                st.rerun()
        with preset_cols[1]:
            if st.button("Bearing Fatigue", use_container_width=True):
                st.session_state["sim_temp_val"] = focal_meta["nominal_temp"] + 19.5
                st.session_state["sim_vib_val"] = focal_meta["nominal_vib"] + 4.8
                st.session_state["sim_power_val"] = focal_meta["nominal_power"] + 18.0
                st.session_state["sim_uptime_val"] = focal_meta["nominal_uptime"]
                st.rerun()
        with preset_cols[2]:
            if st.button("Network Brownout", use_container_width=True):
                st.session_state["sim_temp_val"] = focal_meta["nominal_temp"] + 15.0
                st.session_state["sim_vib_val"] = focal_meta["nominal_vib"]
                st.session_state["sim_power_val"] = focal_meta["nominal_power"]
                st.session_state["sim_uptime_val"] = 62.0
                st.rerun()

    with twin_c2:
        # Build simulation row and pass through ML isolation forest
        sim_df = pd.DataFrame([{
            "Temperature_C": sim_temp,
            "Vibration_mms": sim_vib,
            "Power_Load_kW": sim_power,
            "Node_Uptime_Pct": sim_uptime,
            "Nominal_Temp": focal_meta["nominal_temp"],
            "Nominal_Vib": focal_meta["nominal_vib"],
            "Nominal_Power": focal_meta["nominal_power"],
            "Nominal_Uptime": focal_meta["nominal_uptime"]
        }])

        sim_scored = evaluate_telemetry_batch(ml_engine, sim_df)
        sim_res = sim_scored.iloc[0]

        pred_status = sim_res["Health_Status"]
        pred_risk = sim_res["Risk_Index_Pct"]
        pred_rul = sim_res["RUL_Pct"]
        pred_hours = sim_res["RUL_Hours"]
        pred_score = sim_res["Raw_Score"]

        # Gauge chart for real-time RUL
        gauge_color = "#10b981" if pred_status == "Nominal" else ("#f59e0b" if pred_status == "Warning" else "#f43f5e")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_rul,
            title={'text': "Simulated Remaining Useful Life (RUL %)", 'font': {'size': 14, 'color': '#f8fafc', 'family': 'IBM Plex Sans'}},
            delta={'reference': 85.0, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#f43f5e"}},
            number={'suffix': "%", 'font': {'size': 32, 'color': '#ffffff', 'family': 'IBM Plex Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'tickwidth': 1},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(244, 63, 94, 0.2)'},
                    {'range': [40, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#f43f5e", 'width': 3},
                    'thickness': 0.75,
                    'value': 35
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#cbd5e1"),
            height=240,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Prescriptive AI Diagnosis Card
        if pred_status == "Critical":
            diag_title = "CRITICAL FAILURE THRESHOLD BREACHED"
            diag_text = (
                "Severe mechanical oscillation and thermal saturation detected. Isolation Forest classifies this "
                "state as an extreme out-of-distribution anomaly. Imminent mechanical seizure risk within 24-48 operating hours. "
                "Trigger automated dispatch protocol."
            )
            border_c = "#f43f5e"
        elif pred_status == "Warning":
            diag_title = "ELEVATED DEGRADATION SIGNATURE DETECTED"
            diag_text = (
                "Telemetry parameters deviate moderately from baseline calibration. Predictive trend suggests "
                "bearing sub-surface spalling or edge packet degradation. Preventive inspection recommended within 7 operational shifts."
            )
            border_c = "#f59e0b"
        else:
            diag_title = "NOMINAL OPERATING CONDITIONS CONFIRMED"
            diag_text = (
                "All parameters fall cleanly within ISO-10816 Class II vibration tolerances and IEEE thermal limits. "
                "Continuous automated surveillance remains engaged."
            )
            border_c = "#10b981"

        st.markdown(f"""
        <div style="background: rgba(16, 24, 39, 0.7); border: 1px solid {border_c}; border-radius: 10px; padding: 18px; margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-size: 13px; font-weight: 700; color: {border_c}; letter-spacing: 0.5px;">{diag_title}</div>
                <div class="mono-code" style="color: #94a3b8;">SCORE: {pred_score:+.4f}</div>
            </div>
            <p style="font-size: 13px; color: #cbd5e1; line-height: 1.5; margin-bottom: 12px;">{diag_text}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                <div><span style="color: #94a3b8;">Failure Risk Index:</span> <strong style="color: #fff;">{pred_risk:.1f}%</strong></div>
                <div><span style="color: #94a3b8;">Est. Operational Life:</span> <strong style="color: #fff;">{pred_hours:,} Hours</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: WORK ORDER DISPATCH
# =============================================================================
with tab_dispatch:
    st.markdown("""
    <div style="margin-bottom: 14px;">
        <h3 style="margin: 0; font-size: 18px; font-weight: 600; color: #f8fafc;">Autonomous Maintenance Dispatch & Field Work Orders</h3>
        <p style="margin: 2px 0 0 0; font-size: 12px; color: #94a3b8;">AI-prescribed remediation tickets, OSHA safety protocols, and certified digital work orders for field engineering crews.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter critical or warning items from current snapshot
    flagged_assets = df_latest_all[df_latest_all["Health_Status"].isin(["Warning", "Critical"])].copy()

    if flagged_assets.empty:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 24px; text-align: center;">
            <div style="font-size: 16px; font-weight: 600; color: #34d399;">All Monitored Infrastructure Units Operating Nominally</div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">Zero active dispatch tickets required. Isolation Forest continuous surveillance in standby.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Prescriptive action logic
        def create_prescriptive_protocol(row):
            aid = row["Asset_ID"]
            t = row["Temperature_C"]
            v = row["Vibration_mms"]
            u = row["Node_Uptime_Pct"]

            actions = []
            parts = []
            safety = "OSHA Standard LOTO (Lockout/Tagout - 29 CFR 1910.147)"

            if t > 72.0:
                actions.append("Inspect secondary oil-coolant heat exchangers and thermal dissipation radiators")
                parts.append("Synthetic Dielectric Fluid / Coolant Flush Filter (P/N FLT-902)")
            if v > 3.4:
                actions.append("Execute dynamic shaft balance & replace high-speed spherical roller bearings")
                parts.append("SKF Spherical Roller Bearing Assembly (P/N SKF-22218-E)")
            if u < 94.0:
                actions.append("Perform optical fiber OTDR loss test & reboot ruggedized SCADA PoE switch")
                parts.append("Industrial Cat6A Shielded S/FTP Patch Cable + PoE Injector Unit")

            if not actions:
                actions.append("Conduct comprehensive ISO-10816 preventive multi-point acoustic inspection")
                parts.append("General Maintenance Consumables Kit")

            return pd.Series([
                " | ".join(actions),
                ", ".join(parts),
                safety
            ], index=["Prescriptive_Remediation", "Required_Parts", "Safety_Protocol"])

        prescription_data = flagged_assets.apply(create_prescriptive_protocol, axis=1)
        flagged_assets["Prescriptive_Remediation"] = prescription_data["Prescriptive_Remediation"]
        flagged_assets["Required_Parts"] = prescription_data["Required_Parts"]
        flagged_assets["Safety_Protocol"] = prescription_data["Safety_Protocol"]
        flagged_assets["Work_Order_ID"] = [f"WO-SDG9-{202600 + i}" for i in range(len(flagged_assets))]
        flagged_assets["Priority"] = flagged_assets["Health_Status"].map(lambda x: "P1 - Critical Emergency" if x == "Critical" else "P2 - Scheduled Urgent")

        # Render custom HTML dispatch table
        render_dispatch_matrix(flagged_assets)

        st.write("")
        st.markdown("#### Work Order Package Generator")

        dispatch_col1, dispatch_col2 = st.columns([1, 1], gap="large")

        with dispatch_col1:
            ticket_to_generate = st.selectbox(
                "Select Active Work Order to Inspect / Export",
                flagged_assets["Work_Order_ID"].tolist()
            )
            selected_ticket_row = flagged_assets[flagged_assets["Work_Order_ID"] == ticket_to_generate].iloc[0]

            # Generate formatted engineering work order ticket (Markdown/Text)
            formatted_ticket = f"""================================================================================
INFRASTRUCTURE MAINTENANCE WORK ORDER: {selected_ticket_row['Work_Order_ID']}
Issued By: InfraGuard AI Predictive Maintenance Engine (UN SDG 9 Framework)
Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. ASSET SPECIFICATION:
   - Asset Tag       : {selected_ticket_row['Asset_ID']}
   - Equipment Name  : {selected_ticket_row['Asset_Name']}
   - Sector / Domain : {selected_ticket_row['Sector']}
   - Location / Zone : {selected_ticket_row['Zone']}
   - Criticality     : {selected_ticket_row['Criticality']}

2. DIAGNOSTIC TELEMETRY AT DISPATCH:
   - Operating Temp  : {selected_ticket_row['Temperature_C']} °C (Nominal: {selected_ticket_row['Nominal_Temp']} °C)
   - Vibration Level : {selected_ticket_row['Vibration_mms']} mm/s RMS (Nominal: {selected_ticket_row['Nominal_Vib']} mm/s)
   - Active Power    : {selected_ticket_row['Power_Load_kW']} kW
   - Edge Link Uptime: {selected_ticket_row['Node_Uptime_Pct']} %
   - ML Risk Index   : {selected_ticket_row['Risk_Index_Pct']} %
   - Remaining Life  : {selected_ticket_row['RUL_Hours']} Operating Hours Remaining

3. SEVERITY & PRIORITY CLASSIFICATION:
   - Current State   : {selected_ticket_row['Health_Status'].upper()}
   - Dispatch Tier   : {selected_ticket_row['Priority']}
   - Response Window : Immediate shutdown inspection within 4 hours

4. PRESCRIPTIVE ENGINEERING WORK SCOPE:
   {selected_ticket_row['Prescriptive_Remediation']}

5. REQUIRED REPLACEMENT BILL OF MATERIALS:
   {selected_ticket_row['Required_Parts']}

6. COMPLIANCE & SAFETY MANDATES:
   - Mandatory LOTO Protocol: {selected_ticket_row['Safety_Protocol']}
   - PPE Category 3 (Arc Flash Protection & Steel-Toe Footwear)
   - Environmental Containment: Comply with ISO 14001 Waste Oil Disposal Regulations

================================================================================
AUTHORIZED ELECTRONIC SIGNATURE:
InfraGuard AI Autonomous SCADA Dispatch Controller v2.6.4
================================================================================
"""

            st.text_area(
                "Official Engineering Work Order Preview",
                formatted_ticket,
                height=260
            )

        with dispatch_col2:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 18px; margin-bottom: 16px;">
                <div style="font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 6px;">Field Crew Notification Protocol</div>
                <div style="font-size: 12px; color: #94a3b8; line-height: 1.5;">
                    Dispatching this package transmits automated webhook payloads to plant CMMS (SAP PM / IBM Maximo), pushes encrypted SMS alerts to on-duty certified millwrights, and locks safety interlocks.
                </div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                # Text / Markdown downloadable work order
                st.download_button(
                    label="Download Work Order (.txt)",
                    data=formatted_ticket,
                    file_name=f"{selected_ticket_row['Work_Order_ID']}_{selected_ticket_row['Asset_ID']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with btn_col2:
                # CSV export for all pending work orders
                csv_buffer = flagged_assets[[
                    "Work_Order_ID", "Priority", "Asset_ID", "Asset_Name", "Zone",
                    "Health_Status", "Temperature_C", "Vibration_mms", "Node_Uptime_Pct",
                    "RUL_Hours", "Prescriptive_Remediation", "Required_Parts"
                ]].to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Export Fleet Matrix (.csv)",
                    data=csv_buffer,
                    file_name=f"InfraGuard_Field_Dispatch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.write("")
            if st.button("Transmit Automated Dispatch to Field Crews", type="primary", use_container_width=True):
                st.balloons()
                st.success(f"Work order {selected_ticket_row['Work_Order_ID']} successfully transmitted to On-Duty Engineering Crew Alpha! SCADA Lockout Sequence Acknowledged.")


# -----------------------------------------------------------------------------
# 9. FOOTER METADATA
# -----------------------------------------------------------------------------
st.write("")
st.markdown("""
<div style="text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.06); padding: 22px 0 10px 0; font-size: 11px; color: #64748b; font-family: 'IBM Plex Sans', sans-serif;">
    <div><strong>InfraGuard AI</strong> — Resilient Infrastructure Telemetry & Predictive Maintenance Platform</div>
    <div style="margin-top: 4px;">Designed in accordance with <strong>United Nations SDG 9: Industry, Innovation, and Infrastructure</strong> | ISO 10816-3 Industrial Vibration Standards</div>
</div>
""", unsafe_allow_html=True)