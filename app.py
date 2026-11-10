import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import datetime

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS (Crimson Dark Red Theme)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Oil Viscosity Dashboard",
    page_icon="🛢️",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stSidebar"] {
        background-color: #6B0D0D;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .kpi-card {
        background-color: #7A1212;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-title { font-size: 13px; opacity: 0.9; margin-bottom: 5px; }
    .kpi-value { font-size: 24px; font-weight: bold; }
    .kpi-sub { font-size: 11px; opacity: 0.8; }
    
    .status-normal { background-color: #1B5E20; border: 2px solid #4CAF50; color: white; padding: 15px; border-radius: 10px; text-align: center; }
    .status-warning { background-color: #E65100; border: 2px solid #FF9800; color: white; padding: 15px; border-radius: 10px; text-align: center; }
    .status-alarm { background-color: #FFD4D4; border: 2px solid #D32F2F; color: #B71C1C !important; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load Machine Learning Model (.pkl)
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        return None

model = load_ml_model()

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION & INPUTS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛢️ **OIL VISCOSITY**")
    st.caption("Predictive Maintenance & RUL Analytics")
    st.write("---")
    
    st.subheader("📥 Model Inputs")
    machine_id = st.text_input("Machine ID", "HYD-01")
    op_hours = st.number_input("Operating Hours (h)", min_value=0, value=2000, step=100)
    fresh_40 = st.number_input("Fresh Viscosity 40°C (Baseline)", min_value=1.0, value=46.0, step=1.0)
    fresh_100 = st.number_input("Fresh Viscosity 100°C", min_value=1.0, value=6.8, step=0.1)
    meas_100 = st.number_input("Measured Viscosity 100°C", min_value=1.0, value=6.5, step=0.1)
    meas_40_actual = st.number_input("Measured Viscosity 40°C (Actual)", min_value=1.0, value=52.0, step=0.5)
    
    st.write("---")
    st.subheader("⚙️ Operation Settings")
    daily_run_hours = st.number_input("Running Hours / Day", min_value=1, max_value=24, value=16)

# ---------------------------------------------------------
# 4. CALCULATIONS: ML, CONTROL LIMITS, RUL & TARGET DATE
# ---------------------------------------------------------
# Define Control Limits
warn_upper = fresh_40 * 1.10
warn_lower = fresh_40 * 0.90
alarm_upper = fresh_40 * 1.15
alarm_lower = fresh_40 * 0.85

# ML Prediction
if model is not None:
    input_data = np.array([[op_hours, fresh_40, fresh_100, meas_100]])
    predicted_visc_40 = float(model.predict(input_data)[0])
else:
    predicted_visc_40 = fresh_40 * (1 + (op_hours / 10000) * 0.8)

# Deviation (%)
dev_percent = ((meas_40_actual - fresh_40) / fresh_40) * 100

# Status & Limit Check
if meas_40_actual >= alarm_upper or meas_40_actual <= alarm_lower:
    status_label = "Alarm (Exceeded)"
    status_class = "status-alarm"
    status_msg = f"CRITICAL: Viscosity has breached Alarm limits! ({meas_40_actual:.2f} cSt)"
elif meas_40_actual >= warn_upper or meas_40_actual <= warn_lower:
    status_label = "Warning"
    status_class = "status-warning"
    status_msg = f"WARNING: Viscosity entering Warning limits ({meas_40_actual:.2f} cSt)"
else:
    status_label = "Normal"
    status_class = "status-normal"
    status_msg = "NORMAL: Viscosity within control limits"

# Remaining Useful Life (RUL) Calculation Logic
if op_hours > 0:
    degrade_rate_per_hour = abs(meas_40_actual - fresh_40) / op_hours
else:
    degrade_rate_per_hour = 0.003

if meas_40_actual >= alarm_upper or meas_40_actual <= alarm_lower:
    rul_hours = 0
    change_date_str = "IMMEDIATE (Today)"
else:
    remaining_visc_margin = alarm_upper - meas_40_actual
    rul_hours = max(0, int(remaining_visc_margin / degrade_rate_per_hour)) if degrade_rate_per_hour > 0 else 500
    days_left = rul_hours / daily_run_hours
    target_date = datetime.date.today() + datetime.timedelta(days=int(days_left))
    change_date_str = target_date.strftime("%d %b %Y")

# ---------------------------------------------------------
# 5. DASHBOARD HEADER
# ---------------------------------------------------------
st.markdown("""
<div style="background-color: #8B0000; padding: 15px; border-radius: 8px; color: white; margin-bottom: 15px;">
    <h2 style="margin:0; color:white;">📈 Dashboard: Oil Viscosity & RUL Tracking</h2>
    <p style="margin:0; opacity:0.8;">Hydraulic Oil Condition Monitoring & Replacement Schedule</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. KPI CARDS (รวมอายุที่เหลือและวันที่ต้องเปลี่ยน)
# ---------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💧 Latest Measured</div>
        <div class="kpi-value">{meas_40_actual:.2f} cSt</div>
        <div class="kpi-sub">at {op_hours:,} Hours</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🛢️ Baseline</div>
        <div class="kpi-value">{fresh_40:.2f} cSt</div>
        <div class="kpi-sub">Fresh Oil 40°C</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">⏳ Remaining Life (RUL)</div>
        <div class="kpi-value">{rul_hours:,} hrs</div>
        <div class="kpi-sub">based on {daily_run_hours}h/day</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📅 Change Due Date</div>
        <div class="kpi-value" style="font-size:20px;">{change_date_str}</div>
        <div class="kpi-sub">Target Replacement</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="{status_class}">
        <div class="kpi-title">⚠️ Current Status</div>
        <div class="kpi-value" style="font-size:20px;">{status_label}</div>
        <div class="kpi-sub">{dev_percent:+.1f}% Dev</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# 7. TIME-SERIES TREND WITH FULL CONTROL LIMITS (+ & -)
# ---------------------------------------------------------
c_col1, c_col2 = st.columns([1.3, 1])

with c_col1:
    st.subheader("📈 Viscosity Trend & Control Limits")
    
    hours_axis = np.linspace(0, max(2500, op_hours + rul_hours + 300), 10)
    
    if model is not None:
        sim_X = np.column_stack([
            hours_axis,
            np.full_like(hours_axis, fresh_40),
            np.full_like(hours_axis, fresh_100),
            np.full_like(hours_axis, meas_100)
        ])
        trend_y = model.predict(sim_X)
    else:
        trend_y = fresh_40 + (degrade_rate_per_hour * hours_axis)

    fig_trend = go.Figure()
    
    # Measured & Predicted Lines
    fig_trend.add_trace(go.Scatter(x=hours_axis, y=trend_y, mode='lines', name='Predicted Trend (ML)', line=dict(color='#FB8C00', dash='dash', width=2)))
    fig_trend.add_trace(go.Scatter(x=[op_hours], y=[meas_40_actual], mode='markers', name='Measured (Actual)', marker=dict(color='#1E88E5', size=14, symbol='diamond')))
    
    # Upper Limits
    fig_trend.add_hline(y=alarm_upper, line_dash="dash", line_color="#E53935", annotation_text=f"Alarm (+15%): {alarm_upper:.2f}")
    fig_trend.add_hline(y=warn_upper, line_dash="dash", line_color="#FDD835", annotation_text=f"Warning (+10%): {warn_upper:.2f}")
    
    # Baseline
    fig_trend.add_hline(y=fresh_40, line_color="#4CAF50", annotation_text=f"Baseline: {fresh_40:.2f}")
    
    # Lower Limits (เพิ่มเส้น Control ล่าง)
    fig_trend.add_hline(y=warn_lower, line_dash="dash", line_color="#FDD835", annotation_text=f"Warning (-10%): {warn_lower:.2f}")
    fig_trend.add_hline(y=alarm_lower, line_dash="dash", line_color="#E53935", annotation_text=f"Alarm (-15%): {alarm_lower:.2f}")

    fig_trend.update_layout(
        xaxis_title="Operating Hours (h)",
        yaxis_title="Kinematic Viscosity at 40°C (cSt)",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with c_col2:
    st.subheader("📊 Individual Control Chart (Shewhart)")
    
    fig_control = go.Figure()
    fig_control.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[fresh_40, fresh_40*1.03, fresh_40*1.06, fresh_40*1.09, meas_40_actual], mode='lines+markers', name='Viscosity', line=dict(color='#1E88E5', width=2)))
    
    fig_control.add_hline(y=alarm_upper, line_dash="dash", line_color="#E53935", annotation_text=f"UCL = {alarm_upper:.2f}")
    fig_control.add_hline(y=fresh_40, line_color="#4CAF50", annotation_text=f"CL = {fresh_40:.2f}")
    fig_control.add_hline(y=alarm_lower, line_dash="dash", line_color="#E53935", annotation_text=f"LCL = {alarm_lower:.2f}")

    fig_control.update_layout(
        xaxis_title="Sample No.",
        yaxis_title="Viscosity (cSt)",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_control, use_container_width=True)

# ---------------------------------------------------------
# 8. TABLES & ALERT LOG
# ---------------------------------------------------------
b_col1, b_col2 = st.columns(2)

with b_col1:
    st.subheader("🗄️ Maintenance Schedule & RUL Summary")
    df_rul = pd.DataFrame({
        "Parameter": ["Machine ID", "Current Operating Hours", "RUL (Remaining Hours)", "Daily Operating Hours", "Target Change Date", "Breach Status"],
        "Value": [machine_id, f"{op_hours:,} hrs", f"{rul_hours:,} hrs", f"{daily_run_hours} hrs/day", change_date_str, status_label]
    })
    st.dataframe(df_rul, use_container_width=True, hide_index=True)

with b_col2:
    st.subheader("🔔 Alert Log & Notifications")
    df_alert = pd.DataFrame({
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        "Machine ID": [machine_id],
        "Description": [status_msg],
        "Status": [status_label]
    })
    st.dataframe(df_alert, use_container_width=True, hide_index=True)
