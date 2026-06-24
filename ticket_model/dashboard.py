"""
DIGITAL SHIELD 2 v3 — Interactive Risk Dashboard
Streamlit + Plotly visualisation for ML risk detection results.
"""
import sys
import os

# Add the directory containing this file to sys.path to enable src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Shield 2 | Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .main { background: #0a0e1a; }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 40%, #111827 100%);
    }
    .hero-header {
        background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.15) 50%, rgba(236,72,153,0.1) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
    }
    .kpi-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
        border: 1px solid rgba(71,85,105,0.3);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .kpi-card:hover {
        border-color: rgba(59,130,246,0.5);
        box-shadow: 0 0 30px rgba(59,130,246,0.1);
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        font-weight: 600;
    }
    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .text-blue { color: #60a5fa; }
    .text-red { color: #f87171; }
    .text-amber { color: #fbbf24; }
    .text-emerald { color: #34d399; }
    .text-purple { color: #a78bfa; }
    .text-pink { color: #f472b6; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(59,130,246,0.3);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(71,85,105,0.3);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Plotly layout helper ────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(15,23,42,0.8)",
    font=dict(color="#e2e8f0", family="Inter"),
    title_font=dict(size=16, color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(71,85,105,0.2)"),
    yaxis=dict(gridcolor="rgba(71,85,105,0.2)"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ─── Data Loading (V3: outputs/ directory) ───────────────────────────────────
@st.cache_data(ttl=600)
def load_all_scores():
    return pd.read_csv("outputs/all_risk_scores.csv")

@st.cache_data(ttl=600)
def load_high_risk():
    return pd.read_csv("outputs/high_risk_records.csv")

try:
    df_all = load_all_scores()
    df_high = load_high_risk()
except FileNotFoundError:
    st.error("CSV files not found in outputs/. Run `python risk_detection_engine.py` first!")
    st.stop()


# ─── Sidebar Filters ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    risk_min, risk_max = st.slider(
        "Risk Score Range", 0.0, 10.0, (8.0, 10.0), 0.1,
        help="Filter records by risk score range"
    )
    age_range = st.slider(
        "Age Range",
        int(df_all["age"].min()), int(df_all["age"].max()),
        (0, int(df_all["age"].max())), 1
    )
    sex_options = ["All"] + sorted(df_all["sex"].dropna().unique().tolist())
    sex_filter = st.selectbox("Gender", sex_options)
    search_query = st.text_input("🔎 Search (PNR, Name, User ID)", "")

    st.markdown("---")
    st.markdown("### 📊 Data Summary")
    st.markdown(f"**Total Records:** {len(df_all):,}")
    st.markdown(f"**High-Risk:** {len(df_high):,}")
    st.markdown(f"**Detection Rate:** {len(df_high)/len(df_all)*100:.1f}%")
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("**Engine:** Isolation Forest + LOF")
    st.markdown("**Boost:** Weighted Multiplicative")
    st.markdown("**Features:** 14 (8 categories)")


# ─── Apply Filters ───────────────────────────────────────────────────────────
filtered = df_all[
    (df_all["risk_score"] >= risk_min) &
    (df_all["risk_score"] <= risk_max) &
    (df_all["age"] >= age_range[0]) &
    (df_all["age"] <= age_range[1])
].copy()

if sex_filter != "All":
    filtered = filtered[filtered["sex"] == sex_filter]
if search_query:
    q = search_query.lower()
    filtered = filtered[
        filtered["pnrno"].astype(str).str.lower().str.contains(q, na=False) |
        filtered["psgn_name"].astype(str).str.lower().str.contains(q, na=False) |
        filtered["user_id"].astype(str).str.lower().str.contains(q, na=False)
    ]


# ─── Hero Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🛡️ Digital Shield 2 v3</div>
    <div class="hero-subtitle">ML-Powered Risk Intelligence — Isolation Forest + LOF Cross-Validation + Weighted Hybrid Boost</div>
</div>
""", unsafe_allow_html=True)


# ─── KPI Row ────────────────────────────────────────────────────────────────
total_records = len(df_all)
high_risk_count = len(df_high)
critical_count = len(df_all[df_all["risk_score"] >= 9.0])
minor_risk = len(df_all[(df_all["age"] < 18) & (df_all["risk_score"] > 8.0)])
avg_score = df_all["risk_score"].mean()
filtered_count = len(filtered)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Analyzed</div><div class="kpi-value text-blue">{total_records:,}</div><div class="kpi-delta text-blue">Ticket records</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk</div><div class="kpi-value text-red">{high_risk_count:,}</div><div class="kpi-delta text-red">{high_risk_count/total_records*100:.1f}% of total</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Critical (&gt;9.0)</div><div class="kpi-value text-pink">{critical_count:,}</div><div class="kpi-delta text-pink">Immediate attention</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Minors at Risk</div><div class="kpi-value text-amber">{minor_risk:,}</div><div class="kpi-delta text-amber">Age < 18 & high risk</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Risk Score</div><div class="kpi-value text-purple">{avg_score:.3f}</div><div class="kpi-delta text-purple">Population mean</div></div>', unsafe_allow_html=True)
with k6:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Filtered View</div><div class="kpi-value text-emerald">{filtered_count:,}</div><div class="kpi-delta text-emerald">Current selection</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Charts Row 1: Distribution + Pie ───────────────────────────────────────
st.markdown('<div class="section-header">📈 Risk Distribution Analysis</div>', unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c1:
    fig_hist = px.histogram(
        df_all, x="risk_score", nbins=80,
        color_discrete_sequence=["#3b82f6"],
        labels={"risk_score": "Risk Score", "count": "Records"},
        title="Risk Score Distribution (Weighted Multiplicative)"
    )
    fig_hist.add_vline(x=8.0, line_dash="dash", line_color="#f87171", annotation_text="Threshold (8.0)")
    fig_hist.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    def categorize(s):
        if s >= 9.0: return "Critical"
        elif s >= 7.0: return "High"
        elif s >= 5.0: return "Medium"
        else: return "Low"
    cats = df_all["risk_score"].apply(categorize).value_counts()
    fig_pie = px.pie(
        names=cats.index, values=cats.values, color=cats.index,
        color_discrete_map={"Critical": "#dc2626", "High": "#f59e0b", "Medium": "#3b82f6", "Low": "#10b981"},
        title="Risk Categories"
    )
    fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#e2e8f0", family="Inter"), title_font=dict(size=16, color="#e2e8f0"),
        margin=dict(l=20, r=20, t=50, b=20), legend=dict(font=dict(size=11)))
    fig_pie.update_traces(textfont_size=12, pull=[0.05, 0, 0, 0])
    st.plotly_chart(fig_pie, use_container_width=True)


# ─── Charts Row 2: Age vs Risk + Top Routes ─────────────────────────────────
st.markdown('<div class="section-header">🔬 Deep Dive Analytics</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    sample = df_all.sample(min(5000, len(df_all)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="age", y="risk_score", color="risk_score",
        color_continuous_scale=["#10b981", "#fbbf24", "#dc2626"],
        opacity=0.6, title="Age vs Risk Score",
        labels={"age": "Age", "risk_score": "Risk Score"}
    )
    fig_scatter.add_hline(y=8.0, line_dash="dash", line_color="#f87171", opacity=0.5)
    fig_scatter.add_vline(x=18, line_dash="dot", line_color="#fbbf24", opacity=0.5, annotation_text="Minor threshold")
    fig_scatter.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_scatter, use_container_width=True)

with c4:
    if "from_stn" in df_high.columns and "to_stn" in df_high.columns:
        route_counts = (
            df_high.assign(route=df_high["from_stn"] + " -> " + df_high["to_stn"])
            .groupby("route")["risk_score"].agg(["count", "mean"])
            .sort_values("count", ascending=False).head(10).reset_index()
        )
        route_counts.columns = ["Route", "Alerts", "Avg Risk"]
        fig_route = px.bar(
            route_counts, x="Alerts", y="Route", orientation="h",
            color="Avg Risk", color_continuous_scale=["#3b82f6", "#f59e0b", "#dc2626"],
            title="Top 10 High-Risk Routes"
        )
        layout_r = {**DARK_LAYOUT, "yaxis": dict(gridcolor="rgba(71,85,105,0.2)", autorange="reversed"),
                    "margin": dict(l=100, r=20, t=50, b=40)}
        fig_route.update_layout(**layout_r)
        st.plotly_chart(fig_route, use_container_width=True)


# ─── Charts Row 3: Gender + Reasons ─────────────────────────────────────────
c5, c6 = st.columns(2)
with c5:
    gender_risk = df_all.groupby("sex")["risk_score"].agg(["mean", "count"]).reset_index()
    gender_risk.columns = ["Gender", "Avg Risk", "Count"]
    fig_gender = px.bar(
        gender_risk, x="Gender", y="Avg Risk", color="Gender",
        color_discrete_map={"M": "#3b82f6", "F": "#ec4899"},
        title="Average Risk by Gender", text="Count"
    )
    fig_gender.update_layout(**DARK_LAYOUT, showlegend=False)
    fig_gender.update_traces(textposition="outside", textfont=dict(color="#94a3b8"))
    st.plotly_chart(fig_gender, use_container_width=True)

with c6:
    if "reason" in df_high.columns:
        all_reasons = df_high["reason"].str.split(";").explode().str.strip()
        # Truncate long reasons for chart readability
        all_reasons = all_reasons.str[:50]
        reason_counts = all_reasons.value_counts().head(8).reset_index()
        reason_counts.columns = ["Reason", "Count"]
        fig_reasons = px.bar(
            reason_counts, x="Count", y="Reason", orientation="h",
            color="Count", color_continuous_scale=["#6366f1", "#ec4899"],
            title="Most Common Risk Reasons"
        )
        layout_re = {**DARK_LAYOUT, "yaxis": dict(gridcolor="rgba(71,85,105,0.2)", autorange="reversed"),
                     "margin": dict(l=200, r=20, t=50, b=40), "showlegend": False}
        fig_reasons.update_layout(**layout_re)
        st.plotly_chart(fig_reasons, use_container_width=True)


# ─── Top Suspicious Users ───────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Top Suspicious Users</div>', unsafe_allow_html=True)

top_users = (
    df_high.groupby("user_id")
    .agg(alerts=("risk_score", "count"), avg_risk=("risk_score", "mean"),
         max_risk=("risk_score", "max"), passengers=("psgn_name", "nunique"))
    .sort_values("alerts", ascending=False).head(15).reset_index()
)
fig_users = go.Figure()
fig_users.add_trace(go.Bar(
    x=top_users["user_id"], y=top_users["alerts"], name="Alert Count",
    marker=dict(color=top_users["avg_risk"],
        colorscale=[[0, "#3b82f6"], [0.5, "#f59e0b"], [1, "#dc2626"]],
        colorbar=dict(title="Avg Risk", tickfont=dict(color="#94a3b8"))),
    text=top_users["passengers"].apply(lambda x: f"{x} pax"),
    textposition="outside", textfont=dict(color="#94a3b8", size=10),
))
fig_users.update_layout(
    title="Top 15 Users by Alert Volume (color = avg risk)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(15,23,42,0.8)",
    font=dict(color="#e2e8f0", family="Inter"),
    title_font=dict(size=16, color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(71,85,105,0.2)", tickangle=-45),
    yaxis=dict(gridcolor="rgba(71,85,105,0.2)", title="Alert Count"),
    margin=dict(l=40, r=20, t=60, b=100), height=450,
)
st.plotly_chart(fig_users, use_container_width=True)


# ─── Data Table ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Filtered Records</div>', unsafe_allow_html=True)

display_df = filtered.sort_values("risk_score", ascending=False).head(500).copy()
def risk_level(score):
    if score >= 9.0: return "🔴 CRITICAL"
    elif score >= 7.0: return "🟠 HIGH"
    elif score >= 5.0: return "🟡 MEDIUM"
    else: return "🟢 LOW"

display_df.insert(0, "Risk Level", display_df["risk_score"].apply(risk_level))
display_df["risk_score"] = display_df["risk_score"].round(4)

st.dataframe(
    display_df, use_container_width=True, height=500,
    column_config={
        "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=10),
        "pnrno": "PNR Number", "user_id": "User ID", "psgn_name": "Passenger",
        "age": "Age", "sex": "Gender", "from_stn": "From", "to_stn": "To", "reason": "Risk Reason",
    }
)

st.markdown(f"""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 20px; padding: 16px;">
    Digital Shield 2 v3 | IF + LOF + Weighted Boost | 14 Features | 
    Processed {total_records:,} records | {high_risk_count:,} alerts
</div>
""", unsafe_allow_html=True)
