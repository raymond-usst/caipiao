import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys
import re

# Add project root to path
sys.path.append(str(Path.cwd()))

from lottery import database, analyzer, config
from lottery.utils.logger import logger

st.set_page_config(page_title="Lottery AI Dashboard", layout="wide", page_icon="🎱")

# --- Sidebar & Config ---
st.sidebar.title("🎱 AI Lottery Lab")
st.sidebar.markdown("Advanced Analysis & Control Panel")

# Refresh Button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

cfg = config.load_config("config.yaml")
db_path = cfg.common.db_path

@st.cache_data(ttl=3600)
def load_data():
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn)
    return df

df = load_data()

if df.empty:
    st.error("Dataset is empty. Please run `python cli.py sync` first.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Trends & Omission", "🌀 Chaos Analysis", "🧠 AI Lab"])

# --- Tab 1: Trends & Omission ---
with tab1:
    st.header("Historical Trends")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Sum Trend
        df["sum"] = df[["red1", "red2", "red3", "red4", "red5", "red6"]].sum(axis=1)
        fig_sum = px.line(df, x="draw_date", y="sum", title="Sum Value Trend", markers=True)
        st.plotly_chart(fig_sum, use_container_width=True)
    
    with col2:
        # Key Stats
        st.metric("Total Issues", len(df))
        st.metric("Latest Issue", df["issue"].iloc[-1])
        st.metric("Latest Sum", df["sum"].iloc[-1])

    # Omission Heatmap
    st.subheader("Omission Map (Red Balls)")
    omission_df = analyzer.calculate_omission(df)
    
    # Display last 50 issues
    recent_omission = omission_df.tail(50)
    fig_heat = px.imshow(
        recent_omission.T, 
        labels=dict(x="Issue", y="Ball Number", color="Omission"),
        x=recent_omission.index,
        y=recent_omission.columns,
        color_continuous_scale="Reds",
        aspect="auto"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# --- Tab 2: Chaos Analysis ---
with tab2:
    st.header("Chaos Theory & Phase Space")
    
    # Phase Space 3D
    st.subheader("Red Ball Phase Space (Sum)")
    tau = st.slider("Time Delay (tau)", 1, 10, 1)
    
    # Embed Sum
    series = df["sum"].to_numpy()
    if len(series) > 3 * tau:
        x = series[:-2*tau]
        y = series[tau:-tau]
        z = series[2*tau:]
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=np.arange(len(x)), colorscale='Viridis', width=2),
            opacity=0.8
        )])
        fig_3d.update_layout(scene=dict(xaxis_title='t', yaxis_title=f't+{tau}', zaxis_title=f't+{2*tau}'))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("Not enough data for Phase Space")
        
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Lyapunov Exponent")
        if st.button("Calculate Largest Lyapunov Exponent"):
            le = analyzer.lyapunov_exponent(series, dim=3, tau=tau)
            if le is not None:
                st.metric("LLE", f"{le:.4f}")
                if le > 0:
                    st.success("Positive LLE suggests Chaos")
                else:
                    st.info("Non-positive LLE suggests Order/Stability")
            else:
                st.error("Calculation failed (not enough data)")

# --- Tab 3: AI Lab ---
with tab3:
    st.header("Training Monitor")
    
    log_file = Path("lottery.log")
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Recent Logs
        recent_logs = lines[-50:]
        st.text_area("Recent Logs", "".join(recent_logs), height=200)
        
        # PBT Evolution Chart
        st.subheader("PBT Evolution")
        
        # Parse PBT logs: "[PBT] Generation N complete. Best score: X"
        pbt_pattern = re.compile(r"\[PBT\].*Generation\s+(\d+).*Best\s+(?:score|fitness)[:\s]+([0-9.]+)", re.IGNORECASE)
        generations = []
        scores = []
        
        for line in lines:
            match = pbt_pattern.search(line)
            if match:
                gen = int(match.group(1))
                score = float(match.group(2))
                generations.append(gen)
                scores.append(score)
        
        if generations:
            pbt_df = pd.DataFrame({"Generation": generations, "Best Fitness": scores})
            fig_pbt = px.line(pbt_df, x="Generation", y="Best Fitness", 
                              title="PBT Evolution Curve", markers=True)
            st.plotly_chart(fig_pbt, use_container_width=True)
        else:
            st.info("No PBT evolution data found in logs. Run PBT training to see evolution.")
        
    else:
        st.warning("No lottery.log found.")

# --- Footer ---
st.markdown("---")
st.markdown("Generated by Agentic AI assistant.")

