import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from ui.tabs import live, historical, backtest

st.set_page_config(
    page_title="Algo Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stTabs [data-baseweb="tab-list"] { gap: 12px; }
  .stTabs [data-baseweb="tab"] { padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

st.title("Algo Strategy Comparison")
st.caption("RL Trader · Regime Trader · Claudebot — separate Alpaca paper accounts")

tab_live, tab_hist, tab_bt = st.tabs(["📡 Live Portfolio", "📊 Historical Performance", "🔬 Backtest Comparison"])

with tab_live:
    live.render()

with tab_hist:
    historical.render()

with tab_bt:
    backtest.render()
