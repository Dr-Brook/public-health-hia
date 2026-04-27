"""
Public Health Data Insights Agent (HIA)
=========================================
A Streamlit application for analyzing public health datasets with AI-powered insights,
interactive visualizations, and follow-up Q&A.

Built for Brook Eshete, MD, MPH — Johns Hopkins Bloomberg School of Public Health.
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processor import (
    load_data, clean_data, get_overview, detect_column_types,
    get_summary_statistics, identify_health_indicators,
    compute_correlations, get_strong_correlations, prepare_data_context,
)
from src.ai_service import check_ollama_available, generate_insights, generate_chat_response
from src.visualizer import generate_auto_visualizations
from src.chat import DataChatEngine
from src.report_generator import generate_markdown_report, generate_pdf_report_html

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Public Health HIA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A1A2E 0%, #0D7377 100%);
        padding: 1.5rem 2rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.85; }
    .metric-card {
        background: #F0F4F8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0D7377;
    }
    .insight-box {
        background: linear-gradient(135deg, #f8fffe 0%, #f0f9f9 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #0D7377;
    }
    .stAlert { border-radius: 0.5rem; }
    .chat-message {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-user { background: #F0F4F8; border-left: 3px solid #0D7377; }
    .chat-assistant { background: #f0f9f9; border-left: 3px solid #1A1A2E; }
    div[data-testid="stSidebar"] { background: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.source_name = ""
    st.session_state.overview = {}
    st.session_state.col_types = {}
    st.session_state.indicators = []
    st.session_state.correlations_list = []
    st.session_state.corr_matrix = pd.DataFrame()
    st.session_state.insights = ""
    st.session_state.viz_list = []
    st.session_state.data_context = ""
    st.session_state.chat_engine = None
    st.session_state.chat_history = []
    st.session_state.insights_generated = False


def process_data(df: pd.DataFrame, source_name: str):
    """Process uploaded data and store results in session state."""
    df = clean_data(df)
    st.session_state.df = df
    st.session_state.source_name = source_name

    overview = get_overview(df)
    st.session_state.overview = overview

    col_types = detect_column_types(df)
    st.session_state.col_types = col_types

    indicators = identify_health_indicators(df, col_types)
    st.session_state.indicators = indicators

    numeric_cols = col_types.get("numeric", [])
    corr_matrix = compute_correlations(df, numeric_cols)
    st.session_state.corr_matrix = corr_matrix

    correlations_list = get_strong_correlations(corr_matrix, threshold=0.4)
    st.session_state.correlations_list = correlations_list

    data_context = prepare_data_context(df, col_types, overview, indicators, correlations_list)
    st.session_state.data_context = data_context

    # Generate visualizations
    st.session_state.viz_list = generate_auto_visualizations(df, col_types, corr_matrix)

    # Initialize chat engine
    chat_engine = DataChatEngine()
    chat_engine.initialize(df, data_context)
    st.session_state.chat_engine = chat_engine

    # Reset insights and chat
    st.session_state.insights = ""
    st.session_state.insights_generated = False
    st.session_state.chat_history = []


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Public Health HIA")
    st.markdown("---")

    # File upload
    st.markdown("#### Data Source")
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload a public health dataset for analysis",
    )

    use_sample = st.button("📂 Load Sample Dataset", use_container_width=True)

    if uploaded_file is not None:
        try:
            df, name = load_data(uploaded_file)
            process_data(df, name)
            st.success(f"✅ Loaded: {name}")
        except ValueError as e:
            st.error(str(e))
    elif use_sample:
        df, name = load_data()
        process_data(df, name)
        st.success(f"✅ Loaded: {name}")

    st.markdown("---")

    # AI Status
    st.markdown("#### AI Engine")
    ollama_ok = check_ollama_available()
    if ollama_ok:
        st.success("🟢 Ollama connected")
    else:
        st.warning("🔴 Ollama unavailable")
        st.caption("Start Ollama for AI features")

    st.markdown("---")

    # Data info (when loaded)
    if st.session_state.df is not None:
        st.markdown("#### Current Dataset")
        st.caption(f"**{st.session_state.source_name}**")
        o = st.session_state.overview
        st.metric("Rows", f"{o.get('rows', 0):,}")
        st.metric("Columns", o.get("columns", 0))
        st.metric("Missing", o.get("total_missing", 0))

    st.markdown("---")
    st.caption("Built by **Brook Eshete, MD, MPH**")
    st.caption("Johns Hopkins Bloomberg School of Public Health")


# ─── Main Content ───────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 Public Health Data Insights Agent</h1>
    <p>AI-powered analysis of public health datasets — trends, disparities, correlations & recommendations</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown("""
    ### Welcome! 👋

    Get started by uploading a public health dataset or loading the sample data.

    **Supported formats:** CSV, Excel (.xlsx, .xls)

    **The HIA will automatically:**
    - 📊 Analyze your dataset structure and statistics
    - 🧠 Generate AI-powered public health insights
    - 📈 Create interactive visualizations
    - 💬 Enable follow-up Q&A about your data
    - 📄 Export a professional analysis report
    """)

    # Show what the sample data looks like
    st.markdown("### Sample Dataset Preview")
    try:
        sample_df, _ = load_data()
        st.dataframe(sample_df.head(10), use_container_width=True, height=350)
        st.caption("This demo dataset contains regional US public health indicators (2018-2022)")
    except Exception:
        pass

else:
    # ─── Tabs ───────────────────────────────────────────────────────────────
    tab_overview, tab_insights, tab_viz, tab_chat, tab_export = st.tabs(
        ["📊 Overview", "🧠 Insights", "📈 Visualizations", "💬 Chat", "📄 Export"]
    )

    # ── Overview Tab ────────────────────────────────────────────────────────
    with tab_overview:
        df = st.session_state.df
        overview = st.session_state.overview
        col_types = st.session_state.col_types

        st.markdown("## Dataset Overview")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Rows", f"{overview['rows']:,}")
        col2.metric("📋 Columns", overview["columns"])
        col3.metric("❓ Missing Values", overview["total_missing"])
        col4.metric("🔄 Duplicates", overview["duplicate_rows"])

        # Column type breakdown
        st.markdown("### Column Types")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Numeric** 🔢")
            for col in col_types.get("numeric", []):
                st.markdown(f"- `{col}`")
        with c2:
            st.markdown("**Categorical** 🏷️")
            for col in col_types.get("categorical", []):
                st.markdown(f"- `{col}`")
        with c3:
            st.markdown("**Date** 📅")
            for col in col_types.get("date", []):
                st.markdown(f"- `{col}`")
            if not col_types.get("date"):
                st.caption("None detected")

        # Health indicators
        indicators = st.session_state.indicators
        if indicators:
            st.markdown("### 🏥 Identified Health Indicators")
            ind_df = pd.DataFrame(indicators)
            st.dataframe(ind_df, use_container_width=True, hide_index=True)

        # Missing values detail
        if overview["total_missing"] > 0:
            with st.expander("🔍 Missing Values Detail"):
                missing = {k: v for k, v in overview["missing_values"].items() if v > 0}
                st.dataframe(pd.DataFrame(list(missing.items()), columns=["Column", "Missing Count"]),
                             use_container_width=True, hide_index=True)

        # Summary statistics
        st.markdown("### Summary Statistics")
        stats = get_summary_statistics(df)
        st.dataframe(stats, use_container_width=True)

        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(20), use_container_width=True, height=400)

    # ── Insights Tab ────────────────────────────────────────────────────────
    with tab_insights:
        st.markdown("## AI-Powered Public Health Insights")

        if not ollama_ok:
            st.warning("⚠️ Ollama is not available. AI insights require a running Ollama instance with the model loaded.")
            st.info("Start Ollama and run: `ollama pull glm-5.1:cloud`")

        # Correlations summary
        correlations = st.session_state.correlations_list
        if correlations:
            st.markdown("### Notable Correlations")
            corr_df = pd.DataFrame(correlations)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

        # Generate insights button
        if not st.session_state.insights_generated:
            if st.button("🧠 Generate AI Insights", type="primary", use_container_width=True,
                         disabled=not ollama_ok):
                with st.spinner("Analyzing data and generating insights... This may take 30-60 seconds."):
                    insights = generate_insights(st.session_state.data_context)
                    st.session_state.insights = insights
                    st.session_state.insights_generated = True
                    st.rerun()
        else:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(st.session_state.insights)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔄 Regenerate Insights", use_container_width=True):
                st.session_state.insights_generated = False
                st.rerun()

    # ── Visualizations Tab ──────────────────────────────────────────────────
    with tab_viz:
        st.markdown("## Interactive Visualizations")

        viz_list = st.session_state.viz_list
        if not viz_list:
            st.info("No visualizations could be generated from this dataset.")
        else:
            # Display visualizations in a grid
            for i in range(0, len(viz_list), 2):
                cols = st.columns(min(2, len(viz_list) - i))
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(viz_list):
                        viz = viz_list[idx]
                        with col:
                            st.plotly_chart(viz["figure"], use_container_width=True, key=f"viz_{idx}")

        # Custom visualization builder
        st.markdown("---")
        st.markdown("### Custom Visualization")
        numeric_cols = st.session_state.col_types.get("numeric", [])
        categorical_cols = st.session_state.col_types.get("categorical", [])

        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="custom_x")
            with col2:
                y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="custom_y")
            with col3:
                color_col = st.selectbox("Color by", [None] + categorical_cols, key="custom_color",
                                         format_func=lambda x: "None" if x is None else x)

            from src.visualizer import create_scatter_plot
            fig = create_scatter_plot(st.session_state.df, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True, key="custom_scatter")

    # ── Chat Tab ────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("## Ask Questions About Your Data")

        if not ollama_ok:
            st.warning("⚠️ Ollama is not available. Chat requires a running Ollama instance.")

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message chat-user">**You:** {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message chat-assistant">**HIA:** {msg["content"]}</div>',
                            unsafe_allow_html=True)

        # Chat input
        chat_input = st.chat_input("Ask a question about your data...",
                                   disabled=not ollama_ok)

        if chat_input:
            st.session_state.chat_history.append({"role": "user", "content": chat_input})

            with st.spinner("Analyzing your question..."):
                engine = st.session_state.chat_engine
                if engine and engine._initialized:
                    response = engine.query(chat_input, st.session_state.chat_history)
                else:
                    response = generate_chat_response(
                        st.session_state.data_context, chat_input, st.session_state.chat_history
                    )

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

        # Suggested questions
        if not st.session_state.chat_history:
            st.markdown("### 💡 Suggested Questions")
            suggestions = [
                "What are the strongest predictors of health outcomes in this data?",
                "Which regions show the greatest health disparities?",
                "What correlations exist between socioeconomic factors and health indicators?",
                "What trends are visible over time?",
                "What public health interventions would you recommend based on this data?",
            ]
            for q in suggestions:
                if st.button(q, key=f"sug_{q[:20]}"):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("Analyzing..."):
                        engine = st.session_state.chat_engine
                        if engine and engine._initialized:
                            response = engine.query(q, st.session_state.chat_history)
                        else:
                            response = generate_chat_response(
                                st.session_state.data_context, q, st.session_state.chat_history
                            )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

    # ── Export Tab ───────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("## Export Analysis Report")

        if not st.session_state.insights_generated:
            st.warning("⚠️ Generate AI insights first (go to the Insights tab) before exporting a report.")

        # Markdown report
        st.markdown("### Markdown Report")
        md_report = generate_markdown_report(
            st.session_state.overview,
            st.session_state.col_types,
            st.session_state.indicators,
            st.session_state.correlations_list,
            st.session_state.insights if st.session_state.insights_generated else "*Insights not yet generated.*",
            st.session_state.source_name,
        )
        st.download_button(
            "📥 Download Markdown Report",
            data=md_report,
            file_name="public_health_analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        # HTML/PDF report
        st.markdown("### PDF Report (HTML)")
        html_report = generate_pdf_report_html(
            st.session_state.overview,
            st.session_state.col_types,
            st.session_state.indicators,
            st.session_state.correlations_list,
            st.session_state.insights if st.session_state.insights_generated else "*Insights not yet generated.*",
            st.session_state.source_name,
        )
        st.download_button(
            "📥 Download HTML Report (print to PDF)",
            data=html_report,
            file_name="public_health_analysis_report.html",
            mime="text/html",
            use_container_width=True,
        )

        st.caption("💡 Tip: Open the HTML file in a browser and use Print → Save as PDF for a formatted PDF report.")

        # Preview
        with st.expander("📄 Report Preview"):
            st.markdown(md_report)