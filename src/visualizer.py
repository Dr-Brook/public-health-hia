"""Visualizer module — Plotly chart generation."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


NAVY = "#1A1A2E"
TEAL = "#0D7377"
TEAL_LIGHT = "#14A3A8"
CORAL = "#E94560"
GOLD = "#F5A623"
GRAY = "#6C7A89"

COLOR_SEQUENCE = [TEAL, CORAL, GOLD, TEAL_LIGHT, "#5B2C6F", "#2E86C1", "#E74C3C", "#27AE60"]


def create_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    """Create histogram for a numeric column."""
    fig = px.histogram(
        df, x=col, nbins=30,
        color_discrete_sequence=[TEAL],
        title=f"Distribution of {col}",
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=16,
        xaxis_title=col,
        yaxis_title="Count",
        bargap=0.05,
        height=400,
    )
    fig.add_vline(x=df[col].mean(), line_dash="dash", line_color=CORAL,
                  annotation_text=f"Mean: {df[col].mean():.2f}")
    return fig


def create_bar_chart(df: pd.DataFrame, col: str) -> go.Figure:
    """Create bar chart for a categorical column showing value counts."""
    counts = df[col].value_counts().head(20)
    fig = px.bar(
        x=counts.index.astype(str), y=counts.values,
        color=counts.values,
        color_continuous_scale="Teal",
        title=f"Distribution of {col}",
        labels={"x": col, "y": "Count"},
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=16,
        height=400,
        xaxis_tickangle=-45,
        showlegend=False,
    )
    return fig


def create_time_series(df: pd.DataFrame, date_col: str, numeric_cols: List[str]) -> go.Figure:
    """Create time series chart if date column detected."""
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.sort_values(date_col)

    fig = go.Figure()
    for i, col in enumerate(numeric_cols[:6]):
        fig.add_trace(go.Scatter(
            x=df_copy[date_col], y=df_copy[col],
            mode="lines+markers",
            name=col,
            line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]),
        ))

    fig.update_layout(
        template="plotly_white",
        title=f"Time Series — {date_col}",
        title_font_size=16,
        xaxis_title=date_col,
        yaxis_title="Value",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap."""
    if corr.empty:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        template="plotly_white",
        title="Correlation Matrix",
        title_font_size=16,
        height=max(500, len(corr.columns) * 45),
        width=max(600, len(corr.columns) * 55),
    )
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                        color_col: Optional[str] = None) -> go.Figure:
    """Create scatter plot for two numeric variables, optionally colored by category."""
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"{y_col} vs {x_col}",
        trendline="ols" if color_col is None else None,
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=16,
        height=450,
    )
    return fig


def generate_auto_visualizations(df: pd.DataFrame, col_types: Dict[str, List[str]],
                                  corr: pd.DataFrame) -> List[Dict[str, Any]]:
    """Auto-generate a set of visualizations based on data types.
    Returns list of {"title": str, "figure": Figure, "type": str}.
    """
    viz_list = []

    # Histograms for numeric columns (limit to first 6)
    for col in col_types.get("numeric", [])[:6]:
        viz_list.append({
            "title": f"Distribution: {col}",
            "figure": create_histogram(df, col),
            "type": "histogram",
        })

    # Bar charts for categorical columns (limit to first 4)
    for col in col_types.get("categorical", [])[:4]:
        viz_list.append({
            "title": f"Distribution: {col}",
            "figure": create_bar_chart(df, col),
            "type": "bar",
        })

    # Time series if date column
    if col_types.get("date"):
        date_col = col_types["date"][0]
        viz_list.append({
            "title": f"Time Series by {date_col}",
            "figure": create_time_series(df, date_col, col_types.get("numeric", [])),
            "type": "time_series",
        })

    # Correlation heatmap
    if not corr.empty and len(corr.columns) >= 2:
        viz_list.append({
            "title": "Correlation Matrix",
            "figure": create_correlation_heatmap(corr),
            "type": "heatmap",
        })

    # Scatter plots for top correlations
    if len(col_types.get("numeric", [])) >= 2:
        # Find the strongest correlation pair
        if not corr.empty:
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            corr_vals = corr.where(mask).stack().abs().sort_values(ascending=False)
            if not corr_vals.empty:
                top_pair = corr_vals.index[0]
                x_c, y_c = top_pair
                color_col = col_types.get("categorical", [None])[0]
                viz_list.append({
                    "title": f"Scatter: {y_c} vs {x_c}",
                    "figure": create_scatter_plot(df, x_c, y_c, color_col),
                    "type": "scatter",
                })

    return viz_list