"""Data processor module — loading, cleaning, analysis."""

import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data" / "sample_health_data.csv"


def load_data(uploaded_file=None) -> Tuple[pd.DataFrame, str]:
    """Load data from uploaded file or fall back to sample dataset.
    Returns (DataFrame, source_name).
    """
    if uploaded_file is not None:
        name = uploaded_file.name
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {name}")
            return df, name
        except Exception as e:
            raise ValueError(f"Error reading {name}: {e}")
    else:
        df = pd.read_csv(SAMPLE_DATA_PATH)
        return df, "sample_health_data.csv (demo)"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: strip whitespace from string columns, standardize column names."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def get_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Return dataset overview metadata."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify columns as numeric, categorical, or date."""
    numeric_cols = []
    categorical_cols = []
    date_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Try to parse as date
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                date_cols.append(col)
            except (ValueError, TypeError):
                nunique = df[col].nunique()
                if nunique <= 30:
                    categorical_cols.append(col)
                else:
                    categorical_cols.append(col)  # treat high-cardinity as categorical too

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
    }


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for all columns."""
    desc = df.describe(include="all").T
    desc = desc.round(3)
    return desc


def identify_health_indicators(df: pd.DataFrame, col_types: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Heuristic to flag likely public health indicator columns."""
    health_keywords = [
        "rate", "prevalence", "incidence", "mortality", "morbidity",
        "life_expectancy", "obesity", "diabetes", "heart", "cancer",
        "insurance", "coverage", "hospital", "death", "birth", "infant",
        "immunization", "vaccination", "smoking", "mental_health",
        "depression", "hiv", "aids", "overdose", "suicide", "disability",
        "screening", "access", "beds", "admission", "er_",
        "cholesterol", "hypertension", "bp", "bmi",
    ]

    indicators = []
    for col in col_types.get("numeric", []):
        col_lower = col.lower()
        if any(kw in col_lower for kw in health_keywords):
            indicators.append({"column": col, "type": "health_outcome"})
    for col in col_types.get("categorical", []):
        col_lower = col.lower()
        if any(kw in col_lower for kw in health_keywords):
            indicators.append({"column": col, "type": "health_categorical"})

    # Social determinants
    sdoh_keywords = ["income", "education", "employment", "poverty", "uninsur", "medicaid", "medicare"]
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in sdoh_keywords):
            if not any(i["column"] == col for i in indicators):
                indicators.append({"column": col, "type": "social_determinant"})

    return indicators


def compute_correlations(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    corr = df[numeric_cols].corr()
    return corr.round(3)


def get_strong_correlations(corr: pd.DataFrame, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Extract correlations above threshold (excluding self)."""
    strong = []
    if corr.empty:
        return strong
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i < j:
                val = corr.iloc[i, j]
                if abs(val) >= threshold:
                    strong.append({
                        "var1": c1,
                        "var2": c2,
                        "correlation": round(val, 3),
                        "strength": "strong positive" if val > 0 else "strong negative",
                    })
    strong.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return strong


def prepare_data_context(df: pd.DataFrame, col_types: Dict, overview: Dict,
                         indicators: List, correlations: List) -> str:
    """Build a text summary of the data for LLM context."""
    lines = []
    lines.append(f"Dataset: {overview['rows']} rows, {overview['columns']} columns")
    lines.append(f"Columns: {', '.join(overview['column_names'])}")
    lines.append(f"Numeric columns: {', '.join(col_types['numeric'])}")
    lines.append(f"Categorical columns: {', '.join(col_types['categorical'])}")
    if col_types['date']:
        lines.append(f"Date columns: {', '.join(col_types['date'])}")
    lines.append(f"Missing values: {overview['total_missing']}")
    lines.append(f"Duplicate rows: {overview['duplicate_rows']}")

    if indicators:
        lines.append("\nIdentified health indicators:")
        for ind in indicators:
            lines.append(f"  - {ind['column']} ({ind['type']})")

    if correlations:
        lines.append("\nNotable correlations:")
        for c in correlations[:10]:
            lines.append(f"  - {c['var1']} ↔ {c['var2']}: r={c['correlation']} ({c['strength']})")

    # Add summary stats
    stats = df.describe().round(2)
    lines.append(f"\nSummary statistics:\n{stats.to_string()}")

    # Add a sample of the data
    lines.append(f"\nFirst 10 rows:\n{df.head(10).to_string()}")

    return "\n".join(lines)