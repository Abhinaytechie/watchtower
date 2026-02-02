"""
MCP Server for Anomaly Detection using Statistical Methods
Reads data directly from a PostgreSQL table (e.g., 'tempt').

Compatible with:
- n8n Agents
- MCP tool calls
- Voice (Vapi)

IMPORTANT: FastMCP doesn't support **kwargs, so we explicitly define
optional parameters that n8n might pass as metadata.
"""

from fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, text
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

load_dotenv()
DB_PASS = quote_plus(os.getenv("DB_PASS"))
if not DB_PASS:
    raise RuntimeError("DB_PASS environment variable is not set")

# ==========================
# Initialize FastMCP Server
# ==========================
mcp = FastMCP("anomaly-detection")

# ==========================
# Database Configuration
# ==========================

DATABASE_URL = (
    f"postgresql://postgres.bkyhgraqxvxzboevblil:{DB_PASS}@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=1,
    max_overflow=0,
    pool_timeout=30,
)

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Failed to connect: {e}")

# ==========================
# Anomaly Detection Methods
# ==========================
def get_top_anomalies(
    df: pd.DataFrame,
    flag_col: str,
    score_col: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Return top N anomalies sorted by anomaly score.
    """
    if score_col not in df.columns:
        return []

    return (
        df[df[flag_col]]
        .sort_values(score_col, ascending=False)
        .head(limit)
        .to_dict("records")
    )


def detect_anomalies_moving_average(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    window: int = 7,
    threshold: float = 2.0
) -> pd.DataFrame:
    df = df.sort_values(time_column).copy()

    df["ma"] = df[value_column].rolling(window, min_periods=1).mean()
    df["ma_std"] = df[value_column].rolling(window, min_periods=1).std().fillna(0)

    df["z_score_ma"] = (df[value_column] - df["ma"]) / (df["ma_std"] + 1e-8)
    df["is_anomaly_ma"] = df["z_score_ma"].abs() > threshold
    df["anomaly_score_ma"] = df["z_score_ma"].abs()

    return df


def detect_anomalies_standard_deviation(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    threshold: float = 3.0
) -> pd.DataFrame:
    df = df.sort_values(time_column).copy()

    mean = df[value_column].mean()
    std = df[value_column].std() + 1e-8

    df["z_score_std"] = (df[value_column] - mean) / std
    df["is_anomaly_std"] = df["z_score_std"].abs() > threshold
    df["anomaly_score_std"] = df["z_score_std"].abs()

    return df


def detect_anomalies_iqr(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    multiplier: float = 1.5
) -> pd.DataFrame:
    df = df.sort_values(time_column).copy()

    q1 = df[value_column].quantile(0.25)
    q3 = df[value_column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    df["is_anomaly_iqr"] = (df[value_column] < lower) | (df[value_column] > upper)
    df["anomaly_score_iqr"] = np.where(
        df[value_column] < lower,
        (lower - df[value_column]) / (iqr + 1e-8),
        np.where(
            df[value_column] > upper,
            (df[value_column] - upper) / (iqr + 1e-8),
            0
        )
    )

    return df

# ==========================
# Core Processing Logic
# ==========================

def detect_anomalies_core(
    table: str,
    time_column: str,
    value_column: Optional[str],
    aggregation_level: Optional[str],
    methods: List[str]
) -> Dict[str, Any]:

    try:
        # 🔹 Load data from DB
        query = f'SELECT * FROM "{table}"'
        df = pd.read_sql(query, engine)

        if df.empty:
            return {"error": f"Table '{table}' is empty"}

        if time_column not in df.columns:
            return {"error": f"Missing time column: {time_column}"}

        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

        # 🔹 Auto-detect numeric column if not provided
        if value_column is None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                return {"error": "No numeric columns found"}
            value_column = numeric_cols[0]

        if value_column not in df.columns:
            return {"error": f"Missing value column: {value_column}"}

        results = {}

        if "moving_average" in methods:
            df = detect_anomalies_moving_average(df, value_column, time_column)
            results["moving_average"] = {
                "total_anomalies": int(df["is_anomaly_ma"].sum()),
                "anomaly_rate": float(df["is_anomaly_ma"].mean()),
                "top_anomalies": get_top_anomalies(
                    df,
                    flag_col="is_anomaly_ma",
                    score_col="anomaly_score_ma",
                    limit=5
                )
            }

        if "standard_deviation" in methods:
            df = detect_anomalies_standard_deviation(df, value_column, time_column)
            results["standard_deviation"] = {
                "total_anomalies": int(df["is_anomaly_std"].sum()),
                "anomaly_rate": float(df["is_anomaly_std"].mean()),
                "top_anomalies": get_top_anomalies(
                    df,
                    flag_col="is_anomaly_std",
                    score_col="anomaly_score_std",
                    limit=5
                )
            }

        if "iqr" in methods:
            df = detect_anomalies_iqr(df, value_column, time_column)
            results["iqr"] = {
                "total_anomalies": int(df["is_anomaly_iqr"].sum()),
                "anomaly_rate": float(df["is_anomaly_iqr"].mean()),
                "top_anomalies": get_top_anomalies(
                    df,
                    flag_col="is_anomaly_iqr",
                    score_col="anomaly_score_iqr",
                    limit=5
                )
            }

        return {
            "table": table,
            "time_column": time_column,
            "value_column": value_column,
            "methods_applied": methods,
            "total_records": len(df),
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}

# ==========================
# MCP Tool Definition
# ==========================

@mcp.tool()
def detect_anomalies(
    table: str,
    time_column: str,
    value_column: Optional[str] = None,
    aggregation_level: Optional[str] = None,
    methods: List[str] = ["moving_average", "standard_deviation"],
    # ✅ Explicitly define metadata fields as optional - they'll be ignored
    toolCallId: Optional[str] = None,
    id: Optional[str] = None,
    type: Optional[str] = None,
    tool: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Detect anomalies in a PostgreSQL table using statistical methods.
    The table is expected to be prepared beforehand (e.g., 'tempt').
    
    Args:
        table: Database table name (e.g., 'tempt')
        time_column: Column containing date/time values
        value_column: Numeric column to analyze (auto-detected if not provided)
        aggregation_level: Aggregation level (e.g., 'daily', 'weekly')
        methods: Detection methods to apply (default: ["moving_average", "standard_deviation"])
    
    Note: Additional metadata parameters (toolCallId, id, type, tool, metadata) are 
    accepted but ignored to maintain compatibility with various MCP clients.
    
    Returns:
        Dictionary containing anomaly detection results
    """
    # These metadata fields are intentionally ignored
    # Just call the core logic with the actual parameters
    return detect_anomalies_core(
        table=table,
        time_column=time_column,
        value_column=value_column,
        aggregation_level=aggregation_level,
        methods=methods
    )
