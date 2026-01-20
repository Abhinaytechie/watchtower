"""
MCP Server for Anomaly Detection using Statistical Methods
Reads data directly from a PostgreSQL table (e.g., 'tempt').

Compatible with:
- n8n Agents
- MCP tool calls
- Voice (Vapi)
"""

from fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
import os
load_dotenv()
DB_PASS=os.getenv("DB_PASS")
if not DB_PASS:
    raise RuntimeError("DB_PASS environment variable is not set")
# ==========================
# Initialize FastMCP Server
# ==========================
mcp = FastMCP("anomaly-detection")

# ==========================
# Database Configuration
# ==========================
# ⚠️ Replace with your actual credentials or env vars
DATABASE_URL = (
    f"postgresql+psycopg://postgres:{DB_PASS}"
    "@db.bkyhgraqxvxzboevblil.supabase.co:5432/postgres"
)
engine = create_engine(DATABASE_URL)

# ==========================
# Pydantic Input Model
# ==========================
class DetectAnomaliesInput(BaseModel):
    table: str
    time_column: str
    value_column: Optional[str] = None
    aggregation_level: Optional[str] = None
    methods: List[str] = ["moving_average", "standard_deviation"]

    # ✅ Ignore n8n / MCP extra metadata safely
    model_config = ConfigDict(extra="ignore")

# ==========================
# Anomaly Detection Methods
# ==========================

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
                "anomalies": df[df["is_anomaly_ma"]].to_dict("records")
            }

        if "standard_deviation" in methods:
            df = detect_anomalies_standard_deviation(df, value_column, time_column)
            results["standard_deviation"] = {
                "total_anomalies": int(df["is_anomaly_std"].sum()),
                "anomaly_rate": float(df["is_anomaly_std"].mean()),
                "anomalies": df[df["is_anomaly_std"]].to_dict("records")
            }

        if "iqr" in methods:
            df = detect_anomalies_iqr(df, value_column, time_column)
            results["iqr"] = {
                "total_anomalies": int(df["is_anomaly_iqr"].sum()),
                "anomaly_rate": float(df["is_anomaly_iqr"].mean()),
                "anomalies": df[df["is_anomaly_iqr"]].to_dict("records")
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
    methods: List[str] = ["moving_average", "standard_deviation"]
) -> Dict[str, Any]:
    """
    Detect anomalies in a PostgreSQL table using statistical methods.
    The table is expected to be prepared beforehand (e.g., 'tempt').
    """
    return detect_anomalies_core(
        table=table,
        time_column=time_column,
        value_column=value_column,
        aggregation_level=aggregation_level,
        methods=methods
    )
