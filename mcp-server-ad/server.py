"""
MCP Server for Anomaly Detection using Statistical Methods
Uses FastMCP framework to detect anomalies in time series data.
"""

from fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
import json

# ==========================
# Initialize FastMCP Server
# ==========================
# IMPORTANT: Must be top-level for FastMCP Cloud
mcp = FastMCP("anomaly-detection")



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
# Core Processing Function
# ==========================

def detect_anomalies_core(
    data: str,
    time_column: str,
    aggregation_level: Optional[str],
    value_column: Optional[str],
    methods: List[str],
    window: int = 7,
    threshold: float = 2.0,
    iqr_multiplier: float = 1.5
) -> Dict[str, Any]:
    try:
        # ✅ FIXED: Expect JSON list of records
        records = json.loads(data)
        df = pd.DataFrame(records)

        if time_column not in df.columns:
            return {"error": f"Missing time column: {time_column}"}

        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

        # Optional aggregation
        if aggregation_level and aggregation_level in df.columns:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            agg_dict = {col: "sum" for col in numeric_cols if col != aggregation_level}
            agg_dict[time_column] = "first"

            df = (
                df.groupby([aggregation_level, time_column])
                .agg(agg_dict)
                .reset_index()
            )

        # Auto-detect value column
        if value_column is None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                return {"error": "No numeric columns found"}
            value_column = numeric_cols[0]

        if value_column not in df.columns:
            return {"error": f"Missing value column: {value_column}"}

        results = {}

        if "moving_average" in methods:
            df = detect_anomalies_moving_average(df, value_column, time_column, window, threshold)
            results["moving_average"] = {
                "total_anomalies": int(df["is_anomaly_ma"].sum()),
                "anomaly_rate": float(df["is_anomaly_ma"].mean()),
                "anomalies": df[df["is_anomaly_ma"]].to_dict("records")
            }

        if "standard_deviation" in methods:
            df = detect_anomalies_standard_deviation(df, value_column, time_column, threshold)
            results["standard_deviation"] = {
                "total_anomalies": int(df["is_anomaly_std"].sum()),
                "anomaly_rate": float(df["is_anomaly_std"].mean()),
                "anomalies": df[df["is_anomaly_std"]].to_dict("records")
            }

        if "iqr" in methods:
            df = detect_anomalies_iqr(df, value_column, time_column, iqr_multiplier)
            results["iqr"] = {
                "total_anomalies": int(df["is_anomaly_iqr"].sum()),
                "anomaly_rate": float(df["is_anomaly_iqr"].mean()),
                "anomalies": df[df["is_anomaly_iqr"]].to_dict("records")
            }

        return {
            "total_records": len(df),
            "time_column": time_column,
            "value_column": value_column,
            "aggregation_level": aggregation_level,
            "methods_applied": methods,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================
# MCP Tool
# ==========================

@mcp.tool()
def detect_anomalies(
    data: str,
    time_column: str,
    aggregation_level: Optional[str] = None,
    value_column: Optional[str] = None,
    methods: List[str] = ["moving_average", "standard_deviation"]
) -> Dict[str, Any]:
    """
    Detect anomalies in time series data using statistical methods.
    """
    return detect_anomalies_core(
        data=data,
        time_column=time_column,
        aggregation_level=aggregation_level,
        value_column=value_column,
        methods=methods
    )
