"""
MCP Server for Anomaly Detection using Statistical Methods
Uses FastMCP framework to detect anomalies in time series data.
"""

from fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import json

# Initialize FastMCP server
mcp = FastMCP("Anomaly Detection Server")


def detect_anomalies_moving_average(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    window: int = 7,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalies using moving average method.
    
    Args:
        df: DataFrame with time series data
        value_column: Column name containing values to analyze
        time_column: Column name containing time/datetime
        window: Window size for moving average
        threshold: Number of standard deviations from moving average to consider anomaly
    
    Returns:
        DataFrame with anomaly flags and scores
    """
    df = df.copy()
    df = df.sort_values(time_column)
    
    # Calculate moving average and standard deviation
    df['ma'] = df[value_column].rolling(window=window, min_periods=1, center=True).mean()
    df['ma_std'] = df[value_column].rolling(window=window, min_periods=1, center=True).std()
    
    # Handle NaN values in std (when window is smaller than data points)
    df['ma_std'] = df['ma_std'].fillna(df[value_column].std())
    
    # Calculate z-score from moving average
    df['z_score'] = (df[value_column] - df['ma']) / (df['ma_std'] + 1e-8)
    
    # Mark anomalies
    df['is_anomaly_ma'] = abs(df['z_score']) > threshold
    df['anomaly_score_ma'] = abs(df['z_score'])
    
    return df


def detect_anomalies_standard_deviation(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect anomalies using global standard deviation method.
    
    Args:
        df: DataFrame with time series data
        value_column: Column name containing values to analyze
        time_column: Column name containing time/datetime
        threshold: Number of standard deviations from mean to consider anomaly
    
    Returns:
        DataFrame with anomaly flags and scores
    """
    df = df.copy()
    df = df.sort_values(time_column)
    
    # Calculate global statistics
    mean_val = df[value_column].mean()
    std_val = df[value_column].std()
    
    # Calculate z-score
    df['z_score_std'] = (df[value_column] - mean_val) / (std_val + 1e-8)
    
    # Mark anomalies
    df['is_anomaly_std'] = abs(df['z_score_std']) > threshold
    df['anomaly_score_std'] = abs(df['z_score_std'])
    
    return df


def detect_anomalies_iqr(
    df: pd.DataFrame,
    value_column: str,
    time_column: str,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect anomalies using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame with time series data
        value_column: Column name containing values to analyze
        time_column: Column name containing time/datetime
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with anomaly flags and scores
    """
    df = df.copy()
    df = df.sort_values(time_column)
    
    # Calculate quartiles
    Q1 = df[value_column].quantile(0.25)
    Q3 = df[value_column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Mark anomalies
    df['is_anomaly_iqr'] = (df[value_column] < lower_bound) | (df[value_column] > upper_bound)
    df['anomaly_score_iqr'] = np.where(
        df[value_column] < lower_bound,
        (lower_bound - df[value_column]) / (IQR + 1e-8),
        np.where(
            df[value_column] > upper_bound,
            (df[value_column] - upper_bound) / (IQR + 1e-8),
            0
        )
    )
    
    return df


def detect_anomalies_core(
    data: str,
    time_column: str,
    aggregation_level: Optional[str] = None,
    value_column: Optional[str] = None,
    methods: List[str] = ["moving_average", "standard_deviation"],
    window: int = 7,
    threshold: float = 2.0,
    iqr_multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Core function to detect anomalies in time series data using statistical methods.
    This function can be called directly for testing.
    
    Args:
        data: string containing time series data [comma separated values]
        time_column: Name of the column containing time/datetime values
        aggregation_level: Optional aggregation level (e.g., "product_id", "category")
                          If provided, aggregates data at this level before detection
        value_column: Column name containing values to analyze for anomalies.
                      If None, will try to auto-detect numeric columns
        methods: List of methods to use: "moving_average", "standard_deviation", "iqr"
        window: Window size for moving average method (default: 7)
        threshold: Threshold for moving_average and standard_deviation methods (default: 2.0)
        iqr_multiplier: IQR multiplier for IQR method (default: 1.5)
    
    Returns:
        Dictionary containing detected anomalies with all methods applied
    """
    try:
        # Parse JSON data
        data_dict = [float(x) for x in data.split(',')]
        df = pd.DataFrame(data_dict)
        
        # Convert time column to datetime if needed
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        else:
            return {"error": f"Time column '{time_column}' not found in data."}
        
        # Handle aggregation if specified
        if aggregation_level and aggregation_level in df.columns:
            # Group by aggregation level and aggregate numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {col: 'sum' for col in numeric_cols if col != aggregation_level}
            agg_dict[time_column] = 'first'  # Keep first time value per group
            
            # Aggregate
            df_agg = df.groupby([aggregation_level, time_column]).agg(agg_dict).reset_index()
            df = df_agg
        
        # Auto-detect value column if not provided
        if value_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove time column if it's numeric (unlikely but possible)
            if time_column in numeric_cols:
                numeric_cols.remove(time_column)
            if numeric_cols:
                value_column = numeric_cols[0]  # Use first numeric column
            else:
                return {"error": "No numeric columns found for anomaly detection."}
        
        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found in data."}
        
        # Apply anomaly detection methods
        results = {}
        anomaly_df = df.copy()
        
        if "moving_average" in methods:
            anomaly_df = detect_anomalies_moving_average(
                anomaly_df, value_column, time_column, window, threshold
            )
            results["moving_average"] = {
                "anomalies": anomaly_df[anomaly_df['is_anomaly_ma']].to_dict('records'),
                "total_anomalies": int(anomaly_df['is_anomaly_ma'].sum()),
                "anomaly_rate": float(anomaly_df['is_anomaly_ma'].mean())
            }
        
        if "standard_deviation" in methods:
            anomaly_df = detect_anomalies_standard_deviation(
                anomaly_df, value_column, time_column, threshold
            )
            results["standard_deviation"] = {
                "anomalies": anomaly_df[anomaly_df['is_anomaly_std']].to_dict('records'),
                "total_anomalies": int(anomaly_df['is_anomaly_std'].sum()),
                "anomaly_rate": float(anomaly_df['is_anomaly_std'].mean())
            }
        
        if "iqr" in methods:
            anomaly_df = detect_anomalies_iqr(
                anomaly_df, value_column, time_column, iqr_multiplier
            )
            results["iqr"] = {
                "anomalies": anomaly_df[anomaly_df['is_anomaly_iqr']].to_dict('records'),
                "total_anomalies": int(anomaly_df['is_anomaly_iqr'].sum()),
                "anomaly_rate": float(anomaly_df['is_anomaly_iqr'].mean())
            }
        
        # Prepare summary
        summary = {
            "total_records": len(anomaly_df),
            "time_column": time_column,
            "value_column": value_column,
            "aggregation_level": aggregation_level,
            "methods_applied": methods,
            "results": results
        }
        
        # Add combined anomaly flags (any method detected anomaly)
        if len(methods) > 1:
            # Initialize combined flag with first available method
            anomaly_df['is_anomaly_combined'] = False
            
            # Combine all methods using OR logic
            if "moving_average" in methods:
                anomaly_df['is_anomaly_combined'] = (
                    anomaly_df['is_anomaly_combined'] | 
                    anomaly_df.get('is_anomaly_ma', False)
                )
            if "standard_deviation" in methods:
                anomaly_df['is_anomaly_combined'] = (
                    anomaly_df['is_anomaly_combined'] | 
                    anomaly_df.get('is_anomaly_std', False)
                )
            if "iqr" in methods:
                anomaly_df['is_anomaly_combined'] = (
                    anomaly_df['is_anomaly_combined'] | 
                    anomaly_df.get('is_anomaly_iqr', False)
                )
            
            summary["combined_anomalies"] = {
                "total_anomalies": int(anomaly_df['is_anomaly_combined'].sum()),
                "anomaly_rate": float(anomaly_df['is_anomaly_combined'].mean()),
                "anomalies": anomaly_df[anomaly_df['is_anomaly_combined']].to_dict('records')
            }
        
        return summary
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


@mcp.tool()
def detect_anomalies(
    data: str,
    time_column: str,
    aggregation_level: Optional[str] = None,
    value_column: Optional[str] = None,
    methods: List[str] = ["moving_average", "standard_deviation"]
) -> Dict[str, Any]:
    """
    MCP tool wrapper for anomaly detection.
    Detect anomalies in time series data using statistical methods.
    
    Args:
        data: string containing time series data [comma separated values]
        time_column: Name of the column containing time/datetime values
        aggregation_level: Optional aggregation level (e.g., "product_id", "category")
                          If provided, aggregates data at this level before detection
        value_column: Column name containing values to analyze for anomalies.
                      If None, will try to auto-detect numeric columns
        methods: List of methods to use: "moving_average", "standard_deviation", "iqr"
    
    Returns:
        Dictionary containing detected anomalies with all methods applied
    """
    # Call the core function
    return detect_anomalies_core(
        data=data,
        time_column=time_column,
        aggregation_level=aggregation_level,
        value_column=value_column,
        methods=methods
    )


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="http", host="0.0.0.0", port=8000)
