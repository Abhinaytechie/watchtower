# Anomaly Detection MCP Server

An MCP (Model Context Protocol) server for detecting anomalies in time series data using statistical methods.

## Features

- **Multiple Detection Methods**:
  - Moving Average based detection
  - Standard Deviation based detection
  - Interquartile Range (IQR) based detection

- **Data Aggregation**: Supports aggregation at specified levels using pandas
- **JSON Input/Output**: Accepts JSON data and returns anomaly detection results

## Quick Start

### Installation

```bash
cd mcp-server-ad
pip install -r requirements.txt
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

### Running the Server

**Standalone (for testing):**
```bash
python server.py
```

**As MCP Server (for use with Claude Desktop, Cursor, etc.):**
See [SETUP.md](SETUP.md) for configuration instructions.

## Usage

### Using the Server

The server provides a single tool: `detect_anomalies`

#### Parameters

- `data` (str): JSON string containing time series data
- `time_column` (str): Name of the column containing time/datetime values
- `aggregation_level` (str, optional): Column name to aggregate by (e.g., "product_id")
- `value_column` (str, optional): Column name containing values to analyze. Auto-detected if not provided
- `methods` (List[str]): List of methods to use: `["moving_average", "standard_deviation", "iqr"]`
- `window` (int): Window size for moving average (default: 7)
- `threshold` (float): Threshold for moving_average and standard_deviation methods (default: 2.0)
- `iqr_multiplier` (float): IQR multiplier for IQR method (default: 1.5)

#### Example Input

```json
{
  "data": [
    {"date": "2024-01-01", "product_id": "P001", "sales": 100},
    {"date": "2024-01-02", "product_id": "P001", "sales": 105},
    {"date": "2024-01-03", "product_id": "P001", "sales": 500}
  ],
  "time_column": "date",
  "aggregation_level": "product_id",
  "value_column": "sales",
  "methods": ["moving_average", "standard_deviation"]
}
```

#### Example Output

```json
{
  "total_records": 3,
  "time_column": "date",
  "value_column": "sales",
  "aggregation_level": "product_id",
  "methods_applied": ["moving_average", "standard_deviation"],
  "results": {
    "moving_average": {
      "anomalies": [...],
      "total_anomalies": 1,
      "anomaly_rate": 0.33
    },
    "standard_deviation": {
      "anomalies": [...],
      "total_anomalies": 1,
      "anomaly_rate": 0.33
    }
  }
}
```

## Detection Methods

### 1. Moving Average Method
Uses a rolling window to calculate moving average and standard deviation. Points deviating more than the threshold (default: 2.0) standard deviations from the moving average are flagged as anomalies.

### 2. Standard Deviation Method
Uses global mean and standard deviation. Points deviating more than the threshold (default: 2.0) standard deviations from the global mean are flagged as anomalies.

### 3. IQR Method
Uses Interquartile Range (IQR) to identify outliers. Points outside Q1 - 1.5*IQR or Q3 + 1.5*IQR are flagged as anomalies.

