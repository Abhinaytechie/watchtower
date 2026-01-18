# Testing Guide for Anomaly Detection MCP Server

## Quick Start

### 1. Install Dependencies

```bash
cd mcp-server-ad
pip install -r requirements.txt
```

### 2. Run Tests

There are multiple ways to test the server:

#### Option A: Run the Test Suite (Recommended)

```bash
python test_server.py
```

This runs a comprehensive test suite that covers:
- Basic functionality
- Aggregation level
- All detection methods
- Auto-detection features
- Error handling
- Real data from CSV

#### Option B: Run Example Usage

```bash
python example_usage.py
```

This shows example usage with sample data.

#### Option C: Test as Standalone Function

You can import and use the core function directly (recommended for testing):

```python
from server import detect_anomalies_core
import json

data = [
    {"date": "2024-01-01", "sales": 100},
    {"date": "2024-01-02", "sales": 105},
    {"date": "2024-01-03", "sales": 500},  # Anomaly
]

# Can pass list directly or JSON string
result = detect_anomalies_core(
    data=data,  # or json.dumps(data) for JSON string
    time_column="date",
    value_column="sales",
    methods=["moving_average", "standard_deviation"]
)

print(json.dumps(result, indent=2))
```

**Note:** `detect_anomalies_core` is the testable function. The `detect_anomalies` function is wrapped by `@mcp.tool()` and is only callable through the MCP server.

#### Option D: Test as MCP Server

To test as an actual MCP server:

1. **Start the server:**
   ```bash
   python server.py
   ```

2. **In another terminal or MCP client**, connect to the server and call the `detect_anomalies` tool with:
   - `data`: JSON string with time series data
   - `time_column`: Column name for time/datetime
   - `value_column`: Column to analyze (optional, auto-detected)
   - `aggregation_level`: Optional grouping column
   - `methods`: List of methods to use
   - `window`, `threshold`, `iqr_multiplier`: Method parameters

## Test with Real Data

You can test with the synthetic data generator:

1. **Generate test data:**
   ```bash
   cd "Data Generation"
   python synthetic-data-gen.py
   ```

2. **Test with the generated CSV:**
   The test suite will automatically use `data-gen/synthetic_data.csv` if available.

## Test Scenarios

### Scenario 1: Simple Time Series
```python
data = [
    {"date": "2024-01-01", "sales": 100},
    {"date": "2024-01-02", "sales": 105},
    {"date": "2024-01-03", "sales": 500},  # Anomaly
]
```

### Scenario 2: With Aggregation
```python
data = [
    {"date": "2024-01-01", "product_id": "P001", "sales": 100},
    {"date": "2024-01-01", "product_id": "P002", "sales": 200},
    {"date": "2024-01-02", "product_id": "P001", "sales": 500},  # Anomaly
]
```

### Scenario 3: Multiple Methods
```python
result = detect_anomalies(
    data=json.dumps(data),
    time_column="date",
    value_column="sales",
    methods=["moving_average", "standard_deviation", "iqr"]
)
```

## Expected Output

The server returns a JSON object with:
```json
{
  "total_records": 10,
  "time_column": "date",
  "value_column": "sales",
  "aggregation_level": null,
  "methods_applied": ["moving_average", "standard_deviation"],
  "results": {
    "moving_average": {
      "anomalies": [...],
      "total_anomalies": 2,
      "anomaly_rate": 0.2
    },
    "standard_deviation": {
      "anomalies": [...],
      "total_anomalies": 2,
      "anomaly_rate": 0.2
    }
  },
  "combined_anomalies": {
    "total_anomalies": 2,
    "anomaly_rate": 0.2,
    "anomalies": [...]
  }
}
```

## Troubleshooting

### Import Errors
If you get import errors, make sure dependencies are installed:
```bash
pip install fastmcp pandas numpy
```

### JSON Errors
Make sure your data is valid JSON. You can validate it:
```python
import json
json.loads(your_data_string)  # Should not raise error
```

### Time Column Errors
Make sure the time column exists and can be parsed as datetime. The server will try to convert it automatically.

### No Anomalies Detected
This might be normal if:
- Your threshold is too high
- Your data doesn't have significant outliers
- Try adjusting `threshold` or `window` parameters

