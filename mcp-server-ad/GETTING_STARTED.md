# Getting Started with Anomaly Detection MCP Server

Welcome! This guide will help you set up and use the Anomaly Detection MCP Server.

## 📋 Table of Contents

1. [What is this?](#what-is-this)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Testing](#testing)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)

## What is this?

The Anomaly Detection MCP Server is an MCP (Model Context Protocol) server that detects anomalies in time series data using statistical methods:

- **Moving Average** - Detects deviations from rolling averages
- **Standard Deviation** - Detects outliers using global statistics
- **IQR Method** - Detects outliers using quartiles

## Installation

### Option 1: Automated Setup (Recommended)

**Windows:**
```powershell
cd mcp-server-ad
.\setup.ps1
```

**macOS/Linux:**
```bash
cd mcp-server-ad
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
cd mcp-server-ad
pip install -r requirements.txt
```

Or with virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

### For Claude Desktop

1. **Find your Claude Desktop config file:**
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Add the server configuration:**

   Get your absolute path to `server.py`:
   ```bash
   # Windows PowerShell
   (Resolve-Path "mcp-server-ad\server.py").Path
   
   # macOS/Linux
   realpath mcp-server-ad/server.py
   ```

3. **Add to config file:**

   ```json
   {
     "mcpServers": {
       "anomaly-detection": {
         "command": "python",
         "args": [
           "C:/Dev/Watchtower/mcp-server-ad/server.py"
         ]
       }
     }
   }
   ```

   **Replace the path with your actual path!**

4. **Restart Claude Desktop**

### For Cursor IDE

1. Open Settings (File → Preferences → Settings)
2. Search for "MCP"
3. Add server configuration (same format as above)
4. Restart Cursor

See [SETUP.md](SETUP.md) for detailed configuration instructions.

## Testing

### Test the Core Functionality

```bash
python test_server.py
```

This will run 6 comprehensive tests:
- ✓ Basic functionality
- ✓ Aggregation level
- ✓ All detection methods
- ✓ Auto-detection
- ✓ Real CSV data
- ✓ Error handling

### Test with Examples

```bash
python example_usage.py
```

### Test Directly in Python

```python
from server import detect_anomalies_core

data = [
    {"date": "2024-01-01", "sales": 100},
    {"date": "2024-01-02", "sales": 105},
    {"date": "2024-01-03", "sales": 500},  # Anomaly
]

result = detect_anomalies_core(
    data=data,
    time_column="date",
    value_column="sales",
    methods=["moving_average", "standard_deviation"]
)

print(result)
```

## Usage

### Using via MCP Client (Claude Desktop, Cursor)

Once configured, you can call the `detect_anomalies` tool:

**Parameters:**
- `data`: JSON string with time series data
- `time_column`: Column name for time/datetime
- `value_column`: Column to analyze (optional, auto-detected)
- `aggregation_level`: Optional grouping column
- `methods`: List of methods: `["moving_average", "standard_deviation", "iqr"]`
- `window`: Window size for moving average (default: 7)
- `threshold`: Threshold for MA/SD methods (default: 2.0)
- `iqr_multiplier`: IQR multiplier (default: 1.5)

**Example:**
```json
{
  "data": "[{\"date\":\"2024-01-01\",\"sales\":100},{\"date\":\"2024-01-02\",\"sales\":500}]",
  "time_column": "date",
  "value_column": "sales",
  "methods": ["moving_average", "standard_deviation"],
  "window": 7,
  "threshold": 2.0
}
```

### Using Directly in Python

```python
from server import detect_anomalies_core
import json

# Your data
data = [
    {"date": "2024-01-01", "product_id": "P001", "sales": 100},
    {"date": "2024-01-02", "product_id": "P001", "sales": 105},
    {"date": "2024-01-03", "product_id": "P001", "sales": 500},  # Anomaly
]

# Detect anomalies
result = detect_anomalies_core(
    data=data,  # Can also pass JSON string
    time_column="date",
    aggregation_level="product_id",
    value_column="sales",
    methods=["moving_average", "standard_deviation", "iqr"],
    window=7,
    threshold=2.0
)

# Print results
print(json.dumps(result, indent=2))
```

## Troubleshooting

### Server Not Appearing

1. **Check Python path:**
   ```bash
   python --version
   ```

2. **Verify file path:**
   - Use absolute paths, not relative
   - Check path is correct in config file

3. **Check logs:**
   - Claude Desktop: Check console/logs
   - Cursor: Check MCP logs in settings

### Import Errors

1. **Verify dependencies:**
   ```bash
   pip list | grep -E "(fastmcp|pandas|numpy)"
   ```

2. **Reinstall:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Use virtual environment:**
   See [SETUP.md](SETUP.md) for venv setup

### Path Issues on Windows

- Use forward slashes `/` or double backslashes `\\`
- Use absolute paths
- Enclose paths with spaces in quotes

### Test Server Manually

```bash
python server.py
```

If it starts without errors, the server is working.

## Next Steps

- 📖 Read [README.md](README.md) for detailed usage
- 🔧 See [SETUP.md](SETUP.md) for advanced configuration
- 🧪 See [TESTING.md](TESTING.md) for testing guide
- ⚡ See [QUICK_START.md](QUICK_START.md) for quick reference

## Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Test | `python test_server.py` |
| Run Server | `python server.py` |
| Example | `python example_usage.py` |
| Setup Script | `.\setup.ps1` (Windows) or `./setup.sh` (Mac/Linux) |

## Support

If you encounter issues:

1. Check [SETUP.md](SETUP.md) for detailed setup
2. Check [TESTING.md](TESTING.md) for testing help
3. Verify all dependencies are installed
4. Check server logs for errors

Happy anomaly detecting! 🎯

