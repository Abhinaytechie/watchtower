# Quick Start Guide

Get the Anomaly Detection MCP Server up and running in minutes!

## Step 1: Install Dependencies

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

**Or manually:**
```bash
cd mcp-server-ad
pip install -r requirements.txt
```

## Step 2: Test the Server

```bash
python test_server.py
```

This will run all tests and verify everything works.

## Step 3: Configure MCP Client

### For Claude Desktop

1. Open or create: `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
   or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
   or `~/.config/Claude/claude_desktop_config.json` (Linux)

2. Add this configuration:

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

**Important:** Replace `C:/Dev/Watchtower/mcp-server-ad/server.py` with your actual path!

**Using virtual environment (recommended):**

```json
{
  "mcpServers": {
    "anomaly-detection": {
      "command": "C:/Dev/Watchtower/mcp-server-ad/venv/Scripts/python.exe",
      "args": [
        "C:/Dev/Watchtower/mcp-server-ad/server.py"
      ]
    }
  }
}
```

3. **Restart Claude Desktop**

4. **Test in Claude Desktop**: The `detect_anomalies` tool should now be available!

### For Cursor IDE

1. Open Cursor Settings (File → Preferences → Settings)
2. Search for "MCP" settings
3. Add the server configuration (same format as above)
4. Restart Cursor

## Step 4: Use the Server

### In Claude Desktop or Cursor

Simply call the `detect_anomalies` tool with your data:

```json
{
  "data": "[{\"date\":\"2024-01-01\",\"sales\":100},{\"date\":\"2024-01-02\",\"sales\":500}]",
  "time_column": "date",
  "value_column": "sales",
  "methods": ["moving_average", "standard_deviation"]
}
```

### Directly in Python

```python
from server import detect_anomalies_core

result = detect_anomalies_core(
    data=[{"date": "2024-01-01", "sales": 100}, {"date": "2024-01-02", "sales": 500}],
    time_column="date",
    value_column="sales",
    methods=["moving_average", "standard_deviation"]
)

print(result)
```

## Troubleshooting

### Server not appearing?
- Check Python is in PATH: `python --version`
- Verify the path to `server.py` is correct
- Check Claude Desktop/Cursor logs for errors
- Try running `python server.py` manually to see if it starts

### Import errors?
- Make sure dependencies are installed: `pip install -r requirements.txt`
- Use a virtual environment (recommended)

### Need help?
- See [SETUP.md](SETUP.md) for detailed setup instructions
- See [TESTING.md](TESTING.md) for testing guide
- See [README.md](README.md) for usage examples

## Next Steps

- Run `python example_usage.py` to see examples
- Try with your own data
- Customize detection methods and thresholds

