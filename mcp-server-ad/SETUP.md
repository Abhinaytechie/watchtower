# MCP Server Setup Guide

This guide explains how to set up and configure the Anomaly Detection MCP Server.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Install Dependencies

```bash
cd mcp-server-ad
pip install -r requirements.txt
```

This will install:
- `fastmcp` - FastMCP framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations

### 2. Verify Installation

Test that everything works:

```bash
python test_server.py
```

Or test the core functionality directly:

```bash
python example_usage.py
```

## Running the Server

### Option 1: Standalone Testing

For testing purposes, you can run the server directly:

```bash
python server.py
```

This will start the server in stdio mode (reads from stdin, writes to stdout).

### Option 2: Configure as MCP Server (Recommended)

To use the server with MCP clients (like Claude Desktop, Cursor, etc.), you need to configure it in your MCP client's configuration file.

## Configuration

### For Claude Desktop

1. **Find your Claude Desktop config directory:**
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Add the server configuration:**

   ```json
   {
     "mcpServers": {
       "anomaly-detection": {
         "command": "python",
         "args": [
           "C:/Dev/Watchtower/mcp-server-ad/server.py"
         ],
         "env": {}
       }
     }
   }
   ```

   **Note:** Adjust the path to `server.py` based on your actual installation path.

3. **Restart Claude Desktop** for changes to take effect.

### For Cursor IDE

1. **Open Cursor Settings** (File → Preferences → Settings)

2. **Search for "MCP"** or navigate to MCP settings

3. **Add server configuration:**

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

4. **Restart Cursor** for changes to take effect.

### For Other MCP Clients

Most MCP clients follow a similar configuration pattern. The server communicates via stdio (standard input/output), so you need:

- **Command**: `python` (or `python3` on Linux/Mac)
- **Args**: Path to `server.py`
- **Working Directory**: Optional, but recommended to set to the `mcp-server-ad` directory

#### Example Configuration (Generic)

```json
{
  "mcpServers": {
    "anomaly-detection": {
      "command": "python",
      "args": [
        "/absolute/path/to/mcp-server-ad/server.py"
      ],
      "cwd": "/absolute/path/to/mcp-server-ad"
    }
  }
}
```

## Using Python Virtual Environment (Recommended)

For better dependency management, use a virtual environment:

### 1. Create Virtual Environment

```bash
cd mcp-server-ad
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Update MCP Configuration

When using a virtual environment, update your MCP config to use the venv Python:

**Windows:**
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

**macOS/Linux:**
```json
{
  "mcpServers": {
    "anomaly-detection": {
      "command": "/absolute/path/to/mcp-server-ad/venv/bin/python",
      "args": [
        "/absolute/path/to/mcp-server-ad/server.py"
      ]
    }
  }
}
```

## Verifying the Setup

### 1. Check Server Starts

Run the server directly to verify it starts:

```bash
python server.py
```

You should see the server waiting for input (it reads from stdin).

### 2. Test via MCP Client

Once configured in your MCP client:

1. **Restart your MCP client** (Claude Desktop, Cursor, etc.)
2. **Look for the server** in your MCP tools/menu
3. **Try calling the tool**: `detect_anomalies`

### 3. Test the Tool

Use the `detect_anomalies` tool with sample data:

```json
{
  "data": "[{\"date\":\"2024-01-01\",\"sales\":100},{\"date\":\"2024-01-02\",\"sales\":500}]",
  "time_column": "date",
  "value_column": "sales",
  "methods": ["moving_average", "standard_deviation"]
}
```

## Troubleshooting

### Server Not Appearing in Client

1. **Check Python path**: Make sure `python` is in your PATH
2. **Check file path**: Verify the absolute path to `server.py` is correct
3. **Check permissions**: Ensure the script has execute permissions
4. **Check logs**: Look for error messages in your MCP client's logs

### Import Errors

If you see import errors:

1. **Verify dependencies are installed:**
   ```bash
   pip list | grep -E "(fastmcp|pandas|numpy)"
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Use virtual environment** (recommended)

### Path Issues on Windows

- Use forward slashes `/` or double backslashes `\\` in paths
- Use absolute paths, not relative paths
- Enclose paths with spaces in quotes

### Testing Connection

You can test the server manually:

```bash
# Start server
python server.py

# In another terminal, test with echo (if supported)
echo '{"method":"tools/list"}' | python server.py
```

## Alternative: Using as Python Module

You can also use the server programmatically without MCP:

```python
from server import detect_anomalies_core

result = detect_anomalies_core(
    data=[{"date": "2024-01-01", "sales": 100}],
    time_column="date",
    value_column="sales",
    methods=["moving_average"]
)
```

## Next Steps

- See [TESTING.md](TESTING.md) for testing instructions
- See [README.md](README.md) for usage examples
- Check the server logs for any issues

