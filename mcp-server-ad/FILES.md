# Project Files Overview

This document explains all the files in the `mcp-server-ad` directory.

## Core Files

### `server.py`
**Main MCP server implementation**
- Contains the FastMCP server setup
- Implements three anomaly detection methods (moving average, standard deviation, IQR)
- Exposes `detect_anomalies` tool for MCP clients
- Can be run standalone: `python server.py`

### `requirements.txt`
**Python dependencies**
- Lists all required packages: fastmcp, pandas, numpy
- Install with: `pip install -r requirements.txt`

## Documentation

### `README.md`
**Main documentation**
- Overview of the server
- Features and usage examples
- Quick start guide

### `SETUP.md`
**Detailed setup instructions**
- Installation steps
- Configuration for Claude Desktop, Cursor, and other MCP clients
- Virtual environment setup
- Troubleshooting guide

### `GETTING_STARTED.md`
**Comprehensive getting started guide**
- Step-by-step instructions
- Configuration examples
- Usage examples
- Troubleshooting

### `QUICK_START.md`
**Quick reference guide**
- Fast setup instructions
- Quick configuration examples
- Common commands

### `TESTING.md`
**Testing guide**
- How to test the server
- Test examples
- Testing scenarios

### `FILES.md`
**This file** - Overview of all project files

## Configuration Examples

### `claude_desktop_config.example.json`
**Example Claude Desktop configuration**
- Shows how to configure the server in Claude Desktop
- Copy this format to your Claude Desktop config file

## Setup Scripts

### `setup.ps1`
**PowerShell setup script for Windows**
- Automates installation on Windows
- Creates virtual environment
- Installs dependencies
- Shows next steps

**Usage:**
```powershell
.\setup.ps1
```

### `setup.sh`
**Bash setup script for macOS/Linux**
- Automates installation on macOS/Linux
- Creates virtual environment
- Installs dependencies
- Shows next steps

**Usage:**
```bash
chmod +x setup.sh
./setup.sh
```

## Testing Files

### `test_server.py`
**Comprehensive test suite**
- 6 test scenarios covering all functionality
- Tests basic functionality, aggregation, all methods, auto-detection, CSV data, and error handling
- Run with: `python test_server.py`

### `example_usage.py`
**Example usage script**
- Shows how to use the server
- Demonstrates different scenarios
- Run with: `python example_usage.py`

## File Structure

```
mcp-server-ad/
├── server.py                          # Main MCP server
├── requirements.txt                   # Dependencies
├── README.md                          # Main documentation
├── SETUP.md                           # Detailed setup guide
├── GETTING_STARTED.md                 # Getting started guide
├── QUICK_START.md                     # Quick reference
├── TESTING.md                         # Testing guide
├── FILES.md                           # This file
├── claude_desktop_config.example.json # Example config
├── setup.ps1                          # Windows setup script
├── setup.sh                           # macOS/Linux setup script
├── test_server.py                     # Test suite
└── example_usage.py                   # Usage examples
```

## Quick Reference

| File | Purpose | Command |
|------|---------|---------|
| `server.py` | Main server | `python server.py` |
| `test_server.py` | Run tests | `python test_server.py` |
| `example_usage.py` | See examples | `python example_usage.py` |
| `setup.ps1` | Windows setup | `.\setup.ps1` |
| `setup.sh` | Mac/Linux setup | `./setup.sh` |

## Next Steps

1. **New to the project?** Start with [QUICK_START.md](QUICK_START.md)
2. **Setting up?** See [SETUP.md](SETUP.md)
3. **Want details?** Read [GETTING_STARTED.md](GETTING_STARTED.md)
4. **Ready to test?** See [TESTING.md](TESTING.md)
5. **Need help?** Check [README.md](README.md)

