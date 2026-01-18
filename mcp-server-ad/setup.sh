#!/bin/bash
# Setup script for macOS/Linux
# This script helps set up the MCP server

echo "Setting up Anomaly Detection MCP Server..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "✗ Python not found. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "✓ Python found: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Test the server: python test_server.py"
echo "2. Configure in Claude Desktop config:"
echo "   ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)"
echo "   ~/.config/Claude/claude_desktop_config.json (Linux)"
echo "3. Use this path in config:"
SCRIPT_PATH=$(realpath server.py)
echo "   $SCRIPT_PATH"
echo "4. Or use venv Python:"
VENV_PYTHON=$(realpath venv/bin/python)
echo "   $VENV_PYTHON"

