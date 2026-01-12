#!/bin/bash
# File: setup.sh
# Termux setup script for CRA-POM v2 - Geometric Cognition Kernel
# Run this script in Termux to set up the development environment

set -e

echo "=========================================="
echo " CRA-POM v2 - Geometric Cognition Kernel"
echo " Termux Setup Script"
echo "=========================================="
echo ""

# Check if running in Termux
if [ -z "$PREFIX" ]; then
    echo "Warning: \$PREFIX not set. This script is designed for Termux."
    echo "Continuing anyway..."
else
    echo "Detected Termux environment: $PREFIX"
fi

# Update package lists
echo ""
echo "[1/5] Updating package lists..."
pkg update -y

# Install Node.js
echo ""
echo "[2/5] Installing Node.js..."
pkg install -y nodejs

# Verify Node.js installation
echo ""
echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "[3/5] Setting up project in: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Install dependencies
echo ""
echo "[4/5] Installing npm dependencies..."
npm install

# Create directory structure (if not exists)
echo ""
echo "[5/5] Verifying directory structure..."
mkdir -p src/core
mkdir -p src/components

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "To start the development server, run:"
echo ""
echo "  cd $SCRIPT_DIR"
echo "  npm run dev"
echo ""
echo "Then open your browser to the URL shown"
echo "(usually http://localhost:5173)"
echo ""
echo "For mobile access on the same network:"
echo "  http://<your-ip>:5173"
echo ""
echo "Find your IP with: ip addr show wlan0"
echo ""
