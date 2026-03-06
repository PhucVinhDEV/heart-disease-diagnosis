#!/bin/bash

# Heart Disease Prediction App Deployment Script
echo "🚀 Deploying Heart Disease Prediction App..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install system dependencies (for XGBoost/LightGBM)
echo "🔧 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y libomp-dev gcc g++
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    sudo yum install -y libgomp gcc gcc-c++
elif command -v brew &> /dev/null; then
    # macOS
    brew install libomp
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/saved_models/latest
mkdir -p experiments
mkdir -p data/processed
mkdir -p logs

# Set permissions
echo "🔐 Setting permissions..."
chmod +x deploy.sh
chmod +x start.sh

echo "✅ Deployment completed successfully!"
echo ""
echo "🚀 To start the app, run:"
echo "   ./start.sh"
echo ""
echo "🌐 The app will be available at: http://localhost:8501"
