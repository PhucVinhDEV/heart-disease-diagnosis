#!/bin/bash

###############################################################################
# Heart Disease Diagnosis - Enhanced App Launcher
# Team: Dũng, Anh, Vinh, Hằng, Huy | AIO2025 VietAI
###############################################################################

echo "════════════════════════════════════════════════════════════════════════"
echo "🫀 Heart Disease Diagnosis - Enhanced App"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment not activated${NC}"
    echo "Activating venv..."
    
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    else
        echo -e "${RED}❌ Virtual environment not found${NC}"
        echo "Please create it first: python3 -m venv venv"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Virtual environment already activated${NC}"
fi

echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import streamlit, pandas, plotly, optuna" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Installing missing dependencies...${NC}"
    pip install -r requirements.txt
fi

echo ""

# Check if models exist
echo "Checking model files..."
MODEL_DIR="models/saved_models/latest"

if [ -d "$MODEL_DIR" ]; then
    MODEL_COUNT=$(ls -1 $MODEL_DIR/*.pkl 2>/dev/null | wc -l)
    echo -e "${GREEN}✅ Found $MODEL_COUNT model files in $MODEL_DIR${NC}"
    
    if [ $MODEL_COUNT -lt 9 ]; then
        echo -e "${YELLOW}⚠️  Expected 9 models, found $MODEL_COUNT${NC}"
        echo "Some models may be missing. Please check notebooks/latest.ipynb"
    fi
else
    echo -e "${RED}❌ Model directory not found: $MODEL_DIR${NC}"
    echo "Please train models first using notebooks/latest.ipynb"
    exit 1
fi

echo ""

# Check if experiments directory exists
if [ ! -d "experiments" ]; then
    echo "Creating experiments directory..."
    mkdir -p experiments/logs experiments/results experiments/reports
    echo -e "${GREEN}✅ Experiments directory created${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo -e "${BLUE}🚀 Launching Enhanced Streamlit App...${NC}"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "App will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo ""

# Run the app (from project root)
cd "$(dirname "$0")/.." && streamlit run app/streamlit_app.py

# Cleanup message
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "App stopped. Thank you for using Heart Disease Diagnosis System!"
echo "════════════════════════════════════════════════════════════════════════"
