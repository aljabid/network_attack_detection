#!/bin/bash

# AI-Based Network Attack Detection System - Setup Script
# This script sets up the environment and prepares the system

set -e  # Exit on error

echo "====================================================================="
echo "  AI-Based Network Attack Detection System - Setup"
echo "====================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}Error: Python 3.8 or higher is required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment
echo ""
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo ""
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data/raw/cicids2017
mkdir -p data/raw/unsw_nb15
mkdir -p data/processed
mkdir -p data/models
mkdir -p logs
mkdir -p dashboards
mkdir -p notebooks
echo -e "${GREEN}✓ Directories created${NC}"

# Download sample data (optional)
echo ""
echo -e "${YELLOW}Sample Data Setup${NC}"
echo "To use real datasets, please download:"
echo "  1. CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html"
echo "  2. UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
echo ""
echo "Place CSV files in:"
echo "  - data/raw/cicids2017/"
echo "  - data/raw/unsw_nb15/"
echo ""
echo "Or the system will generate sample data for testing."

# Setup complete
echo ""
echo "====================================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "====================================================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Download datasets (optional - see above)"
echo "  3. Train models: cd src && python train.py"
echo "  4. Run real-time detection: python realtime_detection.py"
echo ""
echo "For more information, see README.md"
echo "====================================================================="
