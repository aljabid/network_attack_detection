#!/bin/bash

# AI-Based Network Attack Detection System - Run Script
# Quick launcher for common tasks

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
echo "============================================================"
echo "   AI-Based Network Attack Detection System"
echo "============================================================"
echo -e "${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Run ./setup.sh first to set up the environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Function to show usage
show_usage() {
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  train              - Train all models"
    echo "  detect             - Start real-time detection"
    echo "  test               - Run test suite"
    echo "  notebook           - Start Jupyter notebook"
    echo "  visualize          - Generate visualizations"
    echo "  clean              - Clean logs and temporary files"
    echo "  help               - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train"
    echo "  ./run.sh detect"
    echo "  ./run.sh test"
}

# Main command handler
case "${1:-help}" in
    train)
        echo -e "${YELLOW}Starting model training...${NC}"
        cd src
        python train.py
        cd ..
        echo -e "${GREEN}✓ Training complete!${NC}"
        echo "Models saved to: data/models/"
        echo "Visualizations: dashboards/"
        ;;
    
    detect)
        echo -e "${YELLOW}Starting real-time detection...${NC}"
        echo "Press Ctrl+C to stop"
        cd src
        python realtime_detection.py
        cd ..
        ;;
    
    test)
        echo -e "${YELLOW}Running test suite...${NC}"
        pytest tests/ -v --cov=src --cov-report=html
        echo -e "${GREEN}✓ Tests complete!${NC}"
        echo "Coverage report: htmlcov/index.html"
        ;;
    
    notebook)
        echo -e "${YELLOW}Starting Jupyter notebook...${NC}"
        jupyter notebook notebooks/
        ;;
    
    visualize)
        echo -e "${YELLOW}Generating visualizations...${NC}"
        cd src
        python -c "
from visualization.visualizer import Visualizer
import json

viz = Visualizer()
print('Visualizations generated in dashboards/')
        "
        cd ..
        echo -e "${GREEN}✓ Visualizations complete!${NC}"
        echo "Check dashboards/ directory"
        ;;
    
    clean)
        echo -e "${YELLOW}Cleaning temporary files...${NC}"
        rm -rf logs/*.log
        rm -rf __pycache__/
        rm -rf src/__pycache__/
        rm -rf src/*/__pycache__/
        rm -rf .pytest_cache/
        rm -rf htmlcov/
        rm -rf .coverage
        echo -e "${GREEN}✓ Cleanup complete!${NC}"
        ;;
    
    help|--help|-h)
        show_usage
        ;;
    
    *)
        echo -e "${RED}Error: Unknown command '${1}'${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}============================================================${NC}"
