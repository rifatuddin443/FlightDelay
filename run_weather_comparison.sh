#!/bin/bash
# Weather Impact Comparison - Quick Start Script
# ================================================
#
# This script runs the complete weather analysis pipeline in one command.
#
# Usage:
#   ./run_weather_comparison.sh
#   ./run_weather_comparison.sh --epochs 100 --classifier both
#
# Prerequisites:
#   - Python 3.8+
#   - PyTorch, tsai, PyG installed
#   - Weather data files (weather_cn.npy or similar)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Default parameters
EPOCHS=50
CLASSIFIER="TSiTPlus"
BATCH_SIZE=16
LR="1e-4"
DATA_SOURCE="cdata"
DEVICE="auto"
SKIP_ANALYSIS="false"
PLOT="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --classifier)
            CLASSIFIER="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --data_source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-analysis)
            SKIP_ANALYSIS="true"
            shift
            ;;
        --plot)
            PLOT="true"
            shift
            ;;
        --help)
            echo ""
            echo "Weather Impact Comparison - Quick Start"
            echo "========================================"
            echo ""
            echo "Usage: ./run_weather_comparison.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N              Number of training epochs (default: 50)"
            echo "  --classifier NAME       TSiTPlus, ConvTranPlus, or both (default: TSiTPlus)"
            echo "  --batch_size N          Batch size (default: 16)"
            echo "  --lr LR                 Learning rate (default: 1e-4)"
            echo "  --data_source SOURCE    cdata or udata (default: cdata)"
            echo "  --device DEVICE         cuda, cpu, or auto (default: auto)"
            echo "  --no-analysis           Skip analysis step"
            echo "  --plot                  Generate plots (requires matplotlib)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_weather_comparison.sh --epochs 100"
            echo "  ./run_weather_comparison.sh --classifier both --epochs 50 --plot"
            echo "  ./run_weather_comparison.sh --batch_size 32 --lr 5e-5"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Verify Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Print header
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         WEATHER IMPACT COMPARISON - QUICK START               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration summary
echo -e "${GREEN}Configuration:${NC}"
echo "  Epochs:       $EPOCHS"
echo "  Classifier:   $CLASSIFIER"
echo "  Batch size:   $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Data source:  $DATA_SOURCE"
echo "  Device:       $DEVICE"
echo ""

# Check for required files
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check Python packages
MISSING_PACKAGES=()
for pkg in torch torchvision torch_geometric; do
    python -c "import $pkg" 2>/dev/null || MISSING_PACKAGES+=("$pkg")
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${RED}Missing packages: ${MISSING_PACKAGES[@]}${NC}"
    echo "Install with: pip install torch torch-geometric"
    exit 1
fi

# Check comparison script exists
if [ ! -f "stacked_gru_transformer_weather_comparison.py" ]; then
    echo -e "${RED}Error: stacked_gru_transformer_weather_comparison.py not found${NC}"
    echo "Make sure you're in the STPN-main directory"
    exit 1
fi

if [ ! -f "analyze_weather_comparison.py" ]; then
    echo -e "${YELLOW}Warning: analyze_weather_comparison.py not found${NC}"
    SKIP_ANALYSIS="true"
fi

echo -e "${GREEN}✓ All dependencies found${NC}"
echo ""

# Run main comparison
echo -e "${BLUE}Step 1/2: Running comparison experiment...${NC}"
echo -e "${YELLOW}Command: python stacked_gru_transformer_weather_comparison.py${NC}"
echo "         --epochs $EPOCHS --classifier $CLASSIFIER --batch_size $BATCH_SIZE"
echo "         --lr $LR --data_source $DATA_SOURCE --device $DEVICE"
echo ""

python stacked_gru_transformer_weather_comparison.py \
    --epochs "$EPOCHS" \
    --classifier "$CLASSIFIER" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --data_source "$DATA_SOURCE" \
    --device "$DEVICE"

RESULT_DIR=$(ls -td weather_comparison_* 2>/dev/null | head -1)

if [ -z "$RESULT_DIR" ]; then
    echo -e "${RED}Error: No results directory found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Comparison complete${NC}"
echo -e "${BLUE}Results saved to: $RESULT_DIR${NC}"
echo ""

# Display summary immediately
if [ "$SKIP_ANALYSIS" = "false" ]; then
    echo -e "${BLUE}Step 2/2: Analyzing results...${NC}"
    echo ""
    
    if [ "$PLOT" = "true" ]; then
        python analyze_weather_comparison.py "$RESULT_DIR" --plot
    else
        python analyze_weather_comparison.py "$RESULT_DIR"
    fi
    
    echo ""
fi

# Show summary CSV path
if [ -f "$RESULT_DIR/WEATHER_COMPARISON_SUMMARY.csv" ]; then
    echo ""
    echo -e "${BLUE}Summary Results (WEATHER_COMPARISON_SUMMARY.csv):${NC}"
    head -5 "$RESULT_DIR/WEATHER_COMPARISON_SUMMARY.csv"
    echo ""
fi

# Final summary
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    WORKFLOW COMPLETE                          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Output directory:${NC}  $RESULT_DIR/"
echo ""
echo -e "${BLUE}Key outputs:${NC}"
echo "  • WEATHER_COMPARISON_SUMMARY.csv   - Main results"
echo "  • {classifier}_WITH_weather_*.csv  - Detailed metrics"
echo "  • {classifier}_NO_weather_*.csv    - Baseline metrics"
echo "  • *.pth                            - Model checkpoints"
if [ -f "$RESULT_DIR/weather_comparison_plots.png" ]; then
    echo "  • weather_comparison_plots.png     - Visualizations"
fi
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review results in: $RESULT_DIR/"
echo "  2. Check WEATHER_COMPARISON_SUMMARY.csv for impact metrics"
echo "  3. Run analyze again for detailed report:"
echo "     python analyze_weather_comparison.py $RESULT_DIR/ --plot"
echo "  4. Update production config based on results"
echo ""

