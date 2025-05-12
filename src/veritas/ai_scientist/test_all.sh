#!/bin/bash
# Script to run all tests for Veritas AI Scientist

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VERITAS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Create logs directory if it doesn't exist
mkdir -p "${VERITAS_ROOT}/logs"

# Activate Python virtual environment if it exists
if [ -d "${VERITAS_ROOT}/venv" ]; then
    source "${VERITAS_ROOT}/venv/bin/activate"
fi

# Set the PYTHONPATH
export PYTHONPATH="${VERITAS_ROOT}:${PYTHONPATH}"

# Echo separator
echo "========================================"
echo "  Veritas AI Scientist Test Suite"
echo "========================================"

# Run the system tests
echo -e "\n1. Running system tests..."
python "${SCRIPT_DIR}/test_system.py"
SYSTEM_STATUS=$?

# Run the simple test
echo -e "\n2. Running simple idea generation test..."
python "${SCRIPT_DIR}/test_simple.py"
SIMPLE_STATUS=$?

# Show the status
echo -e "\n========================================"
echo "  Test Results"
echo "========================================"
echo "System Tests: $([ $SYSTEM_STATUS -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "Simple Test: $([ $SIMPLE_STATUS -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "========================================"

# Make the script executable
if [ $SYSTEM_STATUS -eq 0 ] && [ $SIMPLE_STATUS -eq 0 ]; then
    echo -e "\nAll tests passed! ✨"
    exit 0
else
    echo -e "\nSome tests failed. Check logs for details. 😢"
    exit 1
fi 