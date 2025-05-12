#!/bin/bash
# Script to run Veritas AI Scientist research assistant

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

# Run the AI research assistant
echo "Running Veritas AI Scientist..."
python "${SCRIPT_DIR}/run_scientist.py" --phase idea --num-ideas 1 --use-direct-implementation $@

# Show the status
if [ $? -eq 0 ]; then
    echo "✅ Successfully completed!"
    echo "Generated ideas saved in ${VERITAS_ROOT}/results/"
else
    echo "❌ Failed to run Veritas AI Scientist"
    echo "See logs for details"
fi 