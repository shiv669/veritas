#!/bin/bash
# Run Veritas with UI integration
# This script starts both the Veritas RAG API and the OpenWebUI instance

# Set default ports
VERITAS_API_PORT=8000
UI_PORT=8080

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-port)
      VERITAS_API_PORT="$2"
      shift 2
      ;;
    --ui-port)
      UI_PORT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--api-port PORT] [--ui-port PORT]"
      echo "  --api-port PORT  Port for the Veritas RAG API (default: 8000)"
      echo "  --ui-port PORT   Port for the OpenWebUI (default: 8080)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Create a function to handle cleanup on exit
function cleanup {
    echo "Stopping all services..."
    kill $(jobs -p) 2>/dev/null
    echo "Cleanup complete."
}

# Register the cleanup function on script exit
trap cleanup EXIT

# Start the Veritas RAG API
echo "Starting Veritas RAG API on port $VERITAS_API_PORT..."
python "$PROJECT_ROOT/scripts/veritas_api.py" --port "$VERITAS_API_PORT" &

# Wait for the API to start
sleep 3

# Start the OpenWebUI backend
echo "Starting OpenWebUI backend on port $UI_PORT..."
cd "$PROJECT_ROOT/ui/backend"
./dev.sh &

# Wait for the UI backend to start
sleep 3

# Print usage instructions
echo "===================================================================="
echo "🚀 Services started successfully!"
echo "===================================================================="
echo "Veritas RAG API: http://localhost:$VERITAS_API_PORT/docs"
echo "OpenWebUI: http://localhost:$UI_PORT"
echo "===================================================================="
echo "To initialize Veritas in the UI:"
echo "1. Log in to the OpenWebUI"
echo "2. Go to Settings"
echo "3. Navigate to the 'Veritas' section"
echo "4. Click 'Initialize Veritas RAG'"
echo "===================================================================="
echo "Press Ctrl+C to stop all services"
echo "===================================================================="

# Wait for user to press Ctrl+C
wait 