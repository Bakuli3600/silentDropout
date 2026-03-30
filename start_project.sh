#!/bin/bash

# --- Silent Dropout Detection System: Professional Pipeline Execution ---

echo "============================================================"
echo "SILENT DROPOUT DETECTION SYSTEM (KOLKATA PILOT)"
echo "============================================================"

# Function to check and free ports
free_port() {
    PORT=$1
    PID=$(lsof -ti :$PORT)
    if [ ! -z "$PID" ]; then
        echo "INFO: Freeing port $PORT (killing process $PID)..."
        kill -9 $PID
    fi
}

# 1. Port Cleanup
echo "INFO: Cleaning up existing processes..."
free_port 8000
free_port 3000

# 2. Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo "INFO: Creating virtual environment..."
    python3 -m venv venv
fi

echo "INFO: Installing Backend Dependencies..."
./venv/bin/pip install -r requirements.txt --quiet

# 3. Data & Model Pipeline
echo "INFO: Regenerating Synthetic Dataset (5,000 students)..."
./venv/bin/python3 generate_data.py

echo "INFO: Training Professional LightGBM Model..."
./venv/bin/python3 training.py

echo "INFO: Executing Comprehensive Test Suite..."
./venv/bin/python3 run_tests.py

# 4. Frontend Setup
echo "INFO: Setting up Professional React Dashboard..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "INFO: Installing Frontend Dependencies (this may take a minute)..."
    npm install --silent
fi
cd ..

# 5. Start Backend
echo "INFO: Starting FastAPI Backend on http://0.0.0.0:8000..."
cd backend
nohup ../venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Launch React Frontend on http://localhost:3000...
echo "------------------------------------------------------------"
echo "✅ Pipeline Ready! Access the dashboard at http://localhost:3000"
echo "------------------------------------------------------------"

cd frontend
npm start
