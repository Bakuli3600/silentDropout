#!/bin/bash

# --- Silent Dropout Detection System: Startup Script ---

echo "INFO: Starting Silent Dropout Detection System Setup..."

# 1. Setup Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "INFO: Creating virtual environment..."
    python3 -m venv venv
fi

echo "INFO: Installing Backend Dependencies..."
./venv/bin/pip install -r backend/requirements.txt
./venv/bin/pip install faker shap lightgbm

# 2. Data & Model Generation
if [ ! -f "data/synthetic_students.csv" ]; then
    echo "INFO: Generating synthetic student data (Kolkata Pilot - 5k Dataset)..."
    ./venv/bin/python3 generate_data.py
fi

if [ ! -f "backend/models/dropout_model.joblib" ]; then
    echo "INFO: Training ML Model (LightGBM Optimized)..."
    ./venv/bin/python3 training.py
    echo "INFO: Running Model Verification Tests..."
    ./venv/bin/python3 run_tests.py
fi

# 3. Start Backend (in background)
echo "INFO: Starting FastAPI Backend on http://127.0.0.1:8000..."
cd backend
../venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 4. Start Frontend
echo "INFO: Starting React Frontend on http://localhost:3000..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "INFO: Installing Frontend Dependencies (this may take a minute)..."
    npm install
fi

# Cleanup on exit
trap "kill $BACKEND_PID; exit" INT TERM EXIT

npm start
