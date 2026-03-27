#!/bin/bash

# --- Silent Dropout Detection System: Startup Script ---

echo "🚀 Starting Silent Dropout Detection System Setup..."

# 1. Setup Python Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "📥 Installing Backend Dependencies..."
./venv/bin/pip install -r backend/requirements.txt
./venv/bin/pip install faker shap

# 2. Data & Model Generation
if [ ! -f "data/synthetic_students.csv" ]; then
    echo "📊 Generating synthetic student data (Kolkata Pilot)..."
    ./venv/bin/python3 generate_data.py
fi

if [ ! -f "backend/models/dropout_model.joblib" ]; then
    echo "🤖 Training ML Model (RandomForest Optimized)..."
    ./venv/bin/python3 training.py
fi

# 3. Start Backend (in background)
echo "🌐 Starting FastAPI Backend on http://127.0.0.1:8000..."
cd backend
../venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 4. Start Frontend
echo "💻 Starting React Frontend on http://localhost:3000..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Frontend Dependencies (this may take a minute)..."
    npm install
fi

# Cleanup on exit
trap "kill $BACKEND_PID; exit" INT TERM EXIT

npm start
