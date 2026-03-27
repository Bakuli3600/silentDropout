# Silent Dropout Detection System

A modular platform for detecting student dropouts in Kolkata-based educational institutions, developed for **SDG Goal 4: Quality Education**.

## 🚀 Quick Start
Run everything (install dependencies, generate data, train model, start backend/frontend) with one command:
```bash
./start_project.sh
```

## Features
- **Early Prediction:** Uses behavioral and academic patterns (attendance, LMS activity, grades) to identify at-risk students.
- **Explainable AI (XAI):** Integrated SHAP insights to provide clear explanations for each prediction.
- **Modern Dashboard:** Animated React frontend for easy data entry and risk visualization.
- **Optimized Performance:** Multi-core RandomForest training (8 threads).
- **Verified Accuracy:** Built-in testing suite to validate model performance across 20+ unique scenarios.

## Tech Stack
- **Backend:** FastAPI, Uvicorn, Pydantic
- **ML/Data:** Scikit-Learn, Pandas, SHAP
- **Frontend:** React, TypeScript, CSS3 Animations

## Project Structure
- `backend/`: Root directory for the API.
  - `main.py`: Entry point for the FastAPI application.
  - `app/`, `routes/`, `models/`, `services/`: Modular API components.
- `data/`: Contains datasets, including `synthetic_students.csv`.
- `frontend/`: React frontend application.
- `generate_data.py`: Script to generate synthetic student engagement data.
- `training.py`: ML pipeline for training the dropout prediction model.
- `run_tests.py`: Accuracy verification script with 20+ test cases.
- `test_results.txt`: Log of the latest test results and model accuracy.

## 🧪 Model Testing & Accuracy
To verify the system's performance, run the comprehensive test suite:
```bash
./venv/bin/python3 run_tests.py
```
This script evaluates the model against 20 unique student profiles (including edge cases and random samples) and generates a report in `test_results.txt`.

## Getting Started (Manual)

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
./venv/bin/pip install faker shap
```

### 2. Generate Data & Train Model
```bash
./venv/bin/python3 generate_data.py
./venv/bin/python3 training.py
```

### 3. Run Backend (API)
```bash
cd backend
../venv/bin/uvicorn main:app --reload
```

### 4. Run Frontend (Dashboard)
```bash
cd frontend
npm install
npm start
```

## API Endpoint: `POST /predict`
Input: Student engagement metrics.  
Output: Risk (0/1), probability, and top 3 contributing factors.
