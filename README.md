# Silent Dropout Detection System

A modular platform for detecting student dropouts in Kolkata-based educational institutions, developed for **SDG Goal 4: Quality Education**.

## Features
- **Early Prediction:** Uses behavioral and academic patterns (attendance, LMS activity, grades) to identify at-risk students.
- **Explainable AI (XAI):** Integrated SHAP insights to provide clear explanations for each prediction.
- **Modern Dashboard:** Animated React frontend for easy data entry and risk visualization.
- **Optimized Performance:** Multi-core RandomForest training (8 threads).

## Tech Stack
- **Backend:** FastAPI, Uvicorn, Pydantic
- **ML/Data:** Scikit-Learn, Pandas, SHAP
- **Frontend:** React, TypeScript, CSS3 Animations

## Getting Started

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
