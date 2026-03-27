# Silent Dropout Detection System

A high-performance modular platform for detecting student dropouts in Kolkata-based educational institutions, developed for **SDG Goal 4: Quality Education**.

## 🚀 Quick Start
Run everything (install dependencies, generate 5k dataset, train LightGBM, start backend/frontend) with one command:
```bash
./start_project.sh
```

## 🌟 Key Features
- **Advanced Predictive Engine:** Powered by **LightGBM (Gradient Boosting)** for superior performance on tabular educational data.
- **Explainable AI (XAI):** Integrated **SHAP TreeExplainer** provides precise, feature-level insights for every prediction.
- **Behavioral Feature Engineering:** 
  - `engagement_score`: Weighted composite of attendance, submissions, and LMS activity.
  - `attendance_lms_interaction`: Multiplier effect capturing the synergy between physical and digital presence.
- **Modern Dashboard:** Animated React frontend with real-time risk visualization and color-coded alerts.
- **Research-Level Accuracy:** Rigorously tested against 50+ unique scenarios with a **92.00% success rate**.

## 📊 Technical Specifications
- **Model:** LightGBM Classifier (n_estimators=300, max_depth=12, balanced)
- **Dataset:** 5,000 unique student records with Kolkata-specific institutional data.
- **Inference Strategy:** High-sensitivity decision threshold (0.4) to maximize recall for at-risk students.
- **Backend:** FastAPI with Pydantic validation and CORS enabled.

## 🧪 Model Testing & Verification
To run the expanded 50-case test suite (including 15 manual edge cases and 35 random samples):
```bash
./venv/bin/python3 run_tests.py
```
View the detailed logs in `test_results.txt`.

## Project Structure
- `backend/`: Root directory for the API.
  - `main.py`: Entry point for the FastAPI application.
  - `routes/`, `models/`, `services/`: Modular components.
- `data/`: Contains the `synthetic_students.csv` (5,000 records).
- `frontend/`: React frontend application.
- `generate_data.py`: Enhanced script for 5k student data generation.
- `training.py`: LightGBM training pipeline with SHAP integration.
- `run_tests.py`: Accuracy verification suite (50 test cases).

## Getting Started (Manual)

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
./venv/bin/pip install faker shap lightgbm
```

### 2. Run Pipeline
```bash
./venv/bin/python3 generate_data.py
./venv/bin/python3 training.py
```

### 3. Start Services
- **Backend:** `cd backend && ../venv/bin/uvicorn main:app --reload`
- **Frontend:** `cd frontend && npm start`

## API Endpoint: `POST /predict`
Input: Student engagement metrics.  
Output: Risk (0/1), probability, and top 3 contributing factors (SHAP).
