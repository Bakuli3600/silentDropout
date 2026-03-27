# Silent Dropout Detection System

A modular platform for detecting silent dropouts in Kolkata-based educational institutions.

## Tech Stack
- **Backend:** FastAPI, Uvicorn
- **ML/Data:** Pandas, Numpy, Scikit-Learn, Joblib, SHAP, Matplotlib
- **Frontend:** React (TypeScript)

## Project Structure
- `backend/`: Root directory for the API.
  - `main.py`: Entry point for the FastAPI application.
  - `app/`, `routes/`, `models/`, `services/`: Modular API components.
- `data/`: Contains datasets, including `synthetic_students.csv`.
- `frontend/`: React frontend application.
- `generate_data.py`: Script to generate synthetic student engagement data.
- `training.py`: ML pipeline for training the dropout prediction model.

## Getting Started

### 1. Data Generation
Generate the synthetic student dataset:
```bash
./venv/bin/python3 generate_data.py
```

### 2. Model Training
Train the RandomForest model:
```bash
./venv/bin/python3 training.py
```

### 3. Backend API
1. Install dependencies: `pip install -r backend/requirements.txt`
2. Run the server: `cd backend && ../venv/bin/uvicorn main:app --reload`

### 4. Frontend
1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
