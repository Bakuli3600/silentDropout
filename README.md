# Silent Dropout Detection System

A modular platform for detecting silent dropouts in Kolkata-based educational institutions.

## Tech Stack
- **Backend:** FastAPI, Uvicorn
- **ML/Data:** Pandas, Numpy, Scikit-Learn, Joblib, SHAP, Matplotlib
- **Frontend:** React (TypeScript)
- **Data Generation:** Faker, Numpy, Pandas

## Project Structure
- `backend/`: Root directory for the API.
  - `main.py`: Entry point for the FastAPI application.
  - `app/`, `routes/`, `models/`, `services/`: Modular API components.
- `data/`: Contains datasets, including `synthetic_students.csv`.
- `frontend/`: React frontend application.
- `generate_data.py`: Script to generate synthetic student engagement data.

## Getting Started

### Data Generation
1. Generate the synthetic student dataset:
   ```bash
   ./venv/bin/python3 generate_data.py
   ```

### Backend
1. Install dependencies: `pip install -r backend/requirements.txt`
2. Run the server: `cd backend && ../venv/bin/uvicorn main:app --reload`

### Frontend
1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
