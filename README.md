# Silent Dropout Detection System

A modular API for detecting silent dropouts using machine learning.

## Tech Stack
- **Backend:** FastAPI, Uvicorn
- **ML/Data:** Pandas, Numpy, Scikit-Learn, Joblib, SHAP, Matplotlib
- **Frontend:** React (TypeScript)

## Project Structure
- `backend/`: Root directory for the API.
  - `main.py`: Entry point for the FastAPI application.
  - `app/`: Application-specific logic and configurations.
  - `routes/`: API endpoints (e.g., `/health`).
  - `models/`: Data models and ML schemas.
  - `services/`: Business logic and ML processing.
- `frontend/`: React frontend application.

## Getting Started

### Backend
1. Create a virtual environment: `python3 -m venv venv`
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Run the server: `cd backend && ../venv/bin/uvicorn main:app --reload`

### Frontend
1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
