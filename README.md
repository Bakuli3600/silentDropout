# Silent Dropout Detection System

A modular API for detecting silent dropouts using machine learning.

## Tech Stack
- **Backend:** FastAPI, Uvicorn
- **ML/Data:** Pandas, Numpy, Scikit-Learn, Joblib, SHAP, Matplotlib

## Project Structure
- `backend/app`: FastAPI entry point and application factory.
- `backend/routes`: API endpoints (e.g., `/health`).
- `backend/models`: Data models and ML schemas.
- `backend/services`: Business logic and ML processing.

## Getting Started
1. Create a virtual environment: `python3 -m venv venv`
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Run the server: `cd backend && ../venv/bin/uvicorn app.main:app --reload`
