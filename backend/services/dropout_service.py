import os
import joblib
import sys
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List

# Add the project root to sys.path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.explainability import get_top_factors
from ml.intervention import recommend_intervention

class DropoutService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.explainer = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.features = model_data['features']
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Feature engineering logic
        normalized_lms = (input_data['lms_login_frequency'] / 14) * 100
        engagement_score = (
            input_data['attendance_rate'] * 0.3 + 
            input_data['assignment_submission_rate'] * 0.4 + 
            normalized_lms * 0.3
        )
        
        # Additional interaction features
        attendance_submission_ratio = input_data['attendance_rate'] / (input_data['assignment_submission_rate'] + 1)
        attendance_lms_interaction = input_data['attendance_rate'] * input_data['lms_login_frequency']
        
        input_data['engagement_score'] = round(engagement_score, 2)
        input_data['attendance_submission_ratio'] = round(attendance_submission_ratio, 2)
        input_data['attendance_lms_interaction'] = round(attendance_lms_interaction, 2)
        
        df_input = pd.DataFrame([input_data])
        df_features = df_input[self.features]

        # Prediction
        proba_all = self.model.predict_proba(df_features)[0]
        probability = float(proba_all[1])
        risk = 1 if probability > 0.35 else 0

        # SHAP Explainability
        shap_values_raw = self.explainer.shap_values(df_features)
        
        # Extract correctly based on LightGBM output (handling list vs single array)
        if isinstance(shap_values_raw, list):
            s_vals = shap_values_raw[1][0] if len(shap_values_raw) > 1 else shap_values_raw[0][0]
        else:
            s_vals = shap_values_raw[0, :, 1] if len(shap_values_raw.shape) == 3 else shap_values_raw[0]

        # Use new intelligence modules
        explanation = get_top_factors(s_vals, self.features)
        interventions = recommend_intervention(input_data)

        return {
            "risk": risk,
            "probability": round(probability, 2),
            "explanation": explanation,
            "intervention": interventions
        }

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/dropout_model.joblib")
dropout_service = DropoutService(MODEL_PATH)
