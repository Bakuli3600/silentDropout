import os
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List

class DropoutService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.explainer = None
        self._load_model()

    def _load_model(self):
        """
        Loads the trained model and initializes SHAP.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.features = model_data['features']
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference with Feature Engineering + Lower Decision Threshold (0.4)
        """
        # --- FIX 2: Dynamic Feature Engineering ---
        # Calculate engagement_score based on normalized LMS login frequency
        normalized_lms = (input_data['lms_login_frequency'] / 14) * 100
        engagement_score = (
            input_data['attendance_rate'] * 0.4 +
            input_data['assignment_submission_rate'] * 0.3 +
            normalized_lms * 0.3
        )
        
        # Add engagement_score to input data
        input_data['engagement_score'] = round(engagement_score, 2)
        
        # --- Core Prediction Logic ---
        df_input = pd.DataFrame([input_data])
        df_features = df_input[self.features]

        # Get probability
        proba_all = self.model.predict_proba(df_features)[0]
        probability = float(proba_all[1])

        # --- FIX 4: Lower Decision Threshold (0.4) for higher sensitivity ---
        risk = 1 if probability > 0.4 else 0

        # --- SHAP Explanation ---
        shap_values = self.explainer.shap_values(df_features)
        
        if isinstance(shap_values, list):
            s_vals = shap_values[1][0]
        else:
            s_vals = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

        feature_impacts = []
        for feat, val in zip(self.features, s_vals):
            feature_impacts.append({
                "feature": feat.replace('_', ' ').title(),
                "impact": val
            })

        top_3 = sorted(feature_impacts, key=lambda x: abs(x['impact']), reverse=True)[:3]
        
        explanation = []
        for item in top_3:
            direction = "increased" if item['impact'] > 0 else "decreased"
            explanation.append(f"{item['feature']} {direction} risk by {abs(item['impact']):.2f}")

        return {
            "risk": risk,
            "probability": round(probability, 2),
            "explanation": explanation
        }

# Singleton instance
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/dropout_model.joblib")
dropout_service = DropoutService(MODEL_PATH)
