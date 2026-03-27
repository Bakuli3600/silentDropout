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
        Loads the trained RandomForest model and initializes SHAP explainer.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.features = model_data['features']
        
        # Initialize optimized TreeExplainer for SHAP
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference + SHAP Explanation for a single student.
        """
        df_input = pd.DataFrame([input_data])
        df_features = df_input[self.features]

        # Prediction
        risk = int(self.model.predict(df_features)[0])
        proba_all = self.model.predict_proba(df_features)[0]
        probability = float(proba_all[1])

        # SHAP Explanation (Contribution to the dropout class probability)
        # TreeExplainer returns [contrib_class_0, contrib_class_1]
        shap_values = self.explainer.shap_values(df_features)
        
        # Handle different SHAP output formats (v0.40+ vs older)
        # We want the contribution to class 1 (Dropout)
        if isinstance(shap_values, list):
            # Binary classification usually returns a list [values_for_class_0, values_for_class_1]
            s_vals = shap_values[1][0]
        else:
            # Newer SHAP might return a single array for binary
            s_vals = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

        # Map features to their SHAP values and sort by absolute impact
        feature_impacts = []
        for feat, val in zip(self.features, s_vals):
            # val > 0 means it increased dropout probability
            # val < 0 means it decreased dropout probability
            feature_impacts.append({
                "feature": feat.replace('_', ' ').title(),
                "impact": val
            })

        # Sort by absolute impact and take top 3
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
