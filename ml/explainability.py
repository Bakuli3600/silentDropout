from typing import List, Dict, Any
import numpy as np

def get_top_factors(shap_values: np.ndarray, feature_names: List[str]) -> List[str]:
    """
    Converts raw SHAP values into a sorted list of human-readable factor descriptions.
    """
    # Create a list of (feature, shap_value) pairs
    importance = []
    for name, val in zip(feature_names, shap_values):
        importance.append({"name": name.replace('_', ' ').title(), "val": val})
    
    # Sort by absolute impact (highest first)
    importance.sort(key=lambda x: abs(x["val"]), reverse=True)
    
    # Take top 3 and format for human readability
    explanations = []
    for item in importance[:3]:
        direction = "increased" if item["val"] > 0 else "decreased"
        explanations.append(f"{item['name']} {direction} dropout risk by {abs(item['val']):.2f}")
    
    return explanations
