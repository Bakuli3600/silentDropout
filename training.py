import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import shap
import os

# Set seed for reproducibility
np.random.seed(42)

def train_dropout_model(data_path, model_save_path):
    """
    Trains an optimized RandomForest model with enhanced sensitivity.
    """
    print(f"📂 Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # 1. Feature Selection (Including new Engagement Score)
    features = [
        'attendance_rate', 
        'assignment_submission_rate', 
        'lms_login_frequency', 
        'avg_session_time', 
        'grades',
        'engagement_score' # NEW FEATURE
    ]
    target = 'dropout_risk'

    X = df[features]
    y = df[target]

    # 2. Split Data (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. FIX 3: Initialize & Train Optimized RandomForest (Higher Sensitivity)
    print("🤖 Training Optimized RandomForest (n_estimators=200, depth=12, class_weight='balanced')...")
    model = RandomForestClassifier(
        n_estimators=200,      # Increased for stability
        max_depth=12,          # Deeper for complex rule capture
        class_weight='balanced', # Crucial for detecting minority dropout cases
        random_state=42,
        n_jobs=-1              # Use all CPU cores (Ryzen 7 - 8 threads)
    )
    
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n✅ Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Save Model & Features Metadata
    print(f"💾 Saving model to {model_save_path}...")
    model_data = {
        'model': model,
        'features': features,
        'target': target
    }
    joblib.dump(model_data, model_save_path)

    # 6. SHAP Initialization
    print("🔍 Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("✨ SHAP values calculated successfully.")

    return model

if __name__ == "__main__":
    DATA_PATH = 'data/synthetic_students.csv'
    MODEL_DIR = 'backend/models'
    MODEL_NAME = 'dropout_model.joblib'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    train_dropout_model(DATA_PATH, SAVE_PATH)
