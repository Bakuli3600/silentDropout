import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import joblib
import shap
import os

# Set seed for reproducibility
np.random.seed(42)

def train_dropout_model(data_path, model_save_path):
    print(f"INFO: Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # 1. Feature Selection
    features = [
        'attendance_rate', 
        'assignment_submission_rate', 
        'lms_login_frequency', 
        'avg_session_time', 
        'grades',
        'engagement_score',
        'attendance_submission_ratio',
        'attendance_lms_interaction'
    ]
    target = 'dropout_risk'

    X = df[features]
    y = df[target]

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Model Initialization
    print("INFO: Training LightGBM Classifier (n_estimators=500, depth=10, balanced)...")
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=10,
        num_leaves=63,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nSUCCESS: Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Confusion Matrix
    print("INFO: Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Safe', 'At Risk'], rotation=45)
    plt.yticks(tick_marks, ['Safe', 'At Risk'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Add text to the plot
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/confusion_matrix.png')
    print("SUCCESS: Confusion Matrix saved to assets/confusion_matrix.png")

    # 6. Save Model
    print(f"INFO: Saving model to {model_save_path}...")
    model_data = {
        'model': model,
        'features': features,
        'target': target
    }
    joblib.dump(model_data, model_save_path)

    # 7. SHAP Initialization
    print("INFO: Initializing SHAP TreeExplainer for Research analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("SUCCESS: SHAP values calculated successfully.")

    return model

if __name__ == "__main__":
    DATA_PATH = 'data/synthetic_students.csv'
    MODEL_DIR = 'backend/models'
    MODEL_NAME = 'dropout_model.joblib'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    train_dropout_model(DATA_PATH, SAVE_PATH)
