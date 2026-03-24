import joblib
import pandas as pd
import os

# Define paths
MODEL_PATH = 'backend/models/dropout_model.joblib'

def calculate_engineered_features(row):
    normalized_lms = (row['lms_login_frequency'] / 14) * 100
    engagement_score = (
        row['attendance_rate'] * 0.3 + 
        row['assignment_submission_rate'] * 0.4 + 
        normalized_lms * 0.3
    )
    ratio = row['attendance_rate'] / (row['assignment_submission_rate'] + 1)
    lms_inter = row['attendance_rate'] * row['lms_login_frequency']
    return engagement_score, ratio, lms_inter

def display_table():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']

    edge_cases = [
        {"name": "Perfect Student", "attendance_rate": 100, "assignment_submission_rate": 100, "lms_login_frequency": 14, "avg_session_time": 120, "grades": 95, "expected": 0},
        {"name": "Severe Risk", "attendance_rate": 20, "assignment_submission_rate": 10, "lms_login_frequency": 0.5, "avg_session_time": 5, "grades": 30, "expected": 1},
        {"name": "Silent Risk (Low Att)", "attendance_rate": 30, "assignment_submission_rate": 90, "lms_login_frequency": 10, "avg_session_time": 60, "grades": 80, "expected": 1},
        {"name": "Low Submissions", "attendance_rate": 90, "assignment_submission_rate": 20, "lms_login_frequency": 12, "avg_session_time": 45, "grades": 85, "expected": 0},
        {"name": "Inactive/High Grade", "attendance_rate": 40, "assignment_submission_rate": 30, "lms_login_frequency": 2, "avg_session_time": 10, "grades": 85, "expected": 1},
        {"name": "Zero LMS Activity", "attendance_rate": 90, "assignment_submission_rate": 80, "lms_login_frequency": 0, "avg_session_time": 0, "grades": 80, "expected": 1},
        {"name": "High Att/0 Sub", "attendance_rate": 100, "assignment_submission_rate": 0, "lms_login_frequency": 5, "avg_session_time": 30, "grades": 40, "expected": 1}
    ]

    results = []
    for case in edge_cases:
        eng, ratio, lms_inter = calculate_engineered_features(case)
        input_df = pd.DataFrame([{
            'attendance_rate': case['attendance_rate'],
            'assignment_submission_rate': case['assignment_submission_rate'],
            'lms_login_frequency': case['lms_login_frequency'],
            'avg_session_time': case['avg_session_time'],
            'grades': case['grades'],
            'engagement_score': eng,
            'attendance_submission_ratio': ratio,
            'attendance_lms_interaction': lms_inter
        }])
        
        prob = float(model.predict_proba(input_df[features])[0][1])
        pred = 1 if prob > 0.35 else 0
        
        results.append({
            "Test Case": case['name'],
            "Att %": int(case['attendance_rate']),
            "Sub %": int(case['assignment_submission_rate']),
            "Exp": "RISK" if case['expected'] == 1 else "SAFE",
            "Pred": "RISK" if pred == 1 else "SAFE",
            "Prob": f"{prob:.2f}",
            "Status": "✅ PASS" if pred == case['expected'] else "❌ FAIL"
        })

    df_results = pd.DataFrame(results)
    print("\n" + "="*85)
    print(" SILENT DROPOUT DETECTION SYSTEM: SAMPLE TEST VERIFICATION TABLE")
    print("="*85)
    print(df_results.to_string(index=False))
    print("="*85 + "\n")

if __name__ == "__main__":
    display_table()
