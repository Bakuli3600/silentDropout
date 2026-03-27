import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Define paths
MODEL_PATH = 'backend/models/dropout_model.joblib'
DATA_PATH = 'data/synthetic_students.csv'
RESULTS_FILE = 'test_results.txt'

def calculate_engineered_features(row):
    """Calculates all dynamic features like the service does."""
    normalized_lms = (row['lms_login_frequency'] / 14) * 100
    engagement_score = (
        row['attendance_rate'] * 0.4 +
        row['assignment_submission_rate'] * 0.3 +
        normalized_lms * 0.3
    )
    interaction = row['attendance_rate'] * row['lms_login_frequency']
    return engagement_score, interaction

def run_comprehensive_tests():
    print("🧪 Starting 50 Unique Accuracy & Inference Tests (LightGBM)...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']

    with open(RESULTS_FILE, 'w') as f:
        f.write("--- SILENT DROPOUT DETECTION SYSTEM: MASSIVE TEST RESULTS ---\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Model: LightGBM (Gradient Boosting, lower threshold=0.4)\n")
        f.write("-" * 50 + "\n\n")

    # 15 Manual Edge Cases
    edge_cases = [
        {"name": "Perfect Student", "attendance_rate": 100, "assignment_submission_rate": 100, "lms_login_frequency": 14, "avg_session_time": 120, "grades": 95, "expected": 0},
        {"name": "Severe Risk", "attendance_rate": 20, "assignment_submission_rate": 10, "lms_login_frequency": 0.5, "avg_session_time": 5, "grades": 30, "expected": 1},
        {"name": "Low Attendance Only", "attendance_rate": 30, "assignment_submission_rate": 90, "lms_login_frequency": 10, "avg_session_time": 60, "grades": 80, "expected": 1},
        {"name": "Low Submissions Only", "attendance_rate": 90, "assignment_submission_rate": 20, "lms_login_frequency": 12, "avg_session_time": 45, "grades": 85, "expected": 0},
        {"name": "Borderline Active", "attendance_rate": 60, "assignment_submission_rate": 60, "lms_login_frequency": 5, "avg_session_time": 30, "grades": 60, "expected": 0},
        {"name": "High Grades but Inactive", "attendance_rate": 40, "assignment_submission_rate": 30, "lms_login_frequency": 2, "avg_session_time": 10, "grades": 85, "expected": 1},
        {"name": "Active but Failing", "attendance_rate": 95, "assignment_submission_rate": 95, "lms_login_frequency": 12, "avg_session_time": 90, "grades": 20, "expected": 0},
        {"name": "Average Student", "attendance_rate": 80, "assignment_submission_rate": 75, "lms_login_frequency": 6, "avg_session_time": 40, "grades": 70, "expected": 0},
        {"name": "Disengaged Learner", "attendance_rate": 45, "assignment_submission_rate": 40, "lms_login_frequency": 2, "avg_session_time": 15, "grades": 50, "expected": 1},
        {"name": "Recovering Student", "attendance_rate": 70, "assignment_submission_rate": 50, "lms_login_frequency": 4, "avg_session_time": 25, "grades": 60, "expected": 0},
        {"name": "Zero LMS Activity", "attendance_rate": 90, "assignment_submission_rate": 80, "lms_login_frequency": 0, "avg_session_time": 0, "grades": 80, "expected": 1},
        {"name": "Crammer (High Sub, Low Att)", "attendance_rate": 25, "assignment_submission_rate": 100, "lms_login_frequency": 1, "avg_session_time": 10, "grades": 75, "expected": 1},
        {"name": "Just Scraping By", "attendance_rate": 50, "assignment_submission_rate": 50, "lms_login_frequency": 3, "avg_session_time": 20, "grades": 45, "expected": 1},
        {"name": "Consistent but Slow", "attendance_rate": 100, "assignment_submission_rate": 100, "lms_login_frequency": 14, "avg_session_time": 180, "grades": 55, "expected": 0},
        {"name": "High Att, 0 Submissions", "attendance_rate": 100, "assignment_submission_rate": 0, "lms_login_frequency": 5, "avg_session_time": 30, "grades": 40, "expected": 1}
    ]
    
    # 35 Samples from Dataset
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        samples = df.sample(35)
        for _, row in samples.iterrows():
            edge_cases.append({
                "name": f"Dataset Sample (ID: {row['student_id']})",
                "attendance_rate": row['attendance_rate'],
                "assignment_submission_rate": row['assignment_submission_rate'],
                "lms_login_frequency": row['lms_login_frequency'],
                "avg_session_time": row['avg_session_time'],
                "grades": row['grades'],
                "expected": int(row['dropout_risk'])
            })

    correct_predictions = 0
    total_cases = len(edge_cases)
    
    with open(RESULTS_FILE, 'a') as f:
        for i, case in enumerate(edge_cases, 1):
            eng, inter = calculate_engineered_features(case)
            
            input_df = pd.DataFrame([{
                'attendance_rate': case['attendance_rate'],
                'assignment_submission_rate': case['assignment_submission_rate'],
                'lms_login_frequency': case['lms_login_frequency'],
                'avg_session_time': case['avg_session_time'],
                'grades': case['grades'],
                'engagement_score': eng,
                'attendance_lms_interaction': inter
            }])
            
            prob = float(model.predict_proba(input_df[features])[0][1])
            pred = 1 if prob > 0.4 else 0
            
            is_correct = (pred == case['expected'])
            if is_correct:
                correct_predictions += 1
            
            status = "PASS" if is_correct else "FAIL"
            
            f.write(f"Test #{i}: {case['name']}\n")
            f.write(f"  Input: {case['attendance_rate']}% Att, {case['assignment_submission_rate']}% Sub, {case['lms_login_frequency']} Log\n")
            f.write(f"  Expected: {case['expected']}, Predicted: {pred} (Prob: {prob:.2f})\n")
            f.write(f"  Result: {status}\n")
            f.write("-" * 30 + "\n")

        accuracy = (correct_predictions / total_cases) * 100
        f.write(f"\nFINAL ACCURACY: {accuracy:.2f}% ({correct_predictions}/{total_cases})\n")
        f.write("-" * 50 + "\n")

    print(f"✅ Tests completed. LightGBM Accuracy: {accuracy:.2f}%")
    print(f"📂 Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    run_comprehensive_tests()
