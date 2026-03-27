import os
import pandas as pd
import numpy as np
from faker import Faker
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Initialize Faker with Indian locale for Kolkata-based names
fake = Faker('en_IN')

# Number of students to generate
num_students = 1200

def generate_student_data(n):
    """
    Generates synthetic student data for 'Silent Dropout' detection.
    Target: SDG Goal 4 (Quality Education) - Indian Context (Kolkata).
    """
    data = []
    
    # Schools/Institutions in Kolkata area
    institutions = ["Jadavpur University", "University of Calcutta", "St. Xavier's College", "Presidency University"]
    
    for _ in range(n):
        # Generate core features
        # Higher values typically mean better engagement
        attendance_rate = np.random.normal(85, 15)  # Mean 85, SD 15
        assignment_submission_rate = np.random.normal(80, 20)
        lms_login_frequency = np.random.normal(5, 2)  # Logins per week
        avg_session_time = np.random.normal(45, 20)   # Minutes per session
        grades = np.random.normal(70, 15)             # Percentage 0-100
        
        # Clip values to realistic ranges
        attendance_rate = np.clip(attendance_rate, 0, 100)
        assignment_submission_rate = np.clip(assignment_submission_rate, 0, 100)
        lms_login_frequency = np.clip(lms_login_frequency, 0, 14) # Max 2 logins/day
        avg_session_time = np.clip(avg_session_time, 2, 180)      # 2 min to 3 hours
        grades = np.clip(grades, 0, 100)
        
        # Calculate Dropout Risk Score (0-1) based on engagement
        # Normalized inverse weightings
        attendance_norm = (100 - attendance_rate) / 100
        assignment_norm = (100 - assignment_submission_rate) / 100
        lms_norm = (14 - lms_login_frequency) / 14
        session_norm = (180 - avg_session_time) / 180
        grade_norm = (100 - grades) / 100

        # Risk score calculation (0 to 1)
        risk_score = (
            0.4 * attendance_norm +
            0.3 * assignment_norm +
            0.15 * lms_norm +
            0.1 * session_norm +
            0.05 * grade_norm
        )
        
        # Add some randomness/noise to the risk score
        risk_score += np.random.normal(0, 0.1)
        
        # Convert risk score to binary classification (0: No, 1: Yes)
        # Threshold: > 0.55 risk score is a likely dropout
        dropout_risk = 1 if risk_score > 0.55 else 0
        
        # Student profile info
        student_id = f"KOL-{2026000 + _}"
        student_name = f"{fake.first_name()} {fake.last_name()}"
        institution = random.choice(institutions)
        
        data.append({
            "student_id": student_id,
            "name": student_name,
            "institution": institution,
            "attendance_rate": round(attendance_rate, 2),
            "assignment_submission_rate": round(assignment_submission_rate, 2),
            "lms_login_frequency": round(lms_login_frequency, 1),
            "avg_session_time": round(avg_session_time, 1),
            "grades": round(grades, 2),
            "dropout_risk": dropout_risk
        })
        
    return pd.DataFrame(data)

# Generate the data
df = generate_student_data(num_students)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = 'data/synthetic_students.csv'
df.to_csv(output_path, index=False)

print(f"✅ Successfully generated {num_students} student records.")
print(f"📂 Saved to: {output_path}")

# Display breakdown of dropouts
dropout_counts = df['dropout_risk'].value_counts()
print(f"\nDropout Breakdown:")
print(f"Non-Dropout: {dropout_counts.get(0, 0)}")
print(f"Dropout: {dropout_counts.get(1, 0)}")
print(f"Dropout Rate: {round(dropout_counts.get(1, 0)/len(df)*100, 2)}%")
