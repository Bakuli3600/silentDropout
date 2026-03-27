import os
import pandas as pd
import numpy as np
from faker import Faker
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Initialize Faker with Indian locale
fake = Faker('en_IN')

# Massive 5,000 student dataset
num_students = 5000

def generate_student_data(n):
    """
    Advanced synthetic data generation for 'Silent Dropout' detection.
    Specifically fixes bias where high attendance masks other risks.
    """
    data = []
    institutions = ["Jadavpur University", "University of Calcutta", "St. Xavier's College", "Presidency University"]
    
    for _ in range(n):
        # 1. Base Feature Generation
        attendance_rate = np.random.normal(70, 22)
        assignment_submission_rate = np.random.normal(65, 28)
        lms_login_frequency = np.random.normal(4, 3.5)
        avg_session_time = np.random.normal(35, 25)
        grades = np.random.normal(60, 20)
        
        # Clip to realistic ranges
        attendance_rate = np.clip(attendance_rate, 0, 100)
        assignment_submission_rate = np.clip(assignment_submission_rate, 0, 100)
        lms_login_frequency = np.clip(lms_login_frequency, 0, 14)
        avg_session_time = np.clip(avg_session_time, 2, 180)
        grades = np.clip(grades, 0, 100)
        
        # 2. STRONGER FEATURE ENGINEERING
        # Weighted Engagement Score (Higher weight on Submissions)
        normalized_lms = (lms_login_frequency / 14) * 100
        engagement_score = (
            attendance_rate * 0.3 + 
            assignment_submission_rate * 0.4 + 
            normalized_lms * 0.3
        )
        
        # New Interaction: Attendance / Submission Ratio (Detects "Present but not working")
        # Adding 1 to avoid division by zero
        attendance_submission_ratio = attendance_rate / (assignment_submission_rate + 1)
        
        # Interaction: Attendance x LMS
        attendance_lms_interaction = attendance_rate * lms_login_frequency
        
        # 3. RESEARCH-DRIVEN DROPOUT RISK LOGIC (FIXING BIASES)
        prob = 0.0
        
        # FIX: Zero/Very Low Submission = HIGH RISK (Regardless of attendance)
        if assignment_submission_rate < 15:
            prob += 0.8
            
        # FIX: High Attendance but Low Submission (The "Passive" Student)
        if attendance_rate > 75 and assignment_submission_rate < 30:
            prob += 0.7
            
        # FIX: Borderline Zone Risk
        if 35 < attendance_rate < 55 and lms_login_frequency < 2.5:
            prob += 0.6
            
        # High Grades but Low Engagement (The "Silent" Dropout)
        if grades > 75 and engagement_score < 40:
            prob += 0.6
            
        # Chronic Low LMS Activity
        if lms_login_frequency < 1.2:
            prob += 0.5
            
        # Poor Academic Performance
        if grades < 40:
            prob += 0.3
            
        # Add noise
        prob += np.random.normal(0, 0.05)
        
        # Final Classification
        # We use a combined logic: high prob OR critical engagement failure
        dropout_risk = 1 if prob > 0.45 or (engagement_score < 25 and grades < 55) else 0
        
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
            "engagement_score": round(engagement_score, 2),
            "attendance_submission_ratio": round(attendance_submission_ratio, 2),
            "attendance_lms_interaction": round(attendance_lms_interaction, 2),
            "dropout_risk": dropout_risk
        })
        
    return pd.DataFrame(data)

df = generate_student_data(num_students)
os.makedirs('data', exist_ok=True)
df.to_csv('data/synthetic_students.csv', index=False)

print(f"✅ Generated {len(df)} records with Research-Fix logic.")
print(f"📊 New Dropout Rate: {round(df['dropout_risk'].mean() * 100, 2)}%")
print(f"🔍 Passive Risk Cases (High Att + Low Sub): {len(df[(df['attendance_rate'] > 75) & (df['assignment_submission_rate'] < 30)])}")
