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

# Create a huge dataset
num_students = 5000

def generate_student_data(n):
    """
    Advanced synthetic data generation for 'Silent Dropout' detection.
    Includes edge cases for high grades but low engagement, and interaction features.
    """
    data = []
    institutions = ["Jadavpur University", "University of Calcutta", "St. Xavier's College", "Presidency University"]
    
    for _ in range(n):
        # 1. Base Feature Generation
        attendance_rate = np.random.normal(75, 20)
        assignment_submission_rate = np.random.normal(70, 25)
        lms_login_frequency = np.random.normal(4, 3)
        avg_session_time = np.random.normal(40, 25)
        grades = np.random.normal(65, 18)
        
        # Clip to realistic ranges
        attendance_rate = np.clip(attendance_rate, 0, 100)
        assignment_submission_rate = np.clip(assignment_submission_rate, 0, 100)
        lms_login_frequency = np.clip(lms_login_frequency, 0, 14)
        avg_session_time = np.clip(avg_session_time, 2, 180)
        grades = np.clip(grades, 0, 100)
        
        # 2. FEATURE ENGINEERING
        # Engagement Score
        normalized_lms = (lms_login_frequency / 14) * 100
        engagement_score = (
            attendance_rate * 0.4 +
            assignment_submission_rate * 0.3 +
            normalized_lms * 0.3
        )
        
        # Interaction Feature: Attendance x LMS Activity (Captures compound engagement)
        attendance_lms_interaction = attendance_rate * lms_login_frequency
        
        # 3. ADVANCED DROPOUT RISK LOGIC
        prob = 0.0
        
        # High Grades but LOW Engagement (The "Silent" Dropout)
        if grades > 75 and engagement_score < 45:
            prob += 0.65
            
        # Chronic Low LMS Activity
        if lms_login_frequency < 1.5:
            prob += 0.5
            
        # Declining Attendance
        if attendance_rate < 40:
            prob += 0.4
            
        # Poor Academic Performance
        if grades < 45:
            prob += 0.3
            
        # Compound Risk (Low interaction score)
        if attendance_lms_interaction < 100:  # e.g., 50% attendance * 2 logins
            prob += 0.2
            
        # Add noise
        prob += np.random.normal(0, 0.1)
        
        # Final Classification: (0.4 threshold)
        dropout_risk = 1 if prob > 0.45 or (engagement_score < 30 and grades < 50) else 0
        
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
            "attendance_lms_interaction": round(attendance_lms_interaction, 2),
            "dropout_risk": dropout_risk
        })
        
    return pd.DataFrame(data)

df = generate_student_data(num_students)
os.makedirs('data', exist_ok=True)
df.to_csv('data/synthetic_students.csv', index=False)

print(f"✅ Generated {len(df)} records with enhanced dropout logic and interaction features.")
print(f"📊 Dropout Rate: {round(df['dropout_risk'].mean() * 100, 2)}%")
