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

# Number of students
num_students = 1500

def generate_student_data(n):
    """
    Advanced synthetic data generation for 'Silent Dropout' detection.
    Includes edge cases for high grades but low engagement.
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
        
        # 2. FEATURE ENGINEERING: Engagement Score (Strong Behavioral Signal)
        # Normalizing logins (max 14/week) to 0-100 scale
        normalized_lms = (lms_login_frequency / 14) * 100
        engagement_score = (
            attendance_rate * 0.4 +
            assignment_submission_rate * 0.3 +
            normalized_lms * 0.3
        )
        
        # 3. ADVANCED DROPOUT RISK LOGIC (Beyond just grades)
        # Base probability calculation
        prob = 0.0
        
        # Rule 1: High Grades but LOW Engagement (The "Silent" Dropout)
        if grades > 75 and engagement_score < 45:
            prob += 0.65
            
        # Rule 2: Chronic Low LMS Activity (Strong indicator of disengagement)
        if lms_login_frequency < 1.5:
            prob += 0.5
            
        # Rule 3: Declining Attendance (Traditional indicator)
        if attendance_rate < 40:
            prob += 0.4
            
        # Rule 4: Poor Academic Performance
        if grades < 45:
            prob += 0.3
            
        # Add noise and normalize
        prob += np.random.normal(0, 0.1)
        
        # Final Classification: Higher sensitivity (0.4 threshold)
        dropout_risk = 1 if prob > 0.45 or (engagement_score < 30 and grades < 50) else 0
        
        # Profiles
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
            "dropout_risk": dropout_risk
        })
        
    return pd.DataFrame(data)

# Run generation
df = generate_student_data(num_students)
os.makedirs('data', exist_ok=True)
df.to_csv('data/synthetic_students.csv', index=False)

print(f"✅ Generated {len(df)} records with enhanced dropout logic.")
print(f"📊 Dropout Rate: {round(df['dropout_risk'].mean() * 100, 2)}%")
print(f"🔍 Silent Dropout Cases (High Grades + Low Engagement): {len(df[(df['grades'] > 75) & (df['engagement_score'] < 45)])}")
