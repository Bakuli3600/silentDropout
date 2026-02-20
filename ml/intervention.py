from typing import List, Dict, Any

def recommend_intervention(data: Dict[str, Any]) -> List[str]:
    """
    Suggests specific actions based on the student's engagement data.
    """
    actions = []

    # Attendance Rule: Significant drops in physical/virtual presence
    if data["attendance_rate"] < 55:
        actions.append("Schedule a personalized 1:1 counseling session with a counselor.")
        actions.append("Notify the student's guardian regarding their attendance pattern.")

    # Digital Engagement Rule: Low LMS activity
    if data["lms_login_frequency"] < 2.5:
        actions.append("Assign interactive, bite-sized LMS modules to encourage re-engagement.")
        actions.append("Perform a digital audit to ensure no technical barriers are preventing logins.")

    # Academic Rule: Low assignment submission
    if data["assignment_submission_rate"] < 40:
        actions.append("Provide targeted assignment support and clarification sessions.")
        actions.append("Evaluate if the workload is appropriate and adjust accordingly.")

    # Borderline Case Rule (Passive Students)
    if 60 < data["attendance_rate"] < 80 and data["assignment_submission_rate"] < 50:
        actions.append("Encourage the student to actively participate in class discussions.")
        actions.append("Assign a student mentor for peer-to-peer academic support.")

    # Always provide a baseline if no critical triggers
    if not actions:
        actions.append("Continue regular monitoring of student engagement levels.")
        actions.append("Share periodic feedback on progress to maintain positive momentum.")

    return actions
