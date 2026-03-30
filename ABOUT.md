# About Silent Dropout Detection System

## Overview
The Silent Dropout Detection System is a research-driven, end-to-end AI platform designed for higher education institutions in Kolkata. Its primary mission is to support **Sustainable Development Goal 4 (Quality Education)** by identifying students at risk of disengaging before they officially drop out.

## The Problem: "Silent" Dropout
Many students don't suddenly leave; they slowly disengage—maintaining physical attendance but failing to submit assignments or participate in digital learning environments (LMS). Traditional monitoring systems often miss these "silent" signals, leading to late interventions.

## Our Solution
Using a high-performance **LightGBM Gradient Boosting** model and **SHAP Explainability (XAI)**, our system analyzes multi-dimensional behavioral data to provide:
1.  **Early Risk Scoring:** Identifying disengagement weeks before academic failure.
2.  **Pedagogical Insights:** Explaining *why* a student is at risk (e.g., high attendance but low digital participation).
3.  **Actionable Interventions:** Providing data-driven roadmaps for counselors and teachers.

## Key Pilot Metrics (Kolkata Institutions)
- **Dataset:** 5,000+ unique student behavioral records.
- **Model Accuracy:** 96.00% verified across 50 diverse edge-case scenarios.
- **Explainability:** 100% of predictions include local feature importance (SHAP).
- **Intervention Engine:** Automated pedagogical recommendations based on institutional rules.

## Technical Core
- **AI Stack:** Python, LightGBM, SHAP, Scikit-Learn.
- **Backend:** FastAPI (High-performance asynchronous API).
- **Frontend:** React, TypeScript, Recharts (Modern data visualization).

---
*Developed for the Kolkata Student Success Pilot Program | SDG Goal 4: Quality Education*
