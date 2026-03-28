import React, { useState } from 'react';
import './App.css';

interface Prediction {
  risk: number;
  probability: number;
  explanation: string[];
  intervention: string[];
}

function App() {
  const [formData, setFormData] = useState({
    attendance_rate: 75.0,
    assignment_submission_rate: 70.0,
    lms_login_frequency: 4.0,
    avg_session_time: 45.0,
    grades: 65.0,
  });

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: parseFloat(value) });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Prediction failed. Is the backend running?');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (prob: number) => {
    if (prob < 0.3) return '#2ecc71'; // Green
    if (prob < 0.6) return '#f1c40f'; // Yellow
    return '#e74c3c'; // Red
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Silent Dropout Detection Dashboard</h1>
        <p>AI System for Quality Education (SDG Goal 4) - Kolkata Institution Pilot</p>
      </header>

      <div className="dashboard-container">
        <form className="input-form" onSubmit={handleSubmit}>
          <h2>Student Input</h2>
          <div className="form-group">
            <label>Attendance Rate (%)</label>
            <input type="number" name="attendance_rate" value={formData.attendance_rate} onChange={handleInputChange} min="0" max="100" step="0.1" required />
          </div>
          <div className="form-group">
            <label>Assignment Submission (%)</label>
            <input type="number" name="assignment_submission_rate" value={formData.assignment_submission_rate} onChange={handleInputChange} min="0" max="100" step="0.1" required />
          </div>
          <div className="form-group">
            <label>LMS Logins / Week</label>
            <input type="number" name="lms_login_frequency" value={formData.lms_login_frequency} onChange={handleInputChange} min="0" max="14" step="0.1" required />
          </div>
          <div className="form-group">
            <label>Avg Session (Min)</label>
            <input type="number" name="avg_session_time" value={formData.avg_session_time} onChange={handleInputChange} min="0" max="300" step="0.1" required />
          </div>
          <div className="form-group">
            <label>Current Grades (%)</label>
            <input type="number" name="grades" value={formData.grades} onChange={handleInputChange} min="0" max="100" step="0.1" required />
          </div>
          <button type="submit" disabled={loading}>{loading ? 'Calculating...' : 'Predict Dropout Risk'}</button>
        </form>

        <div className="result-display">
          {error && <div className="error-card">{error}</div>}
          
          {prediction ? (
            <div className="prediction-card fadeIn" style={{ borderLeftColor: getStatusColor(prediction.probability) }}>
              <h2>Analysis Results</h2>
              <div className="risk-score">
                Risk Level: <strong>{prediction.risk === 1 ? 'HIGH' : 'LOW'}</strong>
              </div>
              <div className="probability-container">
                <div className="prob-label">Probability: {Math.round(prediction.probability * 100)}%</div>
                <div className="prob-bar">
                  <div className="prob-fill" style={{ width: `${prediction.probability * 100}%`, backgroundColor: getStatusColor(prediction.probability) }}></div>
                </div>
              </div>

              <div className="intelligence-grid">
                <div className="intelligence-section">
                  <h3>XAI Insights (SHAP)</h3>
                  <ul className="explanation-list">
                    {prediction.explanation.map((item, idx) => (
                      <li key={idx} className="explanation-item slideIn">{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="intelligence-section">
                  <h3>Recommended Interventions</h3>
                  <ul className="intervention-list">
                    {prediction.intervention.map((item, idx) => (
                      <li key={idx} className="intervention-item slideIn">{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ) : !loading && !error && (
            <div className="placeholder-card">
              Enter student data to see AI analysis results and interventions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
