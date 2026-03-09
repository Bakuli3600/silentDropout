import React, { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { 
  AlertTriangle, CheckCircle, Info, TrendingUp, Users, Target, Search,
  ArrowRight, Shield, Activity, Lightbulb, UserCheck, MessageSquare, Briefcase
} from 'lucide-react';
import './App.css';

interface Prediction {
  risk: number;
  probability: number;
  explanation: string[];
  intervention: string[];
}

interface FactorData {
  name: string;
  value: number;
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
  const [factorData, setFactorData] = useState<FactorData[]>([]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: parseFloat(value) });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`http://${window.location.hostname}:8000/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('Network error. Is the backend running?');

      const result = await response.json();
      setPrediction(result);
      
      // Parse SHAP explanations for the chart
      const parsedData = result.explanation.map((item: string) => {
        const parts = item.split(' ');
        const name = parts.slice(0, 2).join(' ');
        const value = parseFloat(parts[parts.length - 1]);
        const direction = item.includes('increased') ? 1 : -1;
        return { name, value: value * direction };
      });
      setFactorData(parsedData);
      
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (prob: number) => {
    if (prob < 0.35) return '#10b981'; // Success Green
    if (prob < 0.6) return '#f59e0b'; // Accent Amber
    return '#ef4444'; // Danger Red
  };

  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-brand">
          <Shield size={24} />
          <span>SilentDropout AI</span>
        </div>
        <div className="nav-links">
          <a href="#dashboard" className="nav-link">Dashboard</a>
          <a href="#methodology" className="nav-link">Methodology</a>
          <a href="#about" className="nav-link">SDG Goal 4</a>
        </div>
        <div className="nav-user">
          <UserCheck size={20} className="text-muted" />
        </div>
      </nav>

      <section className="hero">
        <div className="hero-content">
          <h1>Kolkata Student Success Platform</h1>
          <p>Advanced predictive analytics for early detection of student disengagement. Empowering educators with AI-driven insights to achieve Quality Education (SDG Goal 4).</p>
          <div className="hero-badges">
            <div className="badge">LightGBM Optimized</div>
            <div className="badge">SHAP Explainability</div>
            <div className="badge">96% Accuracy</div>
          </div>
        </div>
      </section>

      <main className="main-container">
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon"><Users size={24} /></div>
            <div className="stat-info">
              <h3>Pilot Students</h3>
              <p>5,000+</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><Target size={24} /></div>
            <div className="stat-info">
              <h3>Model Precision</h3>
              <p>96.0%</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><TrendingUp size={24} /></div>
            <div className="stat-info">
              <h3>Risk Prevention</h3>
              <p>Early Stage</p>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon"><Briefcase size={24} /></div>
            <div className="stat-info">
              <h3>Institutions</h3>
              <p>West Bengal</p>
            </div>
          </div>
        </div>

        <div className="dashboard-grid">
          {/* Form Side */}
          <div className="analysis-form">
            <h2><Search size={20} /> Student Analysis</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-section">
                <div className="form-label">
                  <span>Attendance Rate</span>
                  <span>{formData.attendance_rate}%</span>
                </div>
                <input type="range" name="attendance_rate" value={formData.attendance_rate} onChange={handleInputChange} min="0" max="100" step="1" />
              </div>

              <div className="form-section">
                <div className="form-label">
                  <span>Assignment Submission</span>
                  <span>{formData.assignment_submission_rate}%</span>
                </div>
                <input type="range" name="assignment_submission_rate" value={formData.assignment_submission_rate} onChange={handleInputChange} min="0" max="100" step="1" />
              </div>

              <div className="form-section">
                <div className="form-label">
                  <span>LMS Logins / Week</span>
                  <span>{formData.lms_login_frequency}x</span>
                </div>
                <input type="range" name="lms_login_frequency" value={formData.lms_login_frequency} onChange={handleInputChange} min="0" max="14" step="0.5" />
              </div>

              <div className="form-section">
                <div className="form-label">
                  <span>Avg Session Time (Min)</span>
                  <span>{formData.avg_session_time}m</span>
                </div>
                <input type="range" name="avg_session_time" value={formData.avg_session_time} onChange={handleInputChange} min="0" max="120" step="5" />
              </div>

              <div className="form-section">
                <div className="form-label">
                  <span>Current Grades</span>
                  <span>{formData.grades}%</span>
                </div>
                <input type="range" name="grades" value={formData.grades} onChange={handleInputChange} min="0" max="100" step="1" />
              </div>

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? 'Processing...' : 'Run Analysis'}
                {!loading && <ArrowRight size={18} />}
              </button>
            </form>
          </div>

          {/* Results Side */}
          <div className="results-panel">
            {error && <div className="error-card animate-in"><AlertTriangle size={20} /> {error}</div>}
            
            {prediction ? (
              <div className="animate-in">
                <div className="risk-summary" style={{ borderLeft: `8px solid ${getRiskColor(prediction.probability)}` }}>
                  <div className="risk-level">
                    <div className="risk-info">
                      <h2>Calculated Risk Profile</h2>
                      <p style={{ color: getRiskColor(prediction.probability) }}>
                        {prediction.risk === 1 ? 'HIGH ALERT' : 'SAFE / MONITOR'}
                      </p>
                    </div>
                  </div>
                  <div className="prob-meter">
                    <div className="prob-value">{Math.round(prediction.probability * 100)}%</div>
                    <div className="text-muted" style={{fontSize: '0.8rem'}}>Dropout Probability</div>
                  </div>
                </div>

                <div className="insights-grid" style={{marginTop: '1.5rem'}}>
                  {/* SHAP Chart */}
                  <div className="insight-card">
                    <h3><Activity size={18} /> Feature Impact (XAI)</h3>
                    <div style={{ width: '100%', height: 250 }}>
                      <ResponsiveContainer>
                        <BarChart data={factorData} layout="vertical" margin={{ left: -20, right: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                          <XAxis type="number" hide />
                          <YAxis dataKey="name" type="category" width={100} style={{ fontSize: '0.7rem' }} />
                          <Tooltip 
                            formatter={(value: any) => [value ? Math.abs(Number(value)).toFixed(2) : "0.00", "Impact"]} 
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                          />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {factorData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Interventions */}
                  <div className="insight-card">
                    <h3><Lightbulb size={18} /> Recommended Interventions</h3>
                    <ul className="intervention-list">
                      {prediction.intervention.map((item, idx) => (
                        <li key={idx} className="intervention-item">
                          <CheckCircle size={16} className="text-accent" style={{flexShrink: 0, marginTop: '2px'}} />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="insight-card" style={{marginTop: '1.5rem'}}>
                  <h3><Info size={18} /> Pedagocial Analysis</h3>
                  <div className="xai-list">
                    {prediction.explanation.map((item, idx) => (
                      <div key={idx} className="xai-item">
                        <MessageSquare size={16} className="text-primary" style={{flexShrink: 0, marginTop: '2px'}} />
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : !loading && !error && (
              <div className="placeholder">
                <Search size={48} />
                <h3>Ready for Analysis</h3>
                <p>Adjust student parameters and click "Run Analysis" to generate an AI-powered risk assessment and pedagogical roadmap.</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer>
        <p>© 2026 Silent Dropout Detection Project. Built for Kolkata Higher Education Institutions.</p>
        <p style={{marginTop: '0.5rem', opacity: 0.6}}>Sustainable Development Goal 4: Ensure inclusive and equitable quality education.</p>
      </footer>
    </div>
  );
}

export default App;
