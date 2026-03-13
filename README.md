# 🦜 Project Canary: Predict. Prevent. Prosper.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

**Project Canary** is an AI-powered Early Warning System (EWS) that moves beyond binary alarms. By combining Machine Learning with industry-standard Biological Roadmaps, Canary predicts mortality risk **24 hours before it happens**, turning poultry barns from reactive to proactive.

---

## 🚀 Key Features

### 1. 🛡️ Clinical Early Warning
Predicts tomorrow's health status today by identifying the environmental and behavioral precursors of stress.

### 2. 🧬 4-Stage Precision Roadmap
Snaps barn settings to industry-standard biological plateaus (Starter, Grower, Finisher, Pre-Harvest) at the touch of a button.

### 3. 🧠 Strategic DICE Engine
The **Dynamic Intervention & Correction Engine** simulates the exact environment needed to pull a flock back from the danger zone.

### 4. 🧪 Intervention Lab
A high-fidelity "What If" simulator for real-time risk re-calculation.

---

## 📊 The Machine Learning Approach

### 🛡️ Safety-First Modeling
We prioritize **Recall (Sensitivity)** over raw Accuracy. In a high-stakes agricultural environment, the cost of a missed risk event is catastrophic, while the cost of a false alarm is negligible. 
- **Optimized Threshold**: Recalibrated to **0.35** to catch 50-100% more stress events than standard models.
- **Class Imbalance**: Implemented `class_weight='balanced'` and custom scoring to handle the rarity of health crises.

### 🧬 Biological Feature Engineering
- **Cumulative Stress Tracking**: Rolling 3-day averages capture the "Physical Toll" of sustained heat, which is a stronger predictor than instantaneous spikes.
- **Physiological Interactions**: Explicit `Age * Temperature` interaction terms to model the decreasing heat tolerance of larger broilers.
- **Explainability (SHAP)**: We use Shapley values to prove the model's logic aligns with veterinary science, ensuring transparency and trust with farm operators.

---

## 🛠️ Technical Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (High-performance Data UI)
- **ML Engine**: Scikit-Learn (Random Forest, Gradient Boosting)
- **Interpretability**: [SHAP](https://github.com/slundberg/shap) (Feature Importance)
- **Visualization**: Plotly (Interactive Gauges & Heatmaps)
- **LLM**: Google Gemini AI (Agentic Consultation)
- **Environment**: Python 3.10, Virtualenv

---

## 📂 Project Structure

```bash
├── app.py                     # Primary Streamlit Command Center
├── model_development.ipynb    # Technical Research & Executive Summary
├── broiler_health_dataset.csv # Historical Telemetry Data
├── .venv/                     # Python Virtual Environment
└── requirements.txt           # Project Dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/project-canary.git
   cd project-canary
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # MacOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Secret Key** (For Gemini AI):
   Create `.streamlit/secrets.toml` and add:
   ```toml
   GEMINI_API_KEY = "your_google_api_key_here"
   ```

5. **Run the Command Center**:
   ```bash
   streamlit run app.py
   ```

---

## 🏆 Strategic Impact

- **Proactive Management**: Shifts the reaction window from 0 hours to 24 hours.
- **Economic Value**: Directly reduces Value-at-Risk (VAR) by preventing preventable mortality.
- **Animal Welfare**: Optimizes environmental comfort tailored to the specific biological age of the bird.

---

## 🤝 Contributing
Contributions are welcome! Whether it's adding a new biological sensor or improving the DICE optimization algorithm, feel free to open a PR.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Developed with ❤️ by **Jourdango**
