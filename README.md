# 🦜 Project Canary: Predictive AI for Precision Poultry Management

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Project Canary** is an end-to-end predictive intelligence platform designed to protect commercial poultry flocks from "Silent Stressors." By combining advanced **Random Forest Classifiers** with **Biological Stress Indices** and a **Prescriptive DICE Engine**, the platform provides farm managers with a 24-hour early warning window to prevent mortality and optimize growth.

---

## 🚀 Key Features

### 1. 🛑 Early Warning System (EWS)
Predicts tomorrow's health status today. Unlike reactive tools, Canary identifies subtle environmental and consumption patterns that precede clinical illness.

### 2. 🧪 Intervention Lab
A high-fidelity "What If" simulator. Managers can adjust environmental sliders (Temperature, Humidity, Water flow) to see real-time shifts in risk probability for a specific bird age.

### 3. 🧠 Strategic DICE Engine
The **Dynamic Intervention & Correction Engine** calculates the "Minimum Safe Threshold." It doesn't just predict risk; it prescribes the exact environment needed to stabilize the flock.

### 4. 🤖 Gemini AI Consultation
Integrating **Gemini 2.5 Flash** to provide natural language veterinatry advice. The AI has full context of barn telemetry and provides actionable, site-specific work orders.

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
