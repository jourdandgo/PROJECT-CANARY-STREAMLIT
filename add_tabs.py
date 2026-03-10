import re

def process_app():
    with open('app.py', 'r') as f:
        lines = f.readlines()

    main_content_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "# Main Content":
            main_content_idx = i
            break

    if main_content_idx != -1:
        before = lines[:main_content_idx+1]
        after = lines[main_content_idx+1:]
        
        new_lines = before + [
            "tab_main, tab_ml = st.tabs(['🐓 Command Center', '🧠 ML Methodology'])\n",
            "\n",
            "with tab_main:\n"
        ]
        
        for line in after:
            if line.strip() == "":
                new_lines.append("\n")
            else:
                new_lines.append("    " + line)
                
        # Append Tab 2 content
        ml_tab = """
with tab_ml:
    st.markdown('<h2>Machine Learning Workflow & Methodology</h2>', unsafe_allow_html=True)
    st.markdown('<p style=\"color:#8B949E; font-size:15px; margin-top:-10px;\">A transparent look at the core intelligence powering Project Canary.</p>', unsafe_allow_html=True)
    st.write('')

    st.markdown('### 1. The Dataset & Feature Engineering')
    st.markdown('The model is trained on `broiler_health_noisy_dataset.csv`, encompassing daily telemetry for 6 zones over a 1-year period.')
    
    st.markdown('**Engineered Biological Features:**')
    st.markdown('- **Temperature-Humidity Index (THI):** A standard veterinary metric combining Heat and Humidity into a single stress identifier.')
    st.markdown('- **Water-to-Feed Ratio:** Identifies subtle behavioral shifts (e.g., chickens drinking excessively but not eating due to fever).')
    st.markdown('- **Feed Intake Delta:** Tracks day-over-day drops in feed consumption.')
    
    st.markdown('### 2. Model Pipeline & Validation')
    m_c1, m_c2, m_c3 = st.columns(3)
    with m_c1:
        st.markdown(f\"\"\"<div class='kpi-card'>
            <div class='kpi-title'>TRAIN / TEST SPLIT</div>
            <div class='kpi-value'>80/20</div>
        </div>\"\"\", unsafe_allow_html=True)
    with m_c2:
        st.markdown(f\"\"\"<div class='kpi-card'>
            <div class='kpi-title'>TEST ACCURACY</div>
            <div class='kpi-value' style='color:#00E676;'>{model_acc * 100:.1f}%</div>
        </div>\"\"\", unsafe_allow_html=True)
    with m_c3:
        st.markdown(f\"\"\"<div class='kpi-card'>
            <div class='kpi-title'>ROC AUC SCORE</div>
            <div class='kpi-value' style='color:#00E676;'>{model_auc:.3f}</div>
        </div>\"\"\", unsafe_allow_html=True)
    
    st.markdown('<br><strong>Handling Class Imbalance:</strong> In commercial poultry, mortality events are rare compared to healthy days. We use `class_weight=\"balanced\"` in our Random Forest to heavily penalize the model for False Negatives (missing a stress event).', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown('### 3. Model Interpretability (SHAP)')
    st.markdown('We use Shapley Additive Explanations (SHAP) to unpack the Random Forest decision making. This proves the logic is biologically sound, rather than just mathematically correlated.')
    
    # Generate a global SHAP summary plot representation for the training set
    st.markdown('**(See the Command Center tab for zone-specific, real-time SHAP analysis)**')
    st.image("https://shap.readthedocs.io/en/latest/_images/shap_summary.png", width=600, caption="Example Global Feature Importance (Placeholder for defense deck)")
    
    st.divider()
    
    st.markdown('### 4. The Biological DICE Engine')
    st.markdown('Statistical models predict risk based on history, but they don\\'t prescribe *how to fix it*. Project Canary uses a custom **Dynamic Intervention & Correction Engine (DICE)**.')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('**How it works:**')
        st.markdown('1. Takes the pure ML probability.')
        st.markdown('2. Adds a **Biological Stress Penalty** based on veterinary rules (e.g., 31°C heat threshold).')
        st.markdown('3. Uses an interpolative search space (0% to 100% intervention) to find the **Minimum Safe Threshold**—the exact temperature and water combination that drops the risk back into the Green Zone.')
    
    st.divider()
    
    st.markdown('### 5. Gemini 2.5 Flash Integration')
    st.markdown('The LLM acts as the translation layer between complex data arrays and the non-technical farm operator. The prompt is dynamically injected with:')
    st.markdown('- Active Zone Telemetry & Age')
    st.markdown('- Minimum Safe Thresholds calculated by DICE')
    st.markdown('- Farm-wide summary of all other zones (for cross-contextual awareness)')
"""
        new_lines.append(ml_tab)
        
        with open('app.py', 'w') as f:
            f.writelines(new_lines)
        print("Successfully rebuilt file")
    else:
        print("Could not find '# Main Content'")

process_app()
