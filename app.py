import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="Salary Prediction Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium look (sleek styling, gradients, shadow, font styles)
custom_css = """
<style>
    /* Main App Background and Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Panel Custom Layout */
    .main {
        background-color: #0e1117;
    }
    
    /* Header Gradient styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.35);
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #ffffff, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Glassmorphism Results Card */
    .result-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .result-value-high {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .result-value-low {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #ff9966, #ff5e62);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #888888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Styled Button container styling */
    div.stButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.5) !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Helper function to load model safely
@st.cache_resource
def load_prediction_model():
    model_path = 'best_model.joblib'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# Helper function to load dataset for summary statistics comparison
@st.cache_data
def load_comparison_data():
    dataset_path = 'employee_salary_data.csv'
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return None

# Load resources
pipeline = load_prediction_model()
df_data = load_comparison_data()

# Header Display
st.markdown(
    """
    <div class="header-container">
        <div class="header-title">💼 Employee Salary Prediction Engine</div>
        <div class="header-subtitle">Leverage advanced machine learning to predict employee salary categories and benchmark workforce metrics</div>
    </div>
    """, 
    unsafe_allow_html=True
)

if pipeline is None:
    st.error("⚠️ Model file `best_model.joblib` not found. Please run the `train_model.py` training script first to serialize the model.")
else:
    # Set up columns: Sidebar is handled by Streamlit, but we can also organize main panel into columns
    col_input, col_display = st.columns([1, 1])
    
    # ------------------ SIDEBAR FOR INPUT FEATURES ------------------
    st.sidebar.markdown("### 🛠️ Input Employee Attributes")
    st.sidebar.markdown("Provide employee details below to predict their salary classification.")
    
    # Define options based on generated dataset categories
    workclass_options = ['Private', 'Self-Emp', 'State-Gov', 'Federal-Gov', 'Local-Gov', 'Without-Pay']
    education_options = ['High School', 'Associate', 'Bachelors', 'Masters', 'Doctorate']
    occupation_options = ['Admin', 'Sales', 'Tech-Support', 'Exec-Managerial', 'Prof-Specialty', 
                          'Craft-Repair', 'Other-Service', 'Farming-Fishing']
    
    # Inputs
    age = st.sidebar.slider("Age (Years)", min_value=18, max_value=65, value=35, step=1)
    
    # Max experience is constrained realistically by age (cannot exceed age - 18)
    max_experience = max(0, age - 18)
    experience = st.sidebar.slider("Work Experience (Years)", min_value=0, max_value=max_experience, value=min(10, max_experience), step=1)
        
    hours_per_week = st.sidebar.slider("Hours Worked Per Week", min_value=10, max_value=80, value=40, step=1)
    
    workclass = st.sidebar.selectbox("Work-Class", options=workclass_options, index=0)
    education = st.sidebar.selectbox("Education Level", options=education_options, index=2) # default Bachelors
    occupation = st.sidebar.selectbox("Occupation Category", options=occupation_options, index=3) # default Exec-Managerial
    
    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("Predict Salary Category")
    
    # Create input dictionary for prediction (matching features expected by model)
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'education': education,
        'experience': experience,
        'occupation': occupation,
        'hours_per_week': hours_per_week,
        'experience_ratio': experience / (age + 1e-5)
    }])
    
    # ------------------ MAIN DISPLAY AREA ------------------
    with col_input:
        st.markdown("### 📋 Current Input Summary")
        
        # Display inputs in a clean, structured table
        summary_df = pd.DataFrame({
            "Attribute": ["Age", "Experience", "Hours/Week", "Workclass", "Education", "Occupation"],
            "Value": [f"{age} years", f"{experience} years", f"{hours_per_week} hrs/week", workclass, education, occupation]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.info("ℹ️ Fill out all features in the sidebar and click the **Predict Salary Category** button to see the model predictions and comparisons.")

    with col_display:
        st.markdown("### 🔮 Prediction Analysis")
        
        # Prediction logic triggered on button click OR default render
        if predict_btn or 'prediction_done' in st.session_state or True: # Trigger default to make it look alive
            st.session_state['prediction_done'] = True
            
            try:
                # Predict
                pred_class_encoded = pipeline.predict(input_data)[0]
                pred_proba = pipeline.predict_proba(input_data)[0]
                
                class_labels = {0: '<=25K', 1: '25K-50K', 2: '50K-75K', 3: '75K-100K', 4: '>100K'}
                pred_class = class_labels[pred_class_encoded]
                confidence = pred_proba[pred_class_encoded]
                
                # Dynamic gradients for each segment card
                color_gradients = {
                    0: "linear-gradient(45deg, #ff9966, #ff5e62)", # Orange-Red for <=25K
                    1: "linear-gradient(45deg, #f12711, #f5af19)", # Red-Yellow for 25-50K
                    2: "linear-gradient(45deg, #1e3c72, #2a5298)", # Blue for 50-75K
                    3: "linear-gradient(45deg, #360033, #0b8793)", # Purple-Teal for 75-100K
                    4: "linear-gradient(45deg, #11998e, #38ef7d)"  # Emerald-Green for >100K
                }
                card_style = f"font-size: 2.8rem; font-weight: 800; background: {color_gradients[pred_class_encoded]}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0;"
                emoji = "🚀" if pred_class_encoded >= 3 else "💼"
                
                # Multi-segment meter
                classes = ['<=25K', '25K-50K', '50K-75K', '75K-100K', '>100K']
                meter_html = "<div style='display: flex; gap: 8px; margin-top: 15px; margin-bottom: 15px;'>"
                for idx, cls in enumerate(classes):
                    is_active = (idx == pred_class_encoded)
                    bg_color = color_gradients[idx] if is_active else "#222222"
                    text_color = "white" if is_active else "#888888"
                    border = "1px solid rgba(255,255,255,0.2)" if is_active else "1px solid #444444"
                    font_weight = "bold" if is_active else "normal"
                    shadow = "box-shadow: 0 4px 10px rgba(56,239,125,0.3);" if is_active else ""
                    
                    meter_html += f"""
                    <div style='
                        flex: 1; 
                        text-align: center; 
                        padding: 8px; 
                        background: {bg_color}; 
                        color: {text_color}; 
                        border: {border}; 
                        border-radius: 6px; 
                        font-weight: {font_weight}; 
                        font-size: 0.8rem;
                        {shadow}
                    '>
                        {cls}
                    </div>
                    """
                meter_html += "</div>"
                
                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="metric-label">Predicted Salary Category</span>
                        <div style="{card_style}">{emoji} {pred_class}</div>
                        {meter_html}
                        <span class="metric-label">Model Confidence</span>
                        <div style="font-size: 1.5rem; font-weight: 600; color: white; margin-top:5px;">
                            {confidence*100:.2f}%
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Progress bar for confidence
                st.progress(float(confidence))
            except Exception as e:
                st.error(f"⚠️ An error occurred during prediction: {e}")

    # ------------------ OPTIONAL: COMPARISON SECTION ------------------
    if df_data is not None and 'pred_class' in locals():
        st.markdown("---")
        st.markdown("### 📊 Dataset Benchmark Comparisons")
        
        col_chart, col_stats = st.columns([2, 1])
        
        with col_stats:
            st.markdown("#### Workforce Average Benchmarks")
            # Calculate benchmarks from original dataset
            avg_age = df_data['age'].mean()
            avg_exp = df_data['experience'].mean()
            avg_hours = df_data['hours_per_week'].mean()
            
            # Render benchmarking metrics compared to user inputs
            st.metric("Age Benchmark", f"{avg_age:.1f} yrs", f"{age - avg_age:+.1f} yrs compared to avg")
            st.metric("Experience Benchmark", f"{avg_exp:.1f} yrs", f"{experience - avg_exp:+.1f} yrs compared to avg")
            st.metric("Weekly Hours Benchmark", f"{avg_hours:.1f} hrs", f"{hours_per_week - avg_hours:+.1f} hrs compared to avg")
            
        with col_chart:
            st.markdown("#### Feature Comparison: User vs Database Classes")
            
            # Calculate averages for predicted class and overall workforce
            db_pred_means = df_data[df_data['salary'] == pred_class][['age', 'experience', 'hours_per_week']].mean().values
            db_overall_means = df_data[['age', 'experience', 'hours_per_week']].mean().values
            
            # Prepare plotting data
            categories = ['Age', 'Experience', 'Hours/Week']
            user_vals = [age, experience, hours_per_week]
            
            # Plot
            x = np.arange(len(categories))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0e1117')
            ax.set_facecolor('#0e1117')
            
            # Bar structures
            rects1 = ax.bar(x - width, user_vals, width, label='User Input', color='#38ef7d')
            rects2 = ax.bar(x, db_pred_means, width, label=f'Bracket Avg ({pred_class})', color='#ff9966', alpha=0.8)
            rects3 = ax.bar(x + width, db_overall_means, width, label='Workforce Avg (Overall)', color='#2a5298', alpha=0.8)
            
            # Formatting
            ax.set_title('Benchmarking Numerical Features', color='white', fontsize=12, pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.2)
            
            # Legend
            legend = ax.legend(loc='upper right', frameon=True)
            frame = legend.get_frame()
            frame.set_facecolor('#0e1117')
            frame.set_edgecolor('#333333')
            for text in legend.get_texts():
                text.set_color('white')
                
            plt.tight_layout()
            st.pyplot(fig)
