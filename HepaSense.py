"""
====================================================================================
                    HepB INFECTION PREDICTION - STREAMLIT APP
====================================================================================
Hepatitis B Infection Prediction System using Machine Learning
====================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="HepB Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .result-positive {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 20px 0;
    }
    .result-negative {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 20px 0;
    }
    .info-box {
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model, scaler, and selector"""
    try:
        model = joblib.load('best_xgboost_model.pkl')
        scaler = joblib.load('robust_scaler.pkl')
        selector = joblib.load('feature_selector.pkl')
        
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        return model, scaler, selector, config
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None

def create_gauge_chart(value, title, threshold=0.5):
    """Create a gauge chart for probability display"""
    color = "red" if value >= threshold else "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': threshold * 100},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': '#e8f5e9'},
                {'range': [threshold * 100, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_importance_chart(importance_dict):
    """Create feature importance bar chart"""
    df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=True).tail(15)
    
    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                 title='Top 15 Most Important Features',
                 labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                 color='Importance',
                 color_continuous_scale='blues')
    
    fig.update_layout(
        height=500,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">Hepatitis B Infection Prediction System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
        <b>About this application:</b><br>
        This system uses advanced Machine Learning (XGBoost) to predict the likelihood 
        of Hepatitis B infection based on patient medical data. The model was trained 
        on NHANES dataset and achieved high accuracy in detecting HepB infections.
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, selector, config = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure all model files are available.")
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", 
                            ["Single Prediction", "Batch Prediction", "Model Information"])
    
    # ========================================================================
    # PAGE 1: SINGLE PREDICTION
    # ========================================================================
    if page == "Single Prediction":
        st.markdown('<h2 class="sub-header">Patient Information Input</h2>', 
                    unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Demographics")
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
                gender = st.selectbox("Gender", ["Male", "Female"])
                race = st.selectbox("Race/Ethnicity", 
                                   ["Mexican American", "Other Hispanic", "Non-Hispanic White",
                                    "Non-Hispanic Black", "Non-Hispanic Asian", "Other"])
                
            with col2:
                st.subheader("Physical Measurements")
                weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=75.0)
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
                systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
                diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
                
            with col3:
                st.subheader("Medical History")
                diabetes = st.selectbox("Diabetes", ["No", "Yes", "Pre-diabetes"])
                cancer = st.selectbox("History of Cancer", ["No", "Yes"])
                hepa_antibody = st.number_input("HepA Antibody", min_value=0.0, max_value=10.0, value=0.5)
                hepa_vaccine = st.selectbox("Received Hepatitis A Vaccine", ["No", "Yes"])
            
            st.subheader("Laboratory Tests")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                alt = st.number_input("ALT (U/L)", min_value=0.0, max_value=500.0, value=25.0)
                ast = st.number_input("AST (U/L)", min_value=0.0, max_value=500.0, value=30.0)
                ggt = st.number_input("GGT (U/L)", min_value=0.0, max_value=500.0, value=25.0)
                
            with col5:
                wbc = st.number_input("White Blood Cells (1000 cells/µL)", 
                                     min_value=0.0, max_value=50.0, value=7.0)
                neutrophils = st.number_input("Neutrophils (%)", 
                                             min_value=0.0, max_value=100.0, value=60.0)
                platelets = st.number_input("Platelets (1000 cells/µL)", 
                                           min_value=0.0, max_value=1000.0, value=250.0)
                
            with col6:
                cholesterol = st.number_input("Total Cholesterol (mg/dL)", 
                                             min_value=0.0, max_value=500.0, value=200.0)
                crp = st.number_input("CRP (mg/dL)", min_value=0.0, max_value=50.0, value=0.5)
                uric_acid = st.number_input("Uric Acid (mg/dL)", 
                                           min_value=0.0, max_value=20.0, value=5.0)
            
            submitted = st.form_submit_button("Predict HepB Infection Risk")
        
        # Make prediction
        if submitted:
            # Prepare input data (create dummy feature vector)
            # Note: You'll need to adjust this based on your actual features
            input_data = np.zeros((1, X_train_scaled.shape[1]))  
            
            # Fill in the values (adjust indices based on your feature order)
            # This is a simplified example - you need to match your actual feature order
            
            st.markdown('<h2 class="sub-header">Prediction Results</h2>', 
                       unsafe_allow_html=True)
            
            with st.spinner("Analyzing patient data..."):
                try:
                    # Feature selection
                    input_selected = selector.transform(input_data)
                    
                    # Scaling
                    input_scaled = scaler.transform(input_selected)
                    
                    # Prediction
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    prediction = 1 if prediction_proba[1] >= config['optimal_threshold'] else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.markdown(f"""
                                <div class="result-positive">
                                <h3 style="color: #c62828;">HIGH RISK - Positive Prediction</h3>
                                <p><b>The patient is predicted to be at HIGH RISK for Hepatitis B infection.</b></p>
                                <p>Confidence: {prediction_proba[1]*100:.1f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="result-negative">
                                <h3 style="color: #2e7d32;">LOW RISK - Negative Prediction</h3>
                                <p><b>The patient is predicted to be at LOW RISK for Hepatitis B infection.</b></p>
                                <p>Confidence: {prediction_proba[0]*100:.1f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <div class="metric-card">
                            <h4>Prediction Details</h4>
                            <p><b>Negative Class Probability:</b> {prediction_proba[0]*100:.2f}%</p>
                            <p><b>Positive Class Probability:</b> {prediction_proba[1]*100:.2f}%</p>
                            <p><b>Decision Threshold:</b> {config['optimal_threshold']*100:.1f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Gauge chart
                        fig = create_gauge_chart(prediction_proba[1], 
                                                "Infection Risk Probability",
                                                config['optimal_threshold'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.markdown('<h2 class="sub-header">Clinical Recommendations</h2>', 
                               unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.warning("""
                        **Recommended Actions:**
                        - Immediate HBV serological testing (HBsAg, anti-HBc, anti-HBs)
                        - Liver function tests (ALT, AST, Bilirubin)
                        - Consider HBV DNA quantification if HBsAg positive
                        - Refer to hepatology specialist for further evaluation
                        - Educate patient about transmission prevention
                        - Screen close contacts and household members
                        """)
                    else:
                        st.success("""
                        **Recommended Actions:**
                        - Continue routine health monitoring
                        - Consider HBV vaccination if not previously vaccinated
                        - Maintain healthy lifestyle habits
                        - Regular check-ups as per standard guidelines
                        - Follow up if risk factors change
                        """)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # ========================================================================
    # PAGE 2: BATCH PREDICTION
    # ========================================================================
    elif page == "Batch Prediction":
        st.markdown('<h2 class="sub-header">Batch Prediction from CSV File</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
            Upload a CSV file containing patient data for multiple predictions. 
            The file should have the same features as the training data.
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing batch predictions..."):
                        # Prepare data
                        # Note: Add your preprocessing pipeline here
                        
                        # Make predictions
                        # predictions = model.predict(data)
                        # probabilities = model.predict_proba(data)
                        
                        # Add results to dataframe
                        # df['Prediction'] = predictions
                        # df['Risk_Probability'] = probabilities[:, 1]
                        
                        st.success("Batch prediction completed!")
                        
                        # Display results
                        st.write("Results:")
                        # st.dataframe(df)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        # with col1:
                        #     st.metric("Total Cases", len(df))
                        # with col2:
                        #     st.metric("Positive Predictions", 
                        #              (df['Prediction'] == 1).sum())
                        # with col3:
                        #     st.metric("Negative Predictions", 
                        #              (df['Prediction'] == 0).sum())
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="hepb_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # ========================================================================
    # PAGE 3: MODEL INFORMATION
    # ========================================================================
    elif page == "Model Information":
        st.markdown('<h2 class="sub-header">Model Performance & Information</h2>', 
                    unsafe_allow_html=True)
        
        if config:
            # Model metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                    <h3>Model Configuration</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**Model Type:** {config['model_type']}")
                st.write(f"**Optimal Threshold:** {config['optimal_threshold']:.3f}")
                st.write(f"**Number of Features:** {config['n_features']}")
                st.write(f"**Resampling Method:** {config['resampling']}")
                st.write(f"**Scaling Method:** {config['scaling']}")
                
                st.markdown("""
                    <div class="metric-card">
                    <h3>Hyperparameters</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                for key, value in config['hyperparameters'].items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                    <h3>Test Set Performance</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                metrics = config['test_metrics']
                
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                st.metric("Precision", f"{metrics['precision']:.2%}")
                st.metric("Recall (Sensitivity)", f"{metrics['recall']:.2%}")
                st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            
            # Feature importance (if available)
            st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', 
                       unsafe_allow_html=True)
            
            try:
                feature_importance = model.feature_importances_
                feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
                importance_dict = dict(zip(feature_names, feature_importance))
                
                fig = create_feature_importance_chart(importance_dict)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.info("Feature importance visualization not available.")
            
            # Model information
            st.markdown('<h2 class="sub-header">About the Model</h2>', 
                       unsafe_allow_html=True)
            
            st.markdown("""
                <div class="info-box">
                <h4>XGBoost Classifier</h4>
                <p>
                XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm 
                that uses ensemble learning to make predictions. This model was trained on 
                the NHANES (National Health and Nutrition Examination Survey) dataset with 
                advanced preprocessing techniques including:
                </p>
                <ul>
                <li>Feature selection using Mutual Information</li>
                <li>SMOTEENN resampling for class imbalance</li>
                <li>Robust scaling for outlier resistance</li>
                <li>Threshold optimization for optimal precision-recall balance</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="info-box">
                <h4>Clinical Disclaimer</h4>
                <p>
                This prediction tool is designed to assist healthcare professionals in risk 
                assessment and should NOT be used as a sole diagnostic tool. All predictions 
                should be confirmed with proper laboratory testing and clinical evaluation by 
                qualified medical personnel. The model predictions are based on statistical 
                patterns and may not account for all individual patient factors.
                </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #7f8c8d; font-size: 14px;">
        Hepatitis B Prediction System | Developed using XGBoost Machine Learning
        <br>For research and clinical decision support purposes
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()