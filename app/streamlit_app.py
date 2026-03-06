"""
Heart Disease Diagnosis - Enhanced Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import json
from datetime import datetime
from pathlib import Path

# IMPORTANT: Import model_functions first (needed for unpickling models)
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add src/utils to path for app_utils
utils_path = Path(__file__).parent.parent / "src" / "utils"
sys.path.insert(0, str(utils_path))

# Import model functions and FE transformers
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Try importing from app directory first (same directory as this file)
    from model_functions import BasicFE, EnhancedFE, PolyFE
    print("[OK] Imported FE classes from app/model_functions.py")
except ImportError as e1:
    try:
        # Try importing from src package (if running from project root)
        from src.model_functions import BasicFE, EnhancedFE, PolyFE
        print("[OK] Imported FE classes from src.model_functions")
    except ImportError as e2:
        try:
            # Last resort: use importlib to dynamically load from src/
            import importlib.util
            
            model_functions_path = Path(__file__).parent.parent / "src" / "model_functions.py"
            spec = importlib.util.spec_from_file_location("model_functions", model_functions_path)
            model_functions = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_functions)
            
            BasicFE = model_functions.BasicFE
            EnhancedFE = model_functions.EnhancedFE
            PolyFE = model_functions.PolyFE
            print("[OK] Imported FE classes using importlib from src/")
        except Exception as e3:
            print(f"[ERROR] All import attempts failed:")
            print(f"  1. app/model_functions.py: {e1}")
            print(f"  2. src.model_functions: {e2}")
            print(f"  3. importlib: {e3}")
            raise ImportError("Could not import BasicFE, EnhancedFE, PolyFE from any location")

# Make these available in __main__ namespace for pickle compatibility in Streamlit
import __main__

__main__.BasicFE = BasicFE
__main__.EnhancedFE = EnhancedFE
__main__.PolyFE = PolyFE

# Import custom modules
from pipeline import pipeline
from utils.app_utils import (
    PatientHistoryManager,
    create_patient_report_pdf,
    create_feature_importance_plot,
    calculate_model_agreement,
    get_feature_descriptions,
    get_preset_examples,
    calculate_input_contribution,
    create_input_contribution_plot,
    create_contribution_summary_table,
    generate_personalized_recommendations,
    create_personalized_recommendations_text,
)

# Add scripts to path for experiment_manager
scripts_path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_path))
from experiment_manager import ExperimentManager

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Heart Disease Diagnosis - Enhanced",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1976d2;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .disease-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
    }
    .healthy-card {
        background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976d2;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "history_manager" not in st.session_state:
    st.session_state.history_manager = PatientHistoryManager()

if "pipeline_initialized" not in st.session_state:
    st.session_state.pipeline_initialized = False

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# ============================================================================
# SIDEBAR - INPUT FORM
# ============================================================================


def get_user_input():
    """Enhanced user input form với tooltips và validation"""
    st.sidebar.header("🏥 Patient Data Input")

    # Preset selector
    feature_desc = get_feature_descriptions()
    presets = get_preset_examples()

    st.sidebar.subheader("Quick Presets")
    preset_choice = st.sidebar.selectbox(
        "Select Preset Example:", ["Custom"] + list(presets.keys())
    )

    if preset_choice != "Custom":
        preset_data = presets[preset_choice]
    else:
        preset_data = {}

    st.sidebar.markdown("---")
    st.sidebar.subheader("Patient Information")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=preset_data.get("age", 50),
            help="Patient's age in years (typical range: 29-77)",
        )

        sex_option = st.selectbox(
            "Sex",
            ["Male", "Female"],
            index=0 if preset_data.get("sex", 1) == 1 else 1,
            help="Patient's biological sex",
        )
        sex = 1 if sex_option == "Male" else 0

        cp_options = [
            ("Typical Angina", 1),
            ("Atypical Angina", 2),
            ("Non-anginal Pain", 3),
            ("Asymptomatic", 4),
        ]
        cp_default = preset_data.get("cp", 4) - 1
        cp = st.selectbox(
            "Chest Pain Type",
            cp_options,
            index=cp_default,
            format_func=lambda x: x[0],
            help="Type of chest pain experienced by the patient",
        )

        trestbps = st.slider(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=preset_data.get("trestbps", 120),
            help="Resting blood pressure in mm Hg (normal: ~120)",
        )

        chol = st.slider(
            "Serum Cholesterol",
            min_value=100,
            max_value=600,
            value=preset_data.get("chol", 200),
            help="Serum cholesterol in mg/dl (normal: <200)",
        )

        fbs_option = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["False", "True"],
            index=preset_data.get("fbs", 0),
            help="Is fasting blood sugar greater than 120 mg/dl?",
        )
        fbs = 1 if fbs_option == "True" else 0

    with col2:
        restecg_options = [
            ("Normal", 0),
            ("ST-T wave abnormality", 1),
            ("Left ventricular hypertrophy", 2),
        ]
        restecg = st.selectbox(
            "Resting ECG",
            restecg_options,
            index=preset_data.get("restecg", 0),
            format_func=lambda x: x[0],
            help="Resting electrocardiographic results",
        )

        thalach = st.slider(
            "Max Heart Rate",
            min_value=50,
            max_value=220,
            value=preset_data.get("thalach", 150),
            help="Maximum heart rate achieved (normal: 220 - age)",
        )

        exang_option = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"],
            index=preset_data.get("exang", 0),
            help="Does exercise induce angina (chest pain)?",
        )
        exang = 1 if exang_option == "Yes" else 0

        oldpeak = st.slider(
            "ST Depression",
            min_value=0.0,
            max_value=10.0,
            value=float(preset_data.get("oldpeak", 1.0)),
            step=0.1,
            help="ST depression induced by exercise relative to rest",
        )

        slope_options = [("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)]
        slope = st.selectbox(
            "Slope of ST Segment",
            slope_options,
            index=preset_data.get("slope", 1) - 1,
            format_func=lambda x: x[0],
            help="The slope of the peak exercise ST segment",
        )

        ca_options = [("0", 0), ("1", 1), ("2", 2), ("3", 3)]
        ca = st.selectbox(
            "Number of Major Vessels",
            ca_options,
            index=preset_data.get("ca", 0),
            format_func=lambda x: x[0],
            help="Number of major vessels colored by fluoroscopy (0-3)",
        )

        thal_options = [("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)]
        thal_default = (
            0
            if preset_data.get("thal", 3) == 3
            else (1 if preset_data.get("thal", 3) == 6 else 2)
        )
        thal = st.selectbox(
            "Thalassemia",
            thal_options,
            index=thal_default,
            format_func=lambda x: x[0],
            help="Thalassemia blood disorder status",
        )

    user_data = {
        "age": age,
        "sex": sex,
        "cp": cp[1],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg[1],
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope[1],
        "ca": ca[1],
        "thal": thal[1],
    }

    feature_order = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    return pd.DataFrame([user_data])[feature_order], user_data


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown(
    '<h1 class="main-header">🫀 Heart Disease Diagnosis System</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 1.1rem;">Advanced ML-Based Diagnosis with Multiple Models </p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Get user input
user_input_df, user_data_dict = get_user_input()

# Initialize pipeline button
if not st.session_state.pipeline_initialized:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Initialize Prediction System", type="primary", width="stretch"
        ):
            with st.spinner("Loading models and initializing system..."):
                if pipeline.initialize():
                    st.session_state.pipeline_initialized = True
                    st.success("✅ System initialized successfully!")
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
                else:
                    st.error(
                        "❌ Failed to initialize system. Please check model files."
                    )
                    st.stop()

if st.session_state.pipeline_initialized:
    # Display model performance summary at the top
    if hasattr(pipeline, "metrics") and pipeline.metrics:
        with st.expander("📊 Model Performance Summary", expanded=False):
            metrics_data = []
            for name, metric in pipeline.metrics.items():
                # Kiểm tra model có active không
                if name in [m for m in pipeline.models.keys()]:
                    status = "✅ Active"
                else:
                    status = "⚠️ Not Loaded"

                # Lấy metrics từ cấu trúc mới
                cv_score = metric.get("cv_optimization_score", 0)
                opt_metric = metric.get("optimization_metric", "roc_auc")
                test_metrics = metric.get("test_metrics", {})
                test_auc = test_metrics.get("roc_auc", 0)

                metrics_data.append(
                    {
                        "Model": name,
                        "Status": status,
                        f"CV {opt_metric.upper()}": f"{cv_score:.4f}",
                        "Test AUC": f"{test_auc:.4f}",
                    }
                )

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                # Sort by Test AUC descending
                metrics_df = metrics_df.sort_values("Test AUC", ascending=False)
                st.dataframe(metrics_df, width="stretch", hide_index=True)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏥 Patient Input & Prediction",
            "📊 Model Analysis",
            "🔬 Feature Importance",
            "📈 Experiment Tracking",
            "📝 History & Reports",
        ]
    )

    # ========================================================================
    # TAB 1: PATIENT INPUT & PREDICTION
    # ========================================================================
    with tab1:
        st.subheader("📋 Patient Input Data")
        st.dataframe(user_input_df, width="stretch")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔮 Run Diagnosis (All Models)",
                type="primary",
                width="stretch",
            )

        if predict_button:
            with st.spinner("Running predictions from all models..."):
                try:
                    all_results, predictions = pipeline.predict(user_input_df)

                    if not all_results:
                        st.error(
                            "❌ No predictions could be made. Please check your input data."
                        )
                        st.stop()

                    # Save to session state
                    st.session_state.predictions = (all_results, predictions)
                    st.session_state.patient_data = user_data_dict

                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())
                    st.stop()

        # Display results if available
        if st.session_state.predictions:
            all_results, predictions = st.session_state.predictions

            st.markdown("---")

            # Final Verdict with Majority Vote
            final_prediction, vote_count, total_models = pipeline.get_majority_vote(
                predictions
            )
            final_prediction_label = (
                "Heart Disease"
                if final_prediction == "High Risk"
                else "No Heart Disease"
            )

            st.subheader("🎯 Final Diagnosis (Majority Vote)")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                card_class = (
                    "disease-card"
                    if final_prediction_label == "Heart Disease"
                    else "healthy-card"
                )
                st.markdown(
                    f"""
                <div class="prediction-card {card_class}">
                    <h2 style="margin: 0; font-size: 2rem;">{final_prediction_label}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                        {vote_count} out of {total_models} models agree
                    </p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">
                        Confidence: {(vote_count/total_models)*100:.1f}%
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Model Agreement Statistics
            agreement_stats = calculate_model_agreement(predictions)

            st.markdown("---")
            st.subheader("📊 Model Agreement Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{agreement_stats['total_models']}</div>
                    <div class="metric-label">Total Models</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{agreement_stats['disease_votes']}</div>
                    <div class="metric-label">Disease Votes</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{agreement_stats['no_disease_votes']}</div>
                    <div class="metric-label">No Disease Votes</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{agreement_stats['consensus']}</div>
                    <div class="metric-label">Consensus Level</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Individual Model Predictions
            st.markdown("---")
            st.subheader("🤖 Individual Model Predictions")

            # Create columns for model cards
            num_models = len(all_results)
            cols = st.columns(min(4, num_models))

            for idx, result in enumerate(all_results):
                col_idx = idx % 4
                prediction_label = (
                    "Heart Disease"
                    if result["Prediction"] == "High Risk"
                    else "No Heart Disease"
                )
                color = "#d32f2f" if prediction_label == "Heart Disease" else "#388e3c"

                with cols[col_idx]:
                    st.markdown(
                        f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; 
                                border-left: 5px solid {color}; margin-bottom: 1rem; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0 0 0.5rem 0; color: #333;">{result['Model']}</h4>
                        <p style="margin: 0; color: {color}; font-weight: bold; font-size: 1.1rem;">
                            {prediction_label}
                        </p>
                        <p style="margin: 0.25rem 0 0 0; color: #666;">
                            Confidence: <strong>{result['Confidence']:.1%}</strong>
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            # Visualizations
            st.markdown("---")
            st.subheader("📈 Model Predictions & Confidence")

            # Create summary dataframe
            summary_df = pd.DataFrame(all_results)
            summary_df["Prediction"] = summary_df["Prediction"].apply(
                lambda x: "Heart Disease" if x == "High Risk" else "No Heart Disease"
            )

            # Single bar chart showing all models
            fig = go.Figure()

            for _, row in summary_df.iterrows():
                color = "#d32f2f" if row["Prediction"] == "Heart Disease" else "#388e3c"

                fig.add_trace(
                    go.Bar(
                        x=[row["Model"]],
                        y=[row["Confidence"]],
                        name=row["Prediction"],
                        marker_color=color,
                        text=f"<b>{row['Prediction']}</b>",
                        textposition="inside",
                        textfont=dict(color="white", size=12),
                        hovertemplate=f"<b>{row['Model']}</b><br>"
                        + f"Prediction: {row['Prediction']}<br>"
                        + f"Confidence: {row['Confidence']:.1%}<br>"
                        + "<extra></extra>",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                title="All Model Predictions",
                xaxis_title="Model",
                yaxis_title="Confidence",
                yaxis=dict(tickformat=".0%", range=[0, 1.05]),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.markdown("---")
            st.subheader("📋 Detailed Results Table")
            display_df = summary_df.copy()
            display_df["Confidence"] = display_df["Confidence"].apply(
                lambda x: f"{x:.2%}"
            )
            st.dataframe(display_df, width="stretch")

            # ========================================
            # INPUT CONTRIBUTION ANALYSIS
            # ========================================
            st.markdown("---")
            st.subheader("🎯 Your Input Contribution Analysis")
            st.info(
                """
            📌 **Personalized Analysis:** See how YOUR specific input values contributed to each model's prediction.
            Select a model below to understand which of your values were most influential.
            """
            )

            # Model selector for contribution analysis
            contrib_model_names = list(pipeline.models.keys())
            selected_contrib_model = st.selectbox(
                "Select Model for Input Contribution Analysis:",
                contrib_model_names,
                key="contrib_model_selector",
            )

            if selected_contrib_model:
                try:
                    model_pipeline = pipeline.models[selected_contrib_model]
                    model = model_pipeline["model"]

                    # Get feature names
                    if hasattr(model, "feature_importances_"):
                        n_features = len(model.feature_importances_)
                    elif hasattr(model, "coef_"):
                        n_features = (
                            len(model.coef_[0])
                            if len(model.coef_.shape) > 1
                            else len(model.coef_)
                        )
                    else:
                        n_features = len(pipeline.feature_order)

                    feature_names = pipeline.feature_order[:n_features]

                    # Get the prediction result for this model
                    model_result = next(
                        (
                            r
                            for r in all_results
                            if r["Model"] == selected_contrib_model
                        ),
                        None,
                    )

                    if model_result:
                        prediction_result = (
                            "Heart Disease"
                            if model_result["Prediction"] == "High Risk"
                            else "No Heart Disease"
                        )

                        # Calculate input contributions
                        contributions = calculate_input_contribution(
                            model=model,
                            input_data=user_input_df,
                            feature_names=feature_names,
                            model_name=selected_contrib_model,
                        )

                        if contributions:
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                # Create contribution visualization (pie chart)
                                contrib_fig = create_input_contribution_plot(
                                    contributions=contributions,
                                    input_data=st.session_state.patient_data,
                                    model_name=selected_contrib_model,
                                    prediction_result=prediction_result,
                                )

                                if contrib_fig:
                                    st.plotly_chart(
                                        contrib_fig, use_container_width=True
                                    )
                                else:
                                    st.warning(
                                        "Could not generate contribution visualization."
                                    )

                            with col2:
                                # Key insights in sidebar
                                st.markdown("### 🔍 Key Insights")

                                # Find top positive and negative contributors
                                sorted_contrib = sorted(
                                    contributions.items(),
                                    key=lambda x: x[1]["raw_contribution"],
                                    reverse=True,
                                )

                                if len(sorted_contrib) > 0:
                                    # Top risk factor
                                    top_risk = sorted_contrib[0]
                                    if top_risk[1]["raw_contribution"] > 0:
                                        st.markdown(
                                            f"""
                                        🔴 **Top Risk Factor**  
                                        **{top_risk[0].replace('_', ' ').title()}**  
                                        Value: `{st.session_state.patient_data.get(top_risk[0], 'N/A')}`
                                        """
                                        )

                                    # Top protective factor
                                    top_protective = sorted_contrib[-1]
                                    if top_protective[1]["raw_contribution"] < 0:
                                        st.markdown(
                                            f"""
                                        🟢 **Top Protective Factor**  
                                        **{top_protective[0].replace('_', ' ').title()}**  
                                        Value: `{st.session_state.patient_data.get(top_protective[0], 'N/A')}`
                                        """
                                        )

                                    # Model confidence
                                    st.markdown(
                                        f"""
                                    🎯 **Model Confidence**  
                                    `{model_result['Confidence']:.1%}`
                                    """
                                    )

                            # Detailed contribution table
                            st.markdown("---")
                            st.subheader("📋 Detailed Contribution Breakdown")

                            # Create summary table
                            feature_desc = get_feature_descriptions()
                            summary_df_contrib = create_contribution_summary_table(
                                contributions=contributions,
                                input_data=st.session_state.patient_data,
                                feature_descriptions=feature_desc,
                            )

                            st.dataframe(summary_df_contrib, width="stretch")

                            # ========================================
                            # PERSONALIZED RECOMMENDATIONS
                            # ========================================
                            st.markdown("---")
                            st.subheader("🎯 Personalized Action Plan")
                            st.info(
                                """
                            📌 **Smart Recommendations:** Based on your input values and their contribution to the prediction,
                            here are the TOP 3 modifiable factors that would have the biggest impact on reducing your heart disease risk.
                            """
                            )

                            # Generate recommendations
                            feature_desc = get_feature_descriptions()
                            recommendations_data = (
                                generate_personalized_recommendations(
                                    patient_data=st.session_state.patient_data,
                                    contributions=contributions,
                                    feature_descriptions=feature_desc,
                                )
                            )

                            if recommendations_data["recommendations"]:
                                # Display recommendations as cards
                                for rec in recommendations_data["recommendations"]:
                                    priority_color = {
                                        "High": "#d32f2f",
                                        "Medium": "#f57c00",
                                        "Low": "#388e3c",
                                    }.get(rec["priority"], "#666666")

                                    st.markdown(
                                        f"""
                                    <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                                                border-left: 5px solid {priority_color}; margin-bottom: 1rem; 
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <h4 style="margin: 0 0 0.5rem 0; color: {priority_color};">
                                            {rec['rank']}. {rec['feature']} - {rec['priority']} Priority
                                        </h4>
                                        <p style="margin: 0.25rem 0; color: #666;">
                                            <strong>Current Value:</strong> {rec['current_value']} | 
                                            <strong>Target:</strong> {rec['target_range']}
                                        </p>
                                        <p style="margin: 0.5rem 0 0 0; color: #333;">
                                            <strong>Action Plan:</strong> {rec['recommendation']}
                                        </p>
                                        <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.9rem;">
                                            Risk Contribution: {float(rec['risk_contribution']):.3f}
                                        </p>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )

                                # Implementation timeline
                                with st.expander(
                                    "📅 Implementation Timeline & Next Steps"
                                ):
                                    st.markdown(
                                        """
                                    **📋 Implementation Timeline:**
                                    - **Week 1-2:** Consult with your healthcare provider about these specific areas
                                    - **Month 1:** Begin implementing lifestyle changes for your top priority factor
                                    - **Month 2-3:** Add interventions for your second and third priority factors
                                    - **Month 6:** Follow-up assessment to measure improvements
                                    
                                    **🏥 Next Steps:**
                                    1. Schedule appointment with your primary care physician
                                    2. Share this analysis with your healthcare team
                                    3. Consider referral to specialists (cardiologist, nutritionist) as needed
                                    4. Begin with the highest priority recommendation first
                                    
                                    **⚠️ Important Notes:**
                                    - These recommendations are based on AI analysis and should complement, not replace, professional medical advice
                                    - Work with your healthcare team to create a safe and effective improvement plan
                                    - Regular monitoring is essential to track progress and adjust strategies
                                    """
                                    )
                            else:
                                st.success(
                                    """
                                🎉 **Great News!** Based on your input values, you don't have major modifiable risk factors 
                                that are significantly contributing to heart disease risk. Continue maintaining your healthy lifestyle:
                                
                                • Regular health check-ups (annual cardiac screening)
                                • Maintain a balanced diet and regular exercise  
                                • Monitor blood pressure and cholesterol levels
                                • Avoid smoking and excessive alcohol consumption
                                """
                                )
                        else:
                            st.warning(
                                f"Could not calculate contributions for {selected_contrib_model}. This might be due to model compatibility issues."
                            )
                    else:
                        st.warning("Model prediction result not found.")

                except Exception as e:
                    st.error(f"Error calculating input contributions: {str(e)}")
                    import traceback

                    with st.expander("Debug Information"):
                        st.code(traceback.format_exc())

            # Save to history and download report
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Patient ID (optional)**")
                patient_id = st.text_input(
                    "Patient ID",
                    value="",
                    label_visibility="collapsed",
                    placeholder="Enter patient ID...",
                )

            with col2:
                st.markdown("&nbsp;")  # Empty space to maintain layout
                if st.button("💾 Save to History", width="stretch"):
                    if not patient_id:
                        patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    st.session_state.history_manager.add_prediction(
                        patient_id,
                        st.session_state.patient_data,
                        all_results,
                        final_prediction_label,
                    )
                    st.success(f"✅ Saved to history with ID: {patient_id}")

            with col3:
                st.markdown("&nbsp;")  # Empty space to maintain layout
                try:
                    # Get contributions for PDF if available
                    pdf_contributions = None
                    if "selected_contrib_model" in locals() and selected_contrib_model:
                        try:
                            model_pipeline = pipeline.models[selected_contrib_model]
                            model = model_pipeline["model"]

                            if hasattr(model, "feature_importances_"):
                                n_features = len(model.feature_importances_)
                            elif hasattr(model, "coef_"):
                                n_features = (
                                    len(model.coef_[0])
                                    if len(model.coef_.shape) > 1
                                    else len(model.coef_)
                                )
                            else:
                                n_features = len(pipeline.feature_order)

                            feature_names = pipeline.feature_order[:n_features]

                            pdf_contributions = calculate_input_contribution(
                                model=model,
                                input_data=user_input_df,
                                feature_names=feature_names,
                                model_name=selected_contrib_model,
                            )
                        except:
                            pdf_contributions = None

                    pdf_buffer = create_patient_report_pdf(
                        st.session_state.patient_data,
                        all_results,
                        final_prediction_label,
                        pdf_contributions,
                    )
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )
                except Exception as e:
                    st.warning(f"PDF generation failed: {str(e)}")

    # ========================================================================
    # TAB 2: MODEL ANALYSIS
    # ========================================================================
    with tab2:
        st.subheader("📊 Model Performance Analysis")

        if hasattr(pipeline, "metrics") and pipeline.metrics:
            # Display metrics với cấu trúc JSON mới
            metrics_data = []
            for name, metric in pipeline.metrics.items():
                # Kiểm tra model có active không
                if name in [m for m in pipeline.models.keys()]:
                    status = "✅ Active"
                else:
                    status = "⚠️ Not Loaded"

                # Lấy metrics từ cấu trúc mới
                cv_score = metric.get("cv_optimization_score", 0)
                opt_metric = metric.get("optimization_metric", "roc_auc")
                test_metrics = metric.get("test_metrics", {})

                # Các metrics chi tiết
                test_auc = test_metrics.get("roc_auc", 0)
                accuracy = test_metrics.get("accuracy", 0)
                precision = test_metrics.get("precision", 0)
                recall = test_metrics.get("recall_sensitivity", 0)  # Key mới
                f1 = test_metrics.get("f1_score", 0)  # Key mới
                specificity = test_metrics.get("specificity", 0)  # Key mới

                # Metadata mới
                fe = metric.get("feature_engineering", "N/A")
                scaler = metric.get("scaler", "N/A")
                fs = metric.get("feature_selection", "N/A")

                metrics_data.append(
                    {
                        "Model": name,
                        "Status": status,
                        f"CV {opt_metric.upper()}": f"{cv_score:.4f}",
                        "Test AUC": f"{test_auc:.4f}",
                        "Accuracy": f"{accuracy:.4f}",
                        "Precision": f"{precision:.4f}",
                        "Recall": f"{recall:.4f}",
                        "F1": f"{f1:.4f}",
                        "Specificity": f"{specificity:.4f}",
                        "Feature Engineering": fe,
                        "Scaler": scaler,
                        "Feature Selection": fs,
                    }
                )

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                # Sort by Accuracy descending
                metrics_df = metrics_df.sort_values("Accuracy", ascending=False)
                st.dataframe(metrics_df, width="stretch", hide_index=True)

                # Visualization cập nhật
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        metrics_df,
                        x="Model",
                        y="Test AUC",
                        title="Test AUC Scores",
                        color="Test AUC",
                        color_continuous_scale="Viridis",
                    )
                    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig2 = px.bar(
                        metrics_df,
                        x="Model",
                        y="Accuracy",
                        title="Accuracy Scores",
                        color="Accuracy",
                        color_continuous_scale="RdYlGn",
                    )
                    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ℹ️ Model metrics not available. Please check the metrics file.")

    # ========================================================================
    # TAB 3: FEATURE IMPORTANCE
    # ========================================================================
    with tab3:
        st.subheader("🔬 Model Feature Importance")

        st.info(
            """
        📌 **Global Feature Importance:** Shows which features are generally most important 
        for each model across all patients. This is the model's "global" perspective on feature significance.
        
        💡 **Note:** For personalized analysis of YOUR specific input values, see the 
        "Input Contribution Analysis" section in the **Patient Input & Prediction** tab.
        """
        )

        if st.session_state.predictions and pipeline.models:
            # Model selector
            model_names = list(pipeline.models.keys())
            selected_model_name = st.selectbox(
                "Select Model for Global Feature Importance", model_names
            )

            if selected_model_name:
                try:
                    model_pipeline = pipeline.models[selected_model_name]
                    model = model_pipeline["model"]

                    # Get actual feature names from stored model info
                    stored_feature_names = model_pipeline.get("feature_names")

                    # Determine number of features the model expects
                    if hasattr(model, "feature_importances_"):
                        n_features = len(model.feature_importances_)
                    elif hasattr(model, "coef_"):
                        n_features = (
                            len(model.coef_[0])
                            if len(model.coef_.shape) > 1
                            else len(model.coef_)
                        )
                    else:
                        n_features = len(pipeline.feature_order)

                    # Try to reconstruct feature names from preprocessing pipeline
                    if (
                        stored_feature_names is not None
                        and len(stored_feature_names) == n_features
                    ):
                        feature_names = stored_feature_names
                    else:
                        # Try to get feature names from preprocessor
                        preprocessor = model_pipeline.get("preprocessor")
                        fe_transformer = model_pipeline.get("fe_transformer")

                        if preprocessor is not None:
                            try:
                                # Get feature names after preprocessing (scaling + one-hot encoding)
                                if hasattr(preprocessor, "get_feature_names_out"):
                                    preprocessed_names = (
                                        preprocessor.get_feature_names_out()
                                    )
                                else:
                                    # Fallback: construct names manually
                                    numerical_features = model_pipeline.get(
                                        "numerical_features", []
                                    )
                                    categorical_features = model_pipeline.get(
                                        "categorical_features", []
                                    )
                                    preprocessed_names = (
                                        numerical_features + categorical_features
                                    )

                                # Apply feature selection indices if available
                                fs_indices = model_pipeline.get("fs_indices")
                                if fs_indices is not None and len(fs_indices) > 0:
                                    feature_names = [
                                        (
                                            preprocessed_names[i]
                                            if i < len(preprocessed_names)
                                            else f"feature_{i}"
                                        )
                                        for i in fs_indices
                                    ]
                                else:
                                    feature_names = list(preprocessed_names)[
                                        :n_features
                                    ]
                            except Exception as e:
                                # Fallback to original feature names if reconstruction fails
                                feature_names = pipeline.feature_order[:n_features]
                        else:
                            # No preprocessor, use original feature names
                            feature_names = pipeline.feature_order[:n_features]

                    # Ensure feature_names length matches n_features
                    if len(feature_names) != n_features:
                        # Pad or trim to match
                        if len(feature_names) < n_features:
                            feature_names.extend(
                                [
                                    f"feature_{i}"
                                    for i in range(len(feature_names), n_features)
                                ]
                            )
                        else:
                            feature_names = feature_names[:n_features]

                    # Create feature importance plot
                    fig = create_feature_importance_plot(
                        model, feature_names, selected_model_name
                    )

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                        # Add interpretation
                        st.markdown("---")
                        st.subheader("📖 How to Interpret")
                        st.markdown(
                            """
                        - **Higher values** = More important for the model's decisions
                        - **Lower values** = Less influential in predictions
                        - **Zero values** = Not used by this model (if feature selection was applied)
                        
                        ⚠️ **Important:** This shows general importance across all patients, not specific to your input.
                        """
                        )
                    else:
                        # Provide helpful explanation for models without feature importance
                        st.info(
                            f"ℹ️ **Feature Importance Not Available for {selected_model_name}**"
                        )

                        # Model-specific explanations
                        if selected_model_name == "K-Nearest Neighbors":
                            st.markdown(
                                """
                            **Why doesn't KNN have Feature Importance?**
                            
                            🔍 **How KNN Works:**
                            - KNN is an **instance-based learning** algorithm
                            - It doesn't build a model or learn rules, just memorizes all training data
                            - For predictions: Finds K nearest neighbors → Takes majority vote
                            
                            💡 **Insight:**
                            - All features are used equally to calculate distances
                            - No ranking or weights assigned to individual features
                            - This is a "democratic" model - every feature matters equally
                            """
                            )

                        elif selected_model_name == "Naive Bayes":
                            st.markdown(
                                """
                            **Why doesn't Naive Bayes have Feature Importance?**
                            
                            🔍 **How Naive Bayes Works:**
                            - Based on **Bayes' Theorem** (conditional probability)
                            - Assumption: All features are **independent** of each other
                            - Calculates P(Disease|Features) = P(Features|Disease) × P(Disease)
                            
                            💡 **Insight:**
                            - Each feature has its own probability, not directly comparable
                            - All features contribute to the final probability
                            - This is a "probabilistic" model - no concept of importance ranking
                            """
                            )

                        elif selected_model_name == "Ensemble":
                            st.markdown(
                                """
                            **Why doesn't Ensemble have Direct Feature Importance?**
                            
                            🔍 **How Ensemble (Voting) Works:**
                            - Combines predictions from **multiple models** (LR + RF + SVM)
                            - Each model votes → Takes majority decision
                            - It's a "combination" rather than a single model
                            
                            💡 **Insight:**
                            - Each base model has its own feature importance
                            - You can view importance for individual base models (LR, RF, SVM)
                            - Ensemble focuses on **consensus**, not individual features
                            
                            ✅ **Suggestion:** Select Logistic Regression or Random Forest to view feature importance!
                            """
                            )
                        else:
                            st.markdown(
                                """
                            This model does not support feature importance analysis.
                            Try other models like **Random Forest**, **Logistic Regression**, or **Gradient Boosting**!
                            """
                            )

                except Exception as e:
                    st.error(f"Error analyzing model features: {str(e)}")

            # Feature descriptions
            st.markdown("---")
            st.subheader("📖 Feature Descriptions")

            feature_desc = get_feature_descriptions()
            desc_data = []

            if "feature_names" in locals():
                for feat_name in feature_names:
                    if feat_name in feature_desc:
                        desc = feature_desc[feat_name]
                        desc_data.append(
                            {
                                "Feature": desc["name"],
                                "Description": desc["description"],
                                "Range/Values": desc.get("normal_range", "")
                                or str(desc.get("values", "")),
                            }
                        )

                st.table(pd.DataFrame(desc_data))
        else:
            st.info(
                "ℹ️ Please run a prediction first to see feature importance analysis."
            )

    # ========================================================================
    # TAB 4: EXPERIMENT TRACKING
    # ========================================================================
    with tab4:
        st.subheader("📈 Experiment Tracking & Management")

        st.info(
            """
        📌 **Experiment Management:** Track all model training experiments, compare configurations,
        and analyze performance metrics across different setups.
        """
        )

        try:
            exp_manager = ExperimentManager()

            if len(exp_manager.experiments) > 0:
                # Display statistics
                summary_stats = exp_manager.get_summary_statistics()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Experiments", summary_stats.get("total_experiments", 0)
                    )
                with col2:
                    st.metric(
                        "Models Tested", len(summary_stats.get("models_tested", []))
                    )
                with col3:
                    # Show best performing model
                    if exp_manager.experiments:
                        best_exp = max(
                            exp_manager.experiments,
                            key=lambda x: x.get("metrics", {}).get("f1", x.get("metrics", {}).get("accuracy", 0)),
                        )
                        best_score = best_exp.get("metrics", {}).get("f1", best_exp.get("metrics", {}).get("accuracy", 0))
                        st.metric(
                            "Best F1 Score",
                            f"{best_score:.2%}",
                        )

                # Filters
                st.markdown("---")
                st.subheader("🔍 Filter Experiments")

                filter_model = st.selectbox(
                    "Filter by Model", ["All"] + summary_stats.get("models_tested", [])
                )

                # Get filtered experiments
                model_filter = None if filter_model == "All" else filter_model

                comparison_df = exp_manager.compare_experiments(
                    filter_model=model_filter,
                    filter_dataset=None,
                    sort_by=(
                        "accuracy"
                        if "accuracy" in exp_manager.experiments[0].get("metrics", {})
                        else "f1"
                    ),
                )

                if not comparison_df.empty:
                    # Remove dataset_type column if it exists (we don't track datasets)
                    display_df = comparison_df.copy()
                    if "dataset_type" in display_df.columns:
                        display_df = display_df.drop("dataset_type", axis=1)

                    st.dataframe(display_df, width="stretch")

                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Generate HTML Report"):
                            report_path = exp_manager.generate_report(
                                output_format="html"
                            )
                            st.success(f"✅ Report generated: {report_path}")

                    with col2:
                        csv_data = comparison_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download as CSV",
                            data=csv_data,
                            file_name=f"experiments_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )
                else:
                    st.warning("No experiments match the selected filters.")
            else:
                st.info(
                    """
                ℹ️ No experiments logged yet. 
                
                To start logging experiments, run the hyperparameter tuning script:
                ```bash
                python hyperparameter_tuning.py
                ```
                """
                )

        except Exception as e:
            st.error(f"Error loading experiments: {str(e)}")

    # ========================================================================
    # TAB 5: HISTORY & REPORTS
    # ========================================================================
    with tab5:
        st.subheader("📝 Prediction History & Reports")

        history = st.session_state.history_manager.get_history()

        if history:
            # Statistics
            stats = st.session_state.history_manager.get_statistics()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", stats["total_predictions"])
            with col2:
                st.metric("Unique Patients", stats["unique_patients"])
            with col3:
                st.metric("Heart Disease Cases", stats["heart_disease_count"])
            with col4:
                st.metric("Healthy Cases", stats["no_disease_count"])

            # Additional insights
            if stats["total_predictions"] > 0:
                disease_rate = (
                    stats["heart_disease_count"] / stats["total_predictions"]
                ) * 100

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Disease Detection Rate", f"{disease_rate:.1f}%")
                with col2:
                    avg_models = sum(len(r["predictions"]) for r in history) / len(
                        history
                    )
                    st.metric("Avg Models per Prediction", f"{avg_models:.1f}")
                with col3:
                    # Calculate average age from history
                    ages = [
                        r["patient_data"].get("age", 0)
                        for r in history
                        if "patient_data" in r
                    ]
                    avg_age = sum(ages) / len(ages) if ages else 0
                    st.metric("Average Patient Age", f"{avg_age:.0f} years")

            st.markdown("---")

            # History table
            history_data = []
            for record in reversed(history):  # Show newest first
                history_data.append(
                    {
                        "Patient ID": record["patient_id"],
                        "Timestamp": record["timestamp"],
                        "Age": record["patient_data"].get("age", "N/A"),
                        "Sex": (
                            "Male"
                            if record["patient_data"].get("sex") == 1
                            else "Female"
                        ),
                        "Final Verdict": record["final_verdict"],
                        "Models Used": len(record["predictions"]),
                    }
                )

            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, width="stretch")

            # Download history
            col1, col2 = st.columns(2)
            with col1:
                csv_data = history_df.to_csv(index=False)
                st.download_button(
                    "📥 Download History as CSV",
                    data=csv_data,
                    file_name=f"patient_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

            with col2:
                if st.button("🗑️ Clear History", type="secondary"):
                    if st.button("⚠️ Confirm Clear History"):
                        st.session_state.history_manager.history = []
                        st.session_state.history_manager._save_history()
                        st.success("✅ History cleared!")
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
        else:
            st.info(
                "ℹ️ No prediction history yet. Make some predictions to see them here!"
            )

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <p style="margin: 0; font-size: 0.9rem; color: #666;">
        <strong>Heart Disease Diagnosis System</strong> - AIO2025 Project 4.2<br>
        Team: VietAI Learning <br>
        <em>⚠️ This system is for educational purposes. Always consult with healthcare professionals for medical diagnosis.</em>
    </p>
</div>
""",
    unsafe_allow_html=True,
)
