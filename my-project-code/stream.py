import streamlit as st
import pandas as pd
import numpy as np
# from ensemble_model import DiabetesEnsembleModel
# from interpretability import ModelInterpreter
# from data_processor import DataProcessor
# from visualizations import create_visualizations
# import plotly.express as px
# import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="Diabetes Prediction Ensemble Model",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Diabetes Prediction Ensemble Model")
    st.markdown("### Using RandomForest, Logistic Regression & SVM with Interpretability")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Data Upload & Training", 
        "Model Performance", 
        "Feature Interpretability", 
        "Individual Predictions"
    ])
    
    if page == "Data Upload & Training":
        data_upload_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Feature Interpretability":
        interpretability_page()
    elif page == "Individual Predictions":
        prediction_page()

def data_upload_page():
    st.header("📊 Data Upload & Model Training")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your diabetes dataset (CSV format)", 
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Load and display data
            df = pd.read_csv(uploaded_file)
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
            
            with col2:
                st.subheader("Dataset Info")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            
            # Target column selection
            st.subheader("🎯 Target Column Selection")
            target_col = st.selectbox(
                "Select the target column (diabetes indicator):",
                df.columns.tolist()
            )
            
            if st.button("🚀 Train Ensemble Model", type="primary"):
                with st.spinner("Training ensemble model... This may take a few minutes."):
                    try:
                        # Process data
                        processor = DataProcessor()
                        X, y = processor.prepare_data(df, target_col)
                        
                        # Train ensemble model
                        ensemble = DiabetesEnsembleModel()
                        ensemble.train(X, y)
                        
                        # Store in session state
                        st.session_state.ensemble_model = ensemble
                        st.session_state.data_processor = processor
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.feature_names = X.columns.tolist()
                        st.session_state.model_trained = True
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Model trained successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.info("Please ensure your CSV file has proper formatting and contains numerical features.")

def model_performance_page():
    st.header("📈 Model Performance Analysis")
    
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model first in the 'Data Upload & Training' page.")
        return
    
    ensemble = st.session_state.ensemble_model
    
    # Display model performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Individual Model Scores")
        scores = ensemble.get_model_scores()
        for model_name, score in scores.items():
            st.metric(model_name, f"{score:.4f}")
    
    with col2:
        st.subheader("🏆 Ensemble Performance")
        ensemble_score = ensemble.get_ensemble_score()
        st.metric("Ensemble Accuracy", f"{ensemble_score:.4f}")
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm_fig = ensemble.plot_confusion_matrix()
    st.plotly_chart(cm_fig, use_container_width=True)
    
    # ROC Curves
    st.subheader("📈 ROC Curves Comparison")
    roc_fig = ensemble.plot_roc_curves()
    st.plotly_chart(roc_fig, use_container_width=True)

def interpretability_page():
    st.header("🔍 Feature Interpretability Analysis")
    
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model first in the 'Data Upload & Training' page.")
        return
    
    ensemble = st.session_state.ensemble_model
    X = st.session_state.X
    feature_names = st.session_state.feature_names
    
    # Initialize interpreter
    interpreter = ModelInterpreter(ensemble, X, feature_names)
    
    # Feature Importance
    st.subheader("📊 Feature Importance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Random Forest", "Logistic Regression", "SHAP Global"])
    
    with tab1:
        rf_importance_fig = interpreter.plot_feature_importance('random_forest')
        st.plotly_chart(rf_importance_fig, use_container_width=True)
    
    with tab2:
        lr_importance_fig = interpreter.plot_feature_importance('logistic_regression')
        st.plotly_chart(lr_importance_fig, use_container_width=True)
    
    with tab3:
        with st.spinner("Calculating SHAP values... This may take a moment."):
            shap_fig = interpreter.plot_shap_summary()
            st.plotly_chart(shap_fig, use_container_width=True)
    
    # SHAP Waterfall for sample prediction
    st.subheader("🌊 SHAP Waterfall Plot (Sample Prediction)")
    sample_idx = st.slider("Select sample index", 0, len(X)-1, 0)
    
    with st.spinner("Generating SHAP waterfall plot..."):
        waterfall_fig = interpreter.plot_shap_waterfall(sample_idx)
        st.plotly_chart(waterfall_fig, use_container_width=True)

def prediction_page():
    st.header("🔮 Individual Predictions with Explanations")
    
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model first in the 'Data Upload & Training' page.")
        return
    
    ensemble = st.session_state.ensemble_model
    X = st.session_state.X
    feature_names = st.session_state.feature_names
    
    st.subheader("📝 Enter Feature Values")
    
    # Create input fields for each feature
    input_data = {}
    cols = st.columns(3)
    
    for i, feature in enumerate(feature_names):
        col_idx = i % 3
        with cols[col_idx]:
            # Get feature statistics for reasonable defaults
            feature_mean = X[feature].mean()
            feature_std = X[feature].std()
            feature_min = X[feature].min()
            feature_max = X[feature].max()
            
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=float(feature_min),
                max_value=float(feature_max),
                value=float(feature_mean),
                step=float(feature_std/10)
            )
    
    if st.button("🎯 Make Prediction", type="primary"):
        # Create input dataframe
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = ensemble.predict(input_df)[0]
        prediction_proba = ensemble.predict_proba(input_df)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Prediction Result")
            if prediction == 1:
                st.error(f"**Diabetes Predicted** (Probability: {prediction_proba[1]:.3f})")
            else:
                st.success(f"**No Diabetes** (Probability: {prediction_proba[0]:.3f})")
        
        with col2:
            st.subheader("📊 Prediction Confidence")
            confidence_fig = go.Figure(data=[
                go.Bar(x=['No Diabetes', 'Diabetes'], 
                      y=[prediction_proba[0], prediction_proba[1]],
                      marker_color=['green', 'red'])
            ])
            confidence_fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability"
            )
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        # LIME explanation
        st.subheader("🔍 LIME Explanation")
        interpreter = ModelInterpreter(ensemble, X, feature_names)
        
        with st.spinner("Generating LIME explanation..."):
            lime_fig = interpreter.explain_prediction_lime(input_df.iloc[0])
            st.plotly_chart(lime_fig, use_container_width=True)
            
        st.info("💡 The LIME explanation shows how each feature contributes to this specific prediction. Positive values push towards diabetes prediction, negative values push towards no diabetes.")

if __name__ == "__main__":
    main()