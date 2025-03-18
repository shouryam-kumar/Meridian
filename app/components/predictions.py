import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import traceback
from datetime import timedelta
from src.models.prediction import PredictionModels

def render_predictions(data, data_with_indicators, prediction_model, prediction_days):
    """Render predictions section"""
    
    st.subheader("Price Predictions")
    
    try:
        # Ensure data_with_indicators is not None
        if data_with_indicators is None:
            st.warning("No indicator data available. Using base price data for predictions.")
            data_with_indicators = data.copy()
            
        with st.spinner("Training prediction model..."):
            predictor = PredictionModels()
            predictions = None
            model_info = ""
            
            # Copy the data and fill missing values with modern methods
            data_for_prediction = data_with_indicators.copy()
            
            # Check for missing values
            missing_values = data_for_prediction.isnull().sum().sum()
            if missing_values > 0:
                st.info(f"Preprocessing: Filling {missing_values} missing values for prediction...")
                data_for_prediction = data_for_prediction.ffill().bfill()
            
            # Proceed with model training based on selection
            if prediction_model == "Linear Regression":
                predictions, model_info = train_linear_regression(predictor, data_for_prediction, prediction_days)
            elif prediction_model == "Random Forest":
                predictions, model_info = train_random_forest(predictor, data_for_prediction, prediction_days)
            elif prediction_model == "Gradient Boosting":
                predictions, model_info = train_gradient_boosting(predictor, data_for_prediction, prediction_days)
            elif prediction_model == "SVR":
                predictions, model_info = train_svr(predictor, data_for_prediction, prediction_days)
            elif prediction_model == "Prophet":
                predictions, model_info = train_prophet(predictor, data_for_prediction, prediction_days)
            elif prediction_model == "Ensemble":
                predictions, model_info = train_ensemble(predictor, data_for_prediction, prediction_days)
            
            if predictions is not None and len(predictions) > 0:
                # Create and display prediction chart
                display_prediction_chart(data, predictions, prediction_model, prediction_days)
                
                # Display model information
                if model_info:
                    st.info(model_info)
                
                # Display prediction table and insights
                display_prediction_insights(data, predictions, prediction_days)
            else:
                st.warning("No predictions were generated. The model may need more data or different parameters.")
    except Exception as e:
        st.error(f"Error in prediction process: {str(e)}")
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())

def train_linear_regression(predictor, data, prediction_days):
    """Train Linear Regression model and get predictions"""
    
    model_info = """
    **Linear Regression Model**: 
    This model predicts future prices based on a linear relationship with various features 
    including price history, volume, and technical indicators.
    """
    
    # Add a progress message
    temp_message = st.empty()
    temp_message.info("Training Linear Regression model...")
    
    try:
        # Use a consistent set of base features across all models
        features = ['price', 'volume', 'daily_return']
        # Add only the most reliable technical indicators
        for feature in ['sma_7', 'rsi']:
            if feature in data.columns:
                features.append(feature)
        
        available_features = [f for f in features if f in data.columns]
        
        model, _, _, _, _ = predictor.train_linear_regression(
            data, 
            features=available_features
        )
        predictions = predictor.predict_multiple_days('linear_regression', data, days=prediction_days)
        
        # Clear the progress message
        temp_message.empty()
        return predictions, model_info
    
    except Exception as model_error:
        temp_message.empty()
        st.error(f"Error in Linear Regression model: {str(model_error)}")
        st.warning("Falling back to simplified prediction...")
        
        # Create a very simple prediction (linear trend based on last few days)
        last_prices = data['price'].tail(7)
        if len(last_prices) >= 2:
            slope = (last_prices.iloc[-1] - last_prices.iloc[0]) / (len(last_prices) - 1)
            base_price = last_prices.iloc[-1]
            dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
            predictions = pd.Series([base_price + slope * (i+1) for i in range(prediction_days)], index=dates)
        else:
            # Flat prediction if not enough data
            dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
            predictions = pd.Series([data['price'].iloc[-1]] * prediction_days, index=dates)
        
        return predictions, model_info

def train_random_forest(predictor, data, prediction_days):
    """Train Random Forest model and get predictions"""
    
    model_info = """
    **Random Forest Model**: 
    This ensemble learning method builds multiple decision trees and merges their predictions.
    It can capture complex non-linear relationships in the data and handle noisy inputs.
    """
    
    # Add a progress message
    temp_message = st.empty()
    temp_message.info("Training Random Forest model...")
    
    try:
        # Use the same feature set as Linear Regression for consistency
        features = ['price', 'volume', 'daily_return']
        # Add only the most reliable technical indicators
        for feature in ['sma_7', 'rsi']:
            if feature in data.columns:
                features.append(feature)
                
        available_features = [f for f in features if f in data.columns]
        
        model, _, _, _, _ = predictor.train_random_forest(
            data,
            features=available_features
        )
        predictions = predictor.predict_multiple_days('random_forest', data, days=prediction_days)
        
        # Clear the progress message
        temp_message.empty()
        return predictions, model_info
        
    except Exception as model_error:
        temp_message.empty()
        st.error(f"Error in Random Forest model: {str(model_error)}")
        st.warning("Falling back to simplified prediction...")
        
        # Create a simple prediction with slight upward bias (assuming crypto optimism)
        base_price = data['price'].iloc[-1]
        dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
        # Add a small random walk with slight upward bias
        pred_list = [base_price]
        for i in range(1, prediction_days):
            next_price = pred_list[-1] * (1 + np.random.normal(0.002, 0.01))  # Slight upward bias
            pred_list.append(next_price)
        predictions = pd.Series(pred_list, index=dates)
        
        return predictions, model_info

def train_gradient_boosting(predictor, data, prediction_days):
    """Train Gradient Boosting model and get predictions"""
    
    model_info = """
    **Gradient Boosting Model**: 
    This advanced ensemble technique builds decision trees sequentially, with each tree correcting 
    the errors of its predecessors. It excels at capturing complex market patterns and is less prone 
    to overfitting than other models.
    """
    
    # Add a progress message
    temp_message = st.empty()
    temp_message.info("Training Gradient Boosting model...")
    
    try:
        # Use the same feature set as other models for consistency
        features = ['price', 'volume', 'daily_return']
        # Add only the most reliable technical indicators
        for feature in ['sma_7', 'rsi', 'macd']:
            if feature in data.columns:
                features.append(feature)
                
        available_features = [f for f in features if f in data.columns]
        
        model, _, _, _, _ = predictor.train_gradient_boosting(
            data,
            features=available_features
        )
        predictions = predictor.predict_multiple_days('gradient_boosting', data, days=prediction_days)
        
        # Clear the progress message
        temp_message.empty()
        return predictions, model_info
        
    except Exception as model_error:
        temp_message.empty()
        st.error(f"Error in Gradient Boosting model: {str(model_error)}")
        st.warning("Falling back to simplified prediction...")
        
        # Create a simple prediction with slight upward bias
        base_price = data['price'].iloc[-1]
        dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
        # Add a smooth trend with a small random component
        pred_list = [base_price]
        for i in range(1, prediction_days):
            next_price = pred_list[-1] * (1 + np.random.normal(0.001, 0.008))  # Slightly lower volatility than RF
            pred_list.append(next_price)
        predictions = pd.Series(pred_list, index=dates)
        
        return predictions, model_info

def train_svr(predictor, data, prediction_days):
    """Train Support Vector Regression model and get predictions"""
    
    model_info = """
    **Support Vector Regression (SVR)**: 
    This machine learning technique excels at finding complex non-linear patterns in price data.
    SVR is particularly effective at identifying support and resistance levels in crypto markets
    and tends to be resilient to outliers and noise.
    """
    
    # Add a progress message
    temp_message = st.empty()
    temp_message.info("Training SVR model...")
    
    try:
        # Use a smaller feature set to prevent overfitting
        features = ['price', 'volume']
        # Add just 1-2 technical indicators
        for feature in ['sma_7', 'rsi']:
            if feature in data.columns:
                features.append(feature)
                
        available_features = [f for f in features if f in data.columns]
        
        model, _, _, _, _ = predictor.train_svr(
            data,
            features=available_features
        )
        predictions = predictor.predict_multiple_days('svr', data, days=prediction_days)
        
        # Clear the progress message
        temp_message.empty()
        return predictions, model_info
        
    except Exception as model_error:
        temp_message.empty()
        st.error(f"Error in SVR model: {str(model_error)}")
        st.warning("Falling back to simplified prediction...")
        
        # Create a simple prediction that focuses on recent trend
        recent_data = data['price'].tail(14)  # Look at last 14 days
        trend = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1  # Overall trend
        
        # Convert to daily rate (dampened)
        daily_trend = (1 + trend) ** (1/14) - 1
        daily_trend = daily_trend * 0.7  # Dampen the trend
        
        # Create predictions
        base_price = data['price'].iloc[-1]
        dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
        
        pred_list = [base_price]
        for i in range(1, prediction_days):
            next_price = pred_list[-1] * (1 + daily_trend + np.random.normal(0, 0.005))
            pred_list.append(next_price)
            
        predictions = pd.Series(pred_list, index=dates)
        
        return predictions, model_info

def train_prophet(predictor, data, prediction_days):
    """Train Facebook Prophet model and get predictions"""
    
    model_info = """
    **Prophet Model**: 
    Developed by Facebook, Prophet is specialized for time series forecasting with seasonal patterns.
    It automatically detects trends, weekly/monthly patterns, and can handle outliers effectively.
    Particularly useful for cryptocurrencies which often exhibit cyclical behavior.
    """
    
    # Add a progress message
    temp_message = st.empty()
    temp_message.info("Training Prophet model...")
    
    try:
        # Prophet requires specific column names: 'ds' for dates and 'y' for target
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data['price']
        })
        
        model, _ = predictor.train_prophet(prophet_data)
        predictions = predictor.predict_multiple_days('prophet', data, days=prediction_days)
        
        # Clear the progress message
        temp_message.empty()
        
        # Add confidence information from the model if available
        if hasattr(predictor, 'prophet_uncertainty'):
            uncertainty = predictor.prophet_uncertainty
            model_info += f"\n\nUncertainty Range: ±{uncertainty:.2f}%"
        
        return predictions, model_info
        
    except Exception as model_error:
        temp_message.empty()
        st.error(f"Error in Prophet model: {str(model_error)}")
        st.warning("Falling back to simplified prediction...")
        
        # Create a simple prediction
        base_price = data['price'].iloc[-1]
        dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
        
        # Add cyclical component (simplified weekly pattern)
        pred_list = []
        for i in range(prediction_days):
            # Add slight weekly pattern (higher on weekends)
            day_of_week = dates[i].weekday()
            weekend_effect = 0.002 if day_of_week >= 5 else 0.0  # Slight boost on weekends
            
            if i == 0:
                next_price = base_price * (1 + weekend_effect + np.random.normal(0.001, 0.005))
            else:
                next_price = pred_list[-1] * (1 + weekend_effect + np.random.normal(0.001, 0.005))
            
            pred_list.append(next_price)
            
        predictions = pd.Series(pred_list, index=dates)
        
        return predictions, model_info

def train_ensemble(predictor, data, prediction_days):
    """Train an Ensemble model combining multiple prediction models"""
    
    model_info = """
    **Ensemble Model**: 
    This model combines predictions from multiple algorithms (Linear Regression, Random Forest, 
    and Gradient Boosting) to create a more robust forecast. Ensemble methods often outperform 
    individual models by balancing their strengths and weaknesses.
    """
    
    # Add a progress message
    temp_message = st.empty()
    temp_message.info("Training Ensemble model...")
    
    try:
        # Train individual models first
        lr_predictions, _ = train_linear_regression(predictor, data, prediction_days)
        rf_predictions, _ = train_random_forest(predictor, data, prediction_days)
        gb_predictions, _ = train_gradient_boosting(predictor, data, prediction_days)
        
        # Combine predictions (simple average)
        ensemble_predictions = pd.DataFrame({
            'lr': lr_predictions.values,
            'rf': rf_predictions.values,
            'gb': gb_predictions.values
        }, index=lr_predictions.index)
        
        # Take the average (or weighted average)
        # You could adjust these weights based on past performance
        weights = {
            'lr': 0.3,  # Linear Regression - good for trends
            'rf': 0.35, # Random Forest - good for stability
            'gb': 0.35  # Gradient Boosting - good for accuracy
        }
        
        weighted_predictions = (
            ensemble_predictions['lr'] * weights['lr'] +
            ensemble_predictions['rf'] * weights['rf'] +
            ensemble_predictions['gb'] * weights['gb']
        )
        
        # Clear the progress message
        temp_message.empty()
        
        # Add confidence information
        agreement_level = 1 - (ensemble_predictions.std(axis=1) / ensemble_predictions.mean(axis=1)).mean()
        model_info += f"\n\nModel Agreement Level: {agreement_level:.2f}/1.0 (higher is better)"
        
        return weighted_predictions, model_info
        
    except Exception as model_error:
        temp_message.empty()
        st.error(f"Error in Ensemble model: {str(model_error)}")
        st.warning("Falling back to Linear Regression model...")
        
        # Fallback to linear regression if ensemble fails
        return train_linear_regression(predictor, data, prediction_days)

def display_prediction_chart(data, predictions, prediction_model, prediction_days):
    """Display prediction chart with confidence intervals"""
    
    # Create prediction chart
    pred_fig = go.Figure()
    
    # Add historical data
    pred_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['price'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Add prediction
    pred_fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions.values,
            mode='lines+markers',
            name=f'{prediction_model} Prediction',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        )
    )
    
    # Add confidence intervals (simple approach - 5% above and below)
    upper_bound = predictions.values * 1.05
    lower_bound = predictions.values * 0.95
    
    pred_fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=upper_bound,
            mode='lines',
            name='Upper Bound (5%)',
            line=dict(color='rgba(255, 127, 14, 0.3)', width=1, dash='dot'),
            showlegend=False
        )
    )
    
    pred_fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=lower_bound,
            mode='lines',
            name='Lower Bound (5%)',
            line=dict(color='rgba(255, 127, 14, 0.3)', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.1)',
            showlegend=False
        )
    )
    
    # Update layout
    pred_fig.update_layout(
        height=400,
        title=f"{prediction_model} - {prediction_days} Day Price Prediction",
        yaxis=dict(title='Price (USD)'),
        template='plotly_white'
    )
    
    st.plotly_chart(pred_fig, use_container_width=True)

def display_prediction_insights(data, predictions, prediction_days):
    """Display prediction table and insights"""
    
    # Display prediction values
    st.subheader("Predicted Prices")
    
    # Calculate various metrics
    current_price = data['price'].iloc[-1]
    
    # Calculate min/max prediction
    min_prediction = predictions.min()
    max_prediction = predictions.max()
    predicted_volatility = predictions.std() / predictions.mean() * 100
    
    # Show prediction summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Current Price", 
        f"${current_price:.2f}"
    )
    col2.metric(
        "Predicted Max", 
        f"${max_prediction:.2f}",
        f"{((max_prediction/current_price)-1)*100:.2f}%"
    )
    col3.metric(
        "Predicted Min", 
        f"${min_prediction:.2f}",
        f"{((min_prediction/current_price)-1)*100:.2f}%"
    )
    
    # Format the table
    def format_price_change(price, base_price):
        change = ((price / base_price) - 1) * 100
        color = "green" if change >= 0 else "red"
        return f"<span style='color:{color}'>{change:.2f}%</span>"
    
    # Create DataFrame for display
    pred_df = pd.DataFrame({
        'Date': predictions.index.strftime('%Y-%m-%d'),
        'Predicted Price': [f"${price:.2f}" for price in predictions.values],
        'Change from Current': [format_price_change(price, current_price) for price in predictions.values]
    })
    
    # Display as a styled HTML table
    st.markdown(
        pred_df.style.hide(axis="index")
        .format({'Date': '{}', 'Predicted Price': '{}'})
        .to_html(escape=False), 
        unsafe_allow_html=True
    )
    
    # Add a smart insight based on the prediction
    avg_change = ((predictions.iloc[-1] / current_price) - 1) * 100
    if avg_change > 5:
        insight = f"The model predicts a positive trend of {avg_change:.2f}% over the next {prediction_days} days."
    elif avg_change < -5:
        insight = f"The model predicts a negative trend of {avg_change:.2f}% over the next {prediction_days} days."
    else:
        insight = f"The model predicts relatively stable prices with a change of {avg_change:.2f}% over the next {prediction_days} days."
    
    st.info(f"**Price Prediction Insight**: {insight}")
    
    st.markdown("""
    **Disclaimer:** These predictions are based on historical patterns and technical analysis.
    Cryptocurrency markets are highly volatile and unpredictable. Past performance is not indicative of future results.
    Do not use these predictions as the sole basis for investment decisions.
    """)