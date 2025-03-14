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
        with st.spinner("Training prediction model..."):
            predictor = PredictionModels()
            predictions = None
            model_info = ""
            
            # Copy the data and fill missing values
            data_for_prediction = data_with_indicators.copy()
            
            # Check for missing values
            missing_values = data_for_prediction.isnull().sum().sum()
            if missing_values > 0:
                st.info(f"Preprocessing: Filling {missing_values} missing values for prediction...")
                data_for_prediction = data_for_prediction.fillna(method='ffill').fillna(method='bfill')
            
            # Proceed with model training based on selection
            if prediction_model == "Linear Regression":
                predictions, model_info = train_linear_regression(predictor, data_for_prediction, prediction_days)
            elif prediction_model == "ARIMA":
                predictions, model_info = train_arima(predictor, data_for_prediction, prediction_days)
            elif prediction_model == "Random Forest":
                predictions, model_info = train_random_forest(predictor, data_for_prediction, prediction_days)
            
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
        # Prepare a subset of features that work well together
        features = ['price', 'volume', 'daily_return']
        # Add technical indicators if available
        for feature in ['sma_7', 'sma_25', 'rsi', 'macd', 'bb_middle']:
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

def train_arima(predictor, data, prediction_days):
    """Train ARIMA model and get predictions"""
    
    model_info = """
    **ARIMA Model (AutoRegressive Integrated Moving Average)**: 
    This time series model analyzes historical price patterns to forecast future movements.
    It works by finding patterns in the data's own lags and using them for prediction.
    """
    
    # Add a progress message
    temp_message = st.empty()
    temp_message.info("Training ARIMA model...")
    
    try:
        # For ARIMA, just use the price column with a reasonable timeframe
        # Use just enough data for ARIMA to be effective (too much can cause performance issues)
        recent_data = data.tail(min(90, len(data)))
        price_data = pd.DataFrame({'price': recent_data['price']})
        
        model, _, _ = predictor.train_arima(price_data, order=(1,1,0))
        predictions = predictor.predict_multiple_days('arima', price_data, days=prediction_days)
        
        # Clear the progress message
        temp_message.empty()
        return predictions, model_info
    
    except Exception as model_error:
        temp_message.empty()
        st.error(f"Error in ARIMA model: {str(model_error)}")
        st.warning("Falling back to simplified prediction...")
        
        # Simple moving average forecast
        last_prices = data['price'].tail(7)
        base_price = last_prices.mean()
        dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
        predictions = pd.Series([base_price] * prediction_days, index=dates)
        
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
        # Prepare a simpler feature set (the complex one was causing errors)
        features = ['price', 'volume', 'daily_return']
        # Add only a few technical indicators to avoid overfitting
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