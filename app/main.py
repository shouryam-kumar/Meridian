import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import your components
from components.sidebar import render_sidebar
from components.charts import render_price_chart, render_volume_chart
from components.indicators import render_indicators
from components.predictions import render_predictions
from utils.branding import (
    apply_meridian_branding, 
    render_meridian_header, 
    render_metrics_row,
    render_price_header,
    render_disclaimer,
    render_footer
)

# Import data fetcher
from src.data.fetcher import CryptoDataFetcher

# Configure the page
st.set_page_config(
    page_title="Meridian | Crypto Analysis & Prediction",
    page_icon="◑",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Apply Meridian branding first
    apply_meridian_branding()
    
    # Render sidebar and get user selections
    coin, timeframe, indicators, prediction_model, prediction_days = render_sidebar()
    
    # Create main container for all content
    main_container = st.container()
    
    with main_container:
        # Render the header
        render_meridian_header()
        
        # Display loading message
        with st.spinner(f'Fetching historical data for {coin}...'):
            try:
                # Fetch data
                fetcher = CryptoDataFetcher()
                data = fetcher.fetch_historical_data(coin, timeframe)
                
                if data is None or len(data) == 0:
                    st.error(f"No data available for {coin}. Please try another cryptocurrency.")
                    return
                    
                # Convert to DataFrame if needed
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)
                    
                # Ensure data is sorted by date
                data = data.sort_index()
                
            except Exception as e:
                error_message = str(e)
            if "429" in error_message or "rate limit" in error_message.lower():
                st.error("⚠️ API rate limit exceeded under free tier. Please wait 5-10 minutes before trying again.")
            else:
                st.error(f"Error fetching data: {error_message}")
                return
        
        # Calculate metrics for display
        current_price = data['price'].iloc[-1]
        price_change_24h = ((current_price / data['price'].iloc[-2]) - 1) * 100
        price_change_7d = ((current_price / data['price'].iloc[-7]) - 1) * 100 if len(data) >= 7 else 0
        volume_24h = data['volume'].iloc[-1]
        
        # Display price header
        render_price_header(coin, current_price, price_change_24h)
        
        # Display metrics in a branded row
        metrics = [
            {
                'title': 'Current Price',
                'value': f"${current_price:.2f}",
                'change': f"{price_change_24h:.2f}%"
            },
            {
                'title': '7-Day Change',
                'value': f"{price_change_7d:.2f}%",
                'color': '#10B981' if price_change_7d >= 0 else '#EF4444'
            },
            {
                'title': '24h Volume',
                'value': f"${volume_24h/1000000:.1f}M"
            },
            {
                'title': 'Period',
                'value': f"{timeframe}"
            }
        ]
        render_metrics_row(metrics)
        
        # Add section heading for charts
        st.markdown('<h2 style="color: #1E3A8A; margin-top: 2rem;">Market Data</h2>', unsafe_allow_html=True)
        
        # Create two columns for price and volume charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Render price chart in a branded container
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 1.1rem; font-weight: 600; color: #1E3A8A; margin-bottom: 1rem;">Price History ({coin.upper()})</div>', unsafe_allow_html=True)
            render_price_chart(data)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Render volume chart in a branded container
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.1rem; font-weight: 600; color: #1E3A8A; margin-bottom: 1rem;">Trading Volume</div>', unsafe_allow_html=True)
            render_volume_chart(data)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Apply technical indicators if selected
        data_with_indicators = data.copy()  # Default, will be replaced if indicators are selected
        if indicators:
            st.markdown('<h2 style="color: #1E3A8A; margin-top: 2rem;">Technical Indicators</h2>', unsafe_allow_html=True)
            # Render indicators section with Meridian styling
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            data_with_indicators = render_indicators(data, indicators)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Render predictions section
        if data_with_indicators is None:  # Fallback if indicators function had an error
            data_with_indicators = data.copy()
            
        st.markdown('<h2 style="color: #1E3A8A; margin-top: 2rem;">Price Predictions</h2>', unsafe_allow_html=True)
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        
        # Add model badge if needed
        st.markdown(f'<div style="font-size: 0.9rem; background-color: #DBEAFE; color: #1E3A8A; display: inline-block; padding: 0.3rem 0.6rem; border-radius: 4px; margin-bottom: 1rem;">Model: {prediction_model}</div>', unsafe_allow_html=True)
        
        # Render predictions
        render_predictions(data, data_with_indicators, prediction_model, prediction_days)
        
        # Add disclaimer
        st.markdown("""
        <div style="background-color: #FEFCE8; border-left: 4px solid #EAB308; padding: 1rem; border-radius: 4px; font-size: 0.9rem; color: #854D0E; margin-top: 1rem;">
            <strong>Disclaimer:</strong> These predictions are based on historical patterns and technical analysis.
            Cryptocurrency markets are highly volatile and unpredictable. Past performance is not indicative of future results.
            Do not use these predictions as the sole basis for investment decisions.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some space before the footer
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        
        # Add the footer at the end
        render_footer()

if __name__ == "__main__":
    main()