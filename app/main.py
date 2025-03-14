import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.data.fetcher import CryptoDataFetcher
from src.models.indicators import TechnicalIndicators
from app.components.sidebar import render_sidebar
from app.components.charts import render_price_chart
from app.components.indicators import render_indicators
from app.components.predictions import render_predictions

# Set page config
st.set_page_config(
    page_title="Crypto Market Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize the data fetcher
@st.cache_resource
def get_data_fetcher():
    return CryptoDataFetcher()

fetcher = get_data_fetcher()

# App title and description
st.title("Cryptocurrency Market Analyzer & Predictor")
st.markdown("""
This app analyzes cryptocurrency market trends using technical indicators and provides price predictions using various models.
""")

# Initialize session state for persistence
if 'selected_coin_id' not in st.session_state:
    st.session_state.selected_coin_id = 'bitcoin'  # Bitcoin as default
if 'selected_coin_name' not in st.session_state:
    st.session_state.selected_coin_name = 'Bitcoin (BTC)'  # Default display name
if 'selected_period' not in st.session_state:
    st.session_state.selected_period = 90  # Default to 90 days
if 'selected_period_name' not in st.session_state:
    st.session_state.selected_period_name = "90 days"
if 'show_sma' not in st.session_state:
    st.session_state.show_sma = True
if 'show_rsi' not in st.session_state:
    st.session_state.show_rsi = True
if 'show_macd' not in st.session_state:
    st.session_state.show_macd = True
if 'show_bollinger' not in st.session_state:
    st.session_state.show_bollinger = True
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = "None"
if 'prediction_days' not in st.session_state:
    st.session_state.prediction_days = 7
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_with_indicators' not in st.session_state:
    st.session_state.data_with_indicators = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None

# Check if data needs to be refreshed based on time
def should_refresh_data():
    if st.session_state.last_update_time is None:
        return True
    
    # Refresh data every 5 minutes
    time_since_update = datetime.now() - st.session_state.last_update_time
    return time_since_update > timedelta(minutes=5)

# Fetch historical data - using cache with a direct function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_historical_data(coin_id, days):
    try:
        return fetcher.fetch_historical_data(coin_id, days)
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

# Render sidebar
coin_options, time_periods, predict_button = render_sidebar(fetcher)

# Initialize containers for different sections
status_container = st.container()
data_container = st.container()
chart_container = st.container()
indicators_container = st.container()
prediction_container = st.container()

# Main content area
if coin_options:
    # Auto-load data if needed
    if (st.session_state.selected_coin_id is not None and 
        (st.session_state.data is None or should_refresh_data())):
        
        with status_container:
            with st.spinner(f"Fetching data for {st.session_state.selected_coin_name}..."):
                data = get_historical_data(
                    st.session_state.selected_coin_id, 
                    st.session_state.selected_period
                )
                if data is not None:
                    st.session_state.data = data
                    st.session_state.last_update_time = datetime.now()
                    
                    # Ensure daily_return is present
                    if 'daily_return' not in st.session_state.data.columns:
                        st.session_state.data['daily_return'] = st.session_state.data['price'].pct_change() * 100
                    
                    # Add technical indicators
                    try:
                        st.session_state.data_with_indicators = TechnicalIndicators.add_all_indicators(st.session_state.data)
                    except Exception as e:
                        print(f"Error calculating indicators: {str(e)}")
                        st.session_state.data_with_indicators = st.session_state.data.copy()
                else:
                    if st.session_state.data is None:  # Only show if no previous data
                        st.error("Error fetching data from CoinGecko. Please try again later.")
    
    # Display analysis if data is available
    if st.session_state.data is not None:
        with data_container:
            st.subheader(f"{st.session_state.selected_coin_name} Analysis")
            
            try:
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                current_price = st.session_state.data['price'].iloc[-1]
                price_change = st.session_state.data['price'].iloc[-1] - st.session_state.data['price'].iloc[0]
                price_change_pct = (price_change / st.session_state.data['price'].iloc[0]) * 100
                
                col1.metric(
                    "Current Price", 
                    f"${current_price:.2f}",
                    f"{price_change_pct:.2f}% in {st.session_state.selected_period_name}"
                )
                
                col2.metric(
                    "24h Volume",
                    f"${st.session_state.data['volume'].iloc[-1]/1000000:.2f}M"
                )
                
                col3.metric(
                    "Market Cap",
                    f"${st.session_state.data['market_cap'].iloc[-1]/1000000000:.2f}B"
                )
                
                col4.metric(
                    "Volatility (Std Dev)",
                    f"{st.session_state.data['daily_return'].std():.2f}%"
                )
            except Exception as e:
                print(f"Error displaying metrics: {str(e)}")
                st.error("Error displaying metrics. The data may be incomplete.")
        
        # Render price chart
        with chart_container:
            render_price_chart(st.session_state.data, st.session_state.data_with_indicators, 
                              st.session_state.show_sma, st.session_state.show_bollinger)
        
        # Render indicator tabs
        if st.session_state.data_with_indicators is not None:
            with indicators_container:
                render_indicators(st.session_state.data_with_indicators, 
                                 st.session_state.show_rsi, st.session_state.show_macd)
        
        # Render predictions
        if predict_button and st.session_state.prediction_model != "None":
            with prediction_container:
                render_predictions(st.session_state.data, st.session_state.data_with_indicators, 
                                  st.session_state.prediction_model, st.session_state.prediction_days)
        
        # Raw data display option
        with st.expander("Show Raw Data"):
            # Only show the last 100 rows to prevent rendering issues
            st.dataframe(st.session_state.data.tail(100), use_container_width=True)
        
        # Download option
        csv_data = st.session_state.data.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{st.session_state.selected_coin_id}_{st.session_state.selected_period_name.replace(' ', '_')}_data.csv",
            mime="text/csv"
        )
    else:
        # If no data is available yet, show a loading message
        st.info("Loading cryptocurrency data. Please wait...")

# Add footer
st.markdown("---")
st.markdown("Data provided by CoinGecko API | Built with Streamlit")

# Add auto-refresh script
st.markdown("""
<script>
    // Auto refresh the page every 5 minutes (300000 milliseconds)
    setTimeout(function() {
        window.location.reload();
    }, 300000);
</script>
""", unsafe_allow_html=True)