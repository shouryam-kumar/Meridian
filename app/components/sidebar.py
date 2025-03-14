import streamlit as st
from datetime import datetime

# Function to handle selection changes
def handle_coin_change():
    if 'selected_coin_name' in st.session_state and st.session_state.selected_coin_name in coin_options:
        st.session_state.selected_coin_id = coin_options[st.session_state.selected_coin_name]
        st.session_state.data = None  # Force refresh data

def handle_period_change():
    st.session_state.selected_period = time_periods[st.session_state.selected_period_name]
    st.session_state.data = None  # Force refresh data

def handle_indicator_change():
    pass  # We'll recalculate indicators when needed

def handle_auto_refresh():
    st.session_state.data = None  # Force refresh data

# Get available coins - Not using cache due to object hashing issue
def load_coin_data(fetcher):
    try:
        return fetcher.get_available_coins(limit=50)
    except Exception as e:
        print(f"Error loading coins: {str(e)}")
        return None

def render_sidebar(fetcher):
    """Render the sidebar with all controls"""
    
    # Define global variables to be used in the handlers
    global coin_options, time_periods
    
    # Sidebar for input controls
    st.sidebar.header("User Input Parameters")
    
    # Initialize variables
    coin_options = {}
    time_periods = {
        "7 days": 7,
        "30 days": 30,
        "90 days": 90,
        "1 year": 365,
        "Max": "max"
    }
    predict_button = False
    
    with st.sidebar:
        st.markdown("## Data Selection")
        
        # Load coins data - directly call without cache
        with st.spinner("Loading cryptocurrencies..."):
            coins_df = load_coin_data(fetcher)

        if coins_df is not None:
            # Create selection options
            coin_options = {f"{row['name']} ({row['symbol'].upper()})": row['id'] for _, row in coins_df.iterrows()}
            
            # Ensure Bitcoin is in the options
            if 'Bitcoin (BTC)' not in coin_options:
                coin_options['Bitcoin (BTC)'] = 'bitcoin'
            
            # Find default index for Bitcoin
            options_list = list(coin_options.keys())
            default_index = 0
            if 'Bitcoin (BTC)' in options_list:
                default_index = options_list.index('Bitcoin (BTC)')
            elif 'selected_coin_name' in st.session_state and st.session_state.selected_coin_name in options_list:
                default_index = options_list.index(st.session_state.selected_coin_name)
            
            selected_coin_name = st.selectbox(
                "Select cryptocurrency",
                options=options_list,
                index=default_index,
                key="selected_coin_name",
                on_change=handle_coin_change
            )
            
            # Time period selection
            period_options = list(time_periods.keys())
            default_period_index = 2  # 90 days
            if 'selected_period_name' in st.session_state and st.session_state.selected_period_name in period_options:
                default_period_index = period_options.index(st.session_state.selected_period_name)
            
            selected_period_name = st.selectbox(
                "Select time period",
                options=period_options,
                index=default_period_index,
                key="selected_period_name",
                on_change=handle_period_change
            )
            
            # Technical indicators options
            st.markdown("## Technical Indicators")
            if 'show_sma' not in st.session_state:
                st.session_state.show_sma = True
                
            show_sma = st.checkbox("Moving Averages", value=st.session_state.show_sma, key="show_sma", on_change=handle_indicator_change)
            show_rsi = st.checkbox("RSI", value=st.session_state.show_rsi, key="show_rsi", on_change=handle_indicator_change)
            show_macd = st.checkbox("MACD", value=st.session_state.show_macd, key="show_macd", on_change=handle_indicator_change)
            show_bollinger = st.checkbox("Bollinger Bands", value=st.session_state.show_bollinger, key="show_bollinger", on_change=handle_indicator_change)
            
            # Prediction options
            st.markdown("## Prediction Models")
            prediction_model_options = ["None", "Linear Regression", "ARIMA", "Random Forest"]
            current_model_index = 0
            if 'prediction_model' in st.session_state and st.session_state.prediction_model in prediction_model_options:
                current_model_index = prediction_model_options.index(st.session_state.prediction_model)
            
            prediction_model = st.radio(
                "Select prediction model",
                options=prediction_model_options,
                index=current_model_index,
                key="prediction_model"
            )
            
            prediction_days = st.slider(
                "Prediction horizon (days)",
                min_value=1,
                max_value=30,
                value=st.session_state.prediction_days,
                key="prediction_days"
            )
            
            # Refresh data button
            if st.button("🔄 Refresh Data", on_click=handle_auto_refresh):
                pass
            
            # Add auto-refresh info
            if 'last_update_time' in st.session_state and st.session_state.last_update_time:
                st.caption(f"Last updated: {st.session_state.last_update_time.strftime('%H:%M:%S')}")
                minutes_ago = (datetime.now() - st.session_state.last_update_time).total_seconds() / 60
                st.caption(f"({int(minutes_ago)} minutes ago)")
            
            # Action buttons
            predict_button = st.button("Generate Predictions", disabled=(st.session_state.prediction_model == "None"))
            
    return coin_options, time_periods, predict_button