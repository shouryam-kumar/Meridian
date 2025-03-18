import streamlit as st

def render_sidebar():
    """
    Render sidebar with user controls
    
    Returns:
        tuple: Selected coin, timeframe, indicators, prediction model, and days
    """
    st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1E3A8A;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    .sidebar-subtitle {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    
    /* Vertical space between widgets */
    .stSelectbox, .stMultiselect, .stSlider {
        margin-bottom: 1rem;
    }
    
    .coin-icon {
        margin-right: 8px;
        font-size: 1.2rem;
    }
    
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #F1F5F9;
        padding: 1rem;
        font-size: 0.8rem;
        text-align: center;
        color: #64748B;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar header
    st.sidebar.markdown("""
    <div class="sidebar-title">
        <span class="coin-icon">◑</span> Meridian Controls
    </div>
    """, unsafe_allow_html=True)
    
    # Coin selection section
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-subtitle">Cryptocurrency</div>', unsafe_allow_html=True)
    
    coin = st.sidebar.selectbox(
        "Select a cryptocurrency",
        ["bitcoin", "ethereum", "cardano", "solana", "ripple", "dogecoin", "polkadot"],
        index=0,
        format_func=lambda x: x.capitalize()
    )
    
    timeframe = st.sidebar.selectbox(
        "Select timeframe",
        ["30d", "90d", "180d", "1y", "max"],
        index=2,
        format_func=lambda x: {
            "30d": "30 Days", 
            "90d": "90 Days", 
            "180d": "180 Days", 
            "1y": "1 Year", 
            "max": "Maximum Available"
        }.get(x, x)
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Technical Indicators section
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-subtitle">Technical Indicators</div>', unsafe_allow_html=True)
    
    indicators = st.sidebar.multiselect(
        "Select indicators",
        [
            "SMA (Simple Moving Average)", 
            "EMA (Exponential Moving Average)",
            "RSI (Relative Strength Index)",
            "MACD (Moving Average Convergence Divergence)",
            "Bollinger Bands"
        ],
        default=["SMA (Simple Moving Average)", "RSI (Relative Strength Index)"]
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction section
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-subtitle">Price Prediction</div>', unsafe_allow_html=True)
    
    prediction_model = st.sidebar.selectbox(
        "Prediction Model",
        ["Linear Regression", "Random Forest", "Gradient Boosting", "SVR", "Prophet", "Ensemble"],
        index=5  # Default to Ensemble
    )
    
    prediction_days = st.sidebar.slider(
        "Prediction Days",
        min_value=1,
        max_value=30,
        value=7,
        step=1
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add info about the app
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    with st.sidebar.expander("About Meridian"):
        st.markdown("""
        **Meridian** is an advanced cryptocurrency analysis and prediction platform leveraging multiple 
        machine learning models to forecast price movements.
        
        **Features:**
        - Real-time market data
        - Technical indicators
        - Multiple prediction models
        - Ensemble forecasting
        
        For more information, visit our documentation.
        """)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        Meridian v1.0.0 
    </div>
    """, unsafe_allow_html=True)
    
    # Convert indicators to simpler format for processing
    indicator_map = {
        "SMA (Simple Moving Average)": "sma",
        "EMA (Exponential Moving Average)": "ema",
        "RSI (Relative Strength Index)": "rsi",
        "MACD (Moving Average Convergence Divergence)": "macd",
        "Bollinger Bands": "bollinger"
    }
    
    selected_indicators = [indicator_map[ind] for ind in indicators]
    
    return coin, timeframe, selected_indicators, prediction_model, prediction_days