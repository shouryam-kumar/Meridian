import streamlit as st

def apply_meridian_branding():
    """
    Apply Meridian branding styles to the Streamlit app with improved readability
    """
    # Custom CSS for Meridian branding
    st.markdown("""
    <style>
    /* Main brand colors - refined palette */
    :root {
        --meridian-primary: #1E3A8A;
        --meridian-secondary: #2563EB;
        --meridian-accent: #3B82F6;
        --meridian-light: #DBEAFE;
        --meridian-dark: #1E40AF;
        --meridian-text: #1E293B;
        --meridian-background: #F8FAFC;
        --meridian-card: #FFFFFF;
        --meridian-success: #10B981;
        --meridian-warning: #F59E0B;
        --meridian-danger: #EF4444;
    }
    
    /* Global styles */
    .reportview-container {
        background: linear-gradient(120deg, #f5f7fa 0%, #e4ecfb 100%);
    }
    
    .main .block-container {
        background-color: #F8FAFC;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        max-width: 1200px;
        margin: 0 auto;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--meridian-primary);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Header */
    .meridian-header {
        background: linear-gradient(90deg, var(--meridian-primary) 0%, var(--meridian-secondary) 100%);
        padding: 1.5rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .meridian-logo {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
    }
    
    .meridian-logo-icon {
        margin-right: 10px;
        font-size: 1.8rem;
    }
    
    .meridian-tagline {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.2rem;
    }
    
    /* Cards */
    .meridian-card {
        background: var(--meridian-card);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--meridian-accent);
    }
    
    .meridian-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--meridian-primary);
    }
    
    /* Metrics */
    .meridian-metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .meridian-metric {
        background: var(--meridian-card);
        border-radius: 8px;
        padding: 1rem;
        flex: 1;
        min-width: 120px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-top: 3px solid var(--meridian-accent);
        text-align: center;
    }
    
    .meridian-metric-title {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.3rem;
    }
    
    .meridian-metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--meridian-primary);
    }
    
    .meridian-metric-change {
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }
    
    .positive-change {
        color: var(--meridian-success);
    }
    
    .negative-change {
        color: var(--meridian-danger);
    }
    
    /* Style metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
    }

    [data-testid="stMetricDelta"] > div {
        font-size: 0.9rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #475569 !important;
    }
    
    /* Tables */
    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-bottom: 1rem;
        overflow: hidden;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    table th {
        background-color: #2563EB;
        color: white;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
    }

    table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #E2E8F0;
        background-color: white;
    }

    table tr:last-child td {
        border-bottom: none;
    }

    table tr:hover td {
        background-color: #F1F5F9;
    }
    
    /* Prediction table styling */
    .dataframe {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-bottom: 1rem;
        overflow: hidden;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .dataframe th {
        background-color: #2563EB !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
        text-align: left !important;
        font-weight: 600 !important;
    }

    .dataframe td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid #E2E8F0 !important;
        background-color: white !important;
        color: #1E293B !important;  /* Dark text color for visibility */
    }

    .dataframe tr:last-child td {
        border-bottom: none !important;
    }

    .dataframe tr:hover td {
        background-color: #F1F5F9 !important;
    }

    /* Make sure all table text is visible */
    table, th, td, tr {
        color: #1E293B !important;
    }
    
    /* Sidebar - Improved for better readability with dark theme */
    .css-1d391kg, .css-12oz5g7, div[data-testid="stSidebar"] {
        background-color: #111827 !important;  /* Dark blue/black background */
    }
    
    .sidebar .sidebar-content {
        background: #111827 !important;
    }
    
    /* Make sidebar text readable with high contrast */
    .sidebar .stRadio label, .sidebar .stCheckbox label,
    .sidebar .stSelectbox label, .sidebar .stMultiselect label {
        color: #93C5FD !important;  /* Light blue for better visibility */
        font-weight: 500 !important;
    }
    
    /* Fix specifically for slider label */
    .sidebar .stSlider label, .sidebar .stSlider div[data-baseweb="caption"] {
        color: #93C5FD !important;  /* Light blue for slider labels */
        font-weight: 500 !important;
    }
    
    /* Extra selector to catch all text in sidebar */
    .sidebar p, .sidebar .stMarkdown, .sidebar caption, 
    .sidebar div[role="slider"] span, .sidebar div[data-testid="stSlider"] label {
        color: #93C5FD !important;
    }
    
    /* Fix for number input */
    .sidebar div[data-testid="stNumberInput"] label {
        color: #93C5FD !important;
    }
    
    /* Sidebar section headings */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #60A5FA !important;  /* Light blue for better contrast */
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #4B5563;
    }
    
    .sidebar-subtitle {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #93C5FD !important;  /* Light blue for better visibility */
    }
    
    /* Sidebar widget styling - improved contrast */
    .sidebar [data-testid="stSelectbox"] {
        margin-bottom: 1.2rem;
    }
    
    .sidebar [data-testid="stSelectbox"] > div > div {
        background-color: #1E293B !important;  /* Slightly lighter than sidebar background */
        border: 1px solid #4B5563 !important;
        color: #E5E7EB !important;  /* Light gray text */
        border-radius: 6px !important;
    }
    
    .sidebar [data-testid="stMultiselect"] {
        margin-bottom: 1.2rem;
    }
    
    .sidebar [data-testid="stMultiselect"] > div > div {
        background-color: #1E293B !important;
        border: 1px solid #4B5563 !important;
        color: #E5E7EB !important;
        border-radius: 6px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--meridian-secondary);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: var(--meridian-primary);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border-top: 4px solid #3B82F6;
    }
    
    .chart-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--meridian-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Predictions section */
    .prediction-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border-top: 4px solid #8B5CF6;
    }
    
    .prediction-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .prediction-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--meridian-primary);
    }
    
    /* Model labels */
    .model-label {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        background-color: var(--meridian-light);
        color: var(--meridian-primary);
        margin-right: 0.5rem;
    }
    
    /* Style expanders */
    .streamlit-expanderHeader {
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background-color: #FEFCE8;
        border-left: 4px solid #EAB308;
        padding: 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #854D0E;
        margin-top: 1rem;
    }
    
    /* Footer styling */
    .meridian-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border-radius: 12px;
    }
    
    /* Fix for streamlit slider label colors */
    .stSlider label, .stSlider text, .stSlider div {
        color: #93C5FD !important; 
    }
    
    /* Additional fix for slider text */
    div[data-testid="stSlider"] > div > div > div {
        color: #93C5FD !important;
    }
    
    /* Ensure slider track is visible */
    div[data-testid="stSlider"] > div > div > div[aria-valuemin] {
        background-color: #4B5563 !important;
    }
    
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: #3B82F6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_meridian_header():
    """
    Render the Meridian header with logo
    """
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div>
            <div style="
                font-size: 2.2rem;
                font-weight: 700;
                color: white;
                letter-spacing: 1px;
                display: flex;
                align-items: center;
            ">
                <span style="margin-right: 10px; font-size: 1.8rem;">◑</span> MERIDIAN
            </div>
            <div style="
                font-size: 1rem;
                color: rgba(255, 255, 255, 0.9);
                margin-top: 0.2rem;
            ">
                Cryptocurrency Analysis & Prediction Platform
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_price_header(coin, price, change_24h):
    """
    Render a price info header
    
    Args:
        coin (str): Cryptocurrency name
        price (float): Current price
        change_24h (float): 24-hour percent change
    """
    change_color = "#10B981" if change_24h >= 0 else "#EF4444"
    change_icon = "▲" if change_24h >= 0 else "▼"
    
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div>
            <div style="font-size: 0.9rem; color: #64748B;">Current {coin.upper()} Price</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #1E3A8A;">${price:,.2f}</div>
        </div>
        <div style="
            background-color: {change_color}10;
            color: {change_color};
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
        ">
            {change_icon} {abs(change_24h):.2f}% (24h)
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_row(metrics_data):
    """
    Render a row of metric cards using Streamlit's native components
    
    Args:
        metrics_data (list): List of dicts with keys 'title', 'value', and optionally 'change'
    """
    # Create columns based on the number of metrics
    cols = st.columns(len(metrics_data))
    
    # Display each metric in its own column
    for i, metric in enumerate(metrics_data):
        change = None
        delta_color = "normal"
        
        if 'change' in metric and metric['change']:
            change = metric['change']
            # Determine color based on the change value
            if isinstance(change, str) and '%' in change:
                change_value = float(change.replace('%', ''))
                delta_color = "normal"  # Use Streamlit's default coloring
        
        # Use Streamlit's native metric component
        cols[i].metric(
            label=metric['title'],
            value=metric['value'],
            delta=change,
            delta_color=delta_color
        )

def render_meridian_card(title, content, border_color=None):
    """
    Render a Meridian styled card
    
    Args:
        title (str): Card title
        content (str): HTML content for the card
        border_color (str, optional): CSS color for left border
    """
    border_style = f"border-left-color: {border_color};" if border_color else ""
    st.markdown(f"""
    <div class="meridian-card" style="{border_style}">
        <div class="meridian-card-title">{title}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def render_model_badge(model_name):
    """
    Render a badge for the model type
    
    Args:
        model_name (str): Name of the prediction model
    """
    return f'<span class="model-label">{model_name}</span>'

def render_disclaimer():
    """
    Render a disclaimer about predictions
    """
    st.markdown("""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> These predictions are based on historical patterns and technical analysis.
        Cryptocurrency markets are highly volatile and unpredictable. Past performance is not indicative of future results.
        Do not use these predictions as the sole basis for investment decisions.
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """
    Render a styled footer using Streamlit's native components
    """
    st.markdown("---")  # Add a separator line
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <p style="font-weight: 600; color: #1E3A8A; margin-bottom: 0.5rem;">Meridian Crypto Analysis Platform</p>
            <p style="font-size: 0.9rem; color: #64748B;">Advanced cryptocurrency market analysis and prediction | © 2025</p>
        </div>
        """, unsafe_allow_html=True)