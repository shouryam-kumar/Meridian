import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render_indicators(data_with_indicators, show_rsi=True, show_macd=True):
    """Render technical indicators in tabs"""
    
    st.subheader("Technical Indicators")
    
    # Create tabs for different indicators
    tabs_to_show = []
    if show_rsi:
        tabs_to_show.append("RSI")
    if show_macd:
        tabs_to_show.append("MACD")
    
    if tabs_to_show:
        ind_tabs = st.tabs(tabs_to_show)
        
        tab_index = 0
        if show_rsi:
            with ind_tabs[tab_index]:
                render_rsi(data_with_indicators)
            tab_index += 1
        
        if show_macd:
            with ind_tabs[tab_index]:
                render_macd(data_with_indicators)

def render_rsi(data):
    """Render RSI indicator chart"""
    
    try:
        # RSI Chart
        rsi_fig = go.Figure()
        
        rsi_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#8884d8', width=2)
            )
        )
        
        # Add overbought/oversold lines
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        rsi_fig.update_layout(
            height=300,
            title="Relative Strength Index (RSI)",
            yaxis=dict(title='RSI Value'),
            template='plotly_white'
        )
        
        st.plotly_chart(rsi_fig, use_container_width=True)
        
        st.markdown("""
        **RSI Interpretation:**
        - RSI above 70 is considered overbought (potential sell signal)
        - RSI below 30 is considered oversold (potential buy signal)
        - The direction of the RSI can also indicate momentum
        """)
    except Exception as e:
        print(f"Error displaying RSI: {str(e)}")
        st.error("Error displaying RSI indicator.")

def render_macd(data):
    """Render MACD indicator chart"""
    
    try:
        # MACD Chart
        macd_fig = go.Figure()
        
        macd_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='#8884d8', width=2)
            )
        )
        
        macd_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['macd_signal'],
                mode='lines',
                name='Signal Line',
                line=dict(color='#ff7f0e', width=2)
            )
        )
        
        macd_fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['macd_hist'],
                name='Histogram',
                marker=dict(
                    color=np.where(data['macd_hist'] > 0, 'green', 'red')
                )
            )
        )
        
        macd_fig.update_layout(
            height=300,
            title="Moving Average Convergence Divergence (MACD)",
            yaxis=dict(title='Value'),
            template='plotly_white'
        )
        
        st.plotly_chart(macd_fig, use_container_width=True)
        
        st.markdown("""
        **MACD Interpretation:**
        - When MACD crosses above the signal line: Bullish signal
        - When MACD crosses below the signal line: Bearish signal
        - Histogram shows the difference between MACD and signal line
        """)
    except Exception as e:
        print(f"Error displaying MACD: {str(e)}")
        st.error("Error displaying MACD indicator.")