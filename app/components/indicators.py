import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render_indicators(data, selected_indicators):
    """
    Render technical indicators and return data with indicators added
    
    Args:
        data (pandas.DataFrame): Price data
        selected_indicators (list): List of indicators to display
        
    Returns:
        pandas.DataFrame: Data with indicators added
    """
    # Make a copy of the original data
    data_with_indicators = data.copy()
    
    try:
        # Add indicators based on selection
        for indicator in selected_indicators:
            if indicator == "sma":
                # Add Simple Moving Average
                data_with_indicators['sma_7'] = data['price'].rolling(window=7).mean()
                data_with_indicators['sma_25'] = data['price'].rolling(window=25).mean()
                
                # Display SMA chart
                st.subheader("Simple Moving Average (SMA)")
                # Your visualization code here...
                
            elif indicator == "ema":
                # Add Exponential Moving Average
                data_with_indicators['ema_12'] = data['price'].ewm(span=12, adjust=False).mean()
                data_with_indicators['ema_26'] = data['price'].ewm(span=26, adjust=False).mean()
                
                # Display EMA chart
                st.subheader("Exponential Moving Average (EMA)")
                # Your visualization code here...
                
            elif indicator == "rsi":
                # Calculate RSI
                delta = data['price'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                data_with_indicators['rsi'] = 100 - (100 / (1 + rs))
                
                # Display RSI chart
                st.subheader("Relative Strength Index (RSI)")
                
                
            elif indicator == "macd":
                # Calculate MACD
                ema_12 = data['price'].ewm(span=12, adjust=False).mean()
                ema_26 = data['price'].ewm(span=26, adjust=False).mean()
                data_with_indicators['macd'] = ema_12 - ema_26
                data_with_indicators['macd_signal'] = data_with_indicators['macd'].ewm(span=9, adjust=False).mean()
                
                # Display MACD chart
                st.subheader("Moving Average Convergence Divergence (MACD)")
                
                
            elif indicator == "bollinger":
                # Calculate Bollinger Bands
                window = 20
                data_with_indicators['bb_middle'] = data['price'].rolling(window=window).mean()
                data_with_indicators['bb_std'] = data['price'].rolling(window=window).std()
                data_with_indicators['bb_upper'] = data_with_indicators['bb_middle'] + 2 * data_with_indicators['bb_std']
                data_with_indicators['bb_lower'] = data_with_indicators['bb_middle'] - 2 * data_with_indicators['bb_std']
                
                # Display Bollinger Bands chart
                st.subheader("Bollinger Bands")
                
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
    
    # Fill NaN values that may have been created by indicators
    data_with_indicators = data_with_indicators.fillna(method='ffill').fillna(method='bfill')
    
    return data_with_indicators

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