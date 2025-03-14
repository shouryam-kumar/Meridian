import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Basic Crypto Price Viewer", layout="wide")

st.title("Basic Cryptocurrency Price Viewer")
st.write("A simplified app to view cryptocurrency price data")

# Sidebar controls
st.sidebar.header("Select Options")
crypto = st.sidebar.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "ripple"])
days = st.sidebar.selectbox("Select Timeframe", [7, 14, 30, 90])

# Function to get data
def get_crypto_data(crypto_id, days):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert price data to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)
            
            return df
        else:
            st.error(f"API request failed with status code {response.status_code}")
            st.write(response.text)
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Get data when button is clicked
if st.sidebar.button("Get Data"):
    with st.spinner("Fetching data..."):
        data = get_crypto_data(crypto, days)
    
    if data is not None:
        # Display current price
        current_price = data['price'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
        
        # Create and display chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            )
        )
        
        fig.update_layout(
            title=f"{crypto.capitalize()} Price (Last {days} days)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data
        st.subheader("Price Data")
        st.dataframe(data)