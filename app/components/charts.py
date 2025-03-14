import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_price_chart(data, data_with_indicators, show_sma=True, show_bollinger=True):
    """Render the main price chart with technical indicators"""
    
    st.subheader("Price Chart")
    
    try:
        # Create Plotly chart
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.1,
                           subplot_titles=("Price", "Volume"),
                           row_heights=[0.7, 0.3])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['price'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        if data_with_indicators is not None:
            # Add technical indicators
            if show_sma:
                # Add SMA lines
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['sma_7'],
                        mode='lines',
                        name='SMA 7',
                        line=dict(color='#ff7f0e', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['sma_25'],
                        mode='lines',
                        name='SMA 25',
                        line=dict(color='#2ca02c', width=1)
                    ),
                    row=1, col=1
                )
            
            if show_bollinger:
                # Add Bollinger Bands
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['bb_upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='rgba(255, 127, 14, 0.3)', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['bb_lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='rgba(255, 127, 14, 0.3)', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.1)'
                    ),
                    row=1, col=1
                )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker=dict(color='rgba(46, 204, 113, 0.7)')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(orientation='h', y=1.02),
            xaxis2=dict(
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(title='Price (USD)'),
            yaxis2=dict(title='Volume (USD)'),
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        st.error("Error creating chart. Please try a different selection.")