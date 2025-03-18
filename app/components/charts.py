import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def render_price_chart(data):
    """
    Render price chart using plotly
    
    Args:
        data (pandas.DataFrame): DataFrame with price data
    """
    # Create price figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['price'],
            mode='lines',
            name='Price',
            line=dict(color='#3B82F6', width=2)
        )
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Price (USD)",
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.5)',
            zeroline=False
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Render in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def render_volume_chart(data):
    """
    Render volume chart using plotly
    
    Args:
        data (pandas.DataFrame): DataFrame with volume data
    """
    # Create volume figure
    fig = go.Figure()
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker=dict(color='rgba(59, 130, 246, 0.5)')
        )
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Volume (USD)",
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.5)',
            zeroline=False
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Render in Streamlit
    st.plotly_chart(fig, use_container_width=True)