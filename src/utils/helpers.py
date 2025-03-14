import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def ensure_directory_exists(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def date_to_str(date):
    """
    Convert date to string format
    
    Args:
        date (datetime): Date to convert
        
    Returns:
        str: Formatted date string
    """
    return date.strftime('%Y-%m-%d')

def str_to_date(date_str):
    """
    Convert string to date
    
    Args:
        date_str (str): Date string
        
    Returns:
        datetime: Date object
    """
    return datetime.strptime(date_str, '%Y-%m-%d')

def calculate_returns(prices):
    """
    Calculate returns from price series
    
    Args:
        prices (pandas.Series): Price series
        
    Returns:
        pandas.Series: Returns series
    """
    return prices.pct_change() * 100

def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns from returns series
    
    Args:
        returns (pandas.Series): Returns series
        
    Returns:
        pandas.Series: Cumulative returns series
    """
    return (returns / 100 + 1).cumprod() * 100 - 100

def calculate_volatility(returns, window=30):
    """
    Calculate rolling volatility
    
    Args:
        returns (pandas.Series): Returns series
        window (int): Window size for rolling calculation
        
    Returns:
        pandas.Series: Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(365)

def get_model_performance_metrics(predictions, actuals):
    """
    Calculate model performance metrics
    
    Args:
        predictions (pandas.Series): Predicted values
        actuals (pandas.Series): Actual values
        
    Returns:
        dict: Performance metrics
    """
    # Calculate errors
    errors = actuals - predictions
    abs_errors = np.abs(errors)
    
    # Calculate metrics
    metrics = {
        'MAE': abs_errors.mean(),  # Mean Absolute Error
        'RMSE': np.sqrt((errors ** 2).mean()),  # Root Mean Squared Error
        'MAPE': (abs_errors / actuals).mean() * 100,  # Mean Absolute Percentage Error
        'Direction Accuracy': (np.sign(predictions.diff()) == np.sign(actuals.diff())).mean() * 100  # Direction accuracy
    }
    
    return metrics