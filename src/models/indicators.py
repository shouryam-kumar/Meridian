import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Class for calculating technical indicators for cryptocurrency data analysis.
    """
    
    @staticmethod
    def add_all_indicators(df, window_short=7, window_medium=25, window_long=99):
        """
        Add all technical indicators to a DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            window_short (int): Short-term window period
            window_medium (int): Medium-term window period
            window_long (int): Long-term window period
            
        Returns:
            pandas.DataFrame: DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Add moving averages
        result = TechnicalIndicators.add_moving_averages(
            result, window_short, window_medium, window_long
        )
        
        # Add RSI
        result = TechnicalIndicators.add_rsi(result, window=14)
        
        # Add MACD
        result = TechnicalIndicators.add_macd(result)
        
        # Add Bollinger Bands
        result = TechnicalIndicators.add_bollinger_bands(result, window=20)
        
        # Add ATR
        result = TechnicalIndicators.add_atr(result, window=14)
        
        # Add momentum
        result = TechnicalIndicators.add_momentum(result, window=10)
        
        return result
    
    @staticmethod
    def add_moving_averages(df, window_short=7, window_medium=25, window_long=99):
        """
        Add moving average indicators to the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            window_short (int): Short-term window period
            window_medium (int): Medium-term window period
            window_long (int): Long-term window period
            
        Returns:
            pandas.DataFrame: DataFrame with moving averages
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Add simple moving averages
        result[f'sma_{window_short}'] = result['price'].rolling(window=window_short).mean()
        result[f'sma_{window_medium}'] = result['price'].rolling(window=window_medium).mean()
        result[f'sma_{window_long}'] = result['price'].rolling(window=window_long).mean()
        
        # Add exponential moving averages
        result[f'ema_{window_short}'] = result['price'].ewm(span=window_short, adjust=False).mean()
        result[f'ema_{window_medium}'] = result['price'].ewm(span=window_medium, adjust=False).mean()
        result[f'ema_{window_long}'] = result['price'].ewm(span=window_long, adjust=False).mean()
        
        # Add moving average crossover signals
        result['sma_cross'] = 0
        result.loc[result[f'sma_{window_short}'] > result[f'sma_{window_medium}'], 'sma_cross'] = 1
        result.loc[result[f'sma_{window_short}'] < result[f'sma_{window_medium}'], 'sma_cross'] = -1
        
        result['ema_cross'] = 0
        result.loc[result[f'ema_{window_short}'] > result[f'ema_{window_medium}'], 'ema_cross'] = 1
        result.loc[result[f'ema_{window_short}'] < result[f'ema_{window_medium}'], 'ema_cross'] = -1
        
        return result
    
    @staticmethod
    def add_rsi(df, window=14):
        """
        Add Relative Strength Index (RSI) to the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            window (int): Window period for RSI calculation
            
        Returns:
            pandas.DataFrame: DataFrame with RSI
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate daily price changes
        delta = result['price'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Add RSI signal
        result['rsi_signal'] = 0
        result.loc[result['rsi'] < 30, 'rsi_signal'] = 1  # Oversold - potential buy
        result.loc[result['rsi'] > 70, 'rsi_signal'] = -1  # Overbought - potential sell
        
        return result
    
    @staticmethod
    def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """
        Add Moving Average Convergence Divergence (MACD) to the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            
        Returns:
            pandas.DataFrame: DataFrame with MACD
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate MACD
        ema_fast = result['price'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = result['price'].ewm(span=slow_period, adjust=False).mean()
        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=signal_period, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # Add MACD crossover signal
        result['macd_cross'] = 0
        result.loc[result['macd'] > result['macd_signal'], 'macd_cross'] = 1
        result.loc[result['macd'] < result['macd_signal'], 'macd_cross'] = -1
        
        return result
    
    @staticmethod
    def add_bollinger_bands(df, window=20, num_std=2):
        """
        Add Bollinger Bands to the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            window (int): Window period for moving average
            num_std (int): Number of standard deviations
            
        Returns:
            pandas.DataFrame: DataFrame with Bollinger Bands
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate Bollinger Bands
        result['bb_middle'] = result['price'].rolling(window=window).mean()
        result['bb_std'] = result['price'].rolling(window=window).std()
        result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * num_std)
        result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * num_std)
        
        # Add Bollinger Band signals
        result['bb_signal'] = 0
        result.loc[result['price'] < result['bb_lower'], 'bb_signal'] = 1  # Below lower band - potential buy
        result.loc[result['price'] > result['bb_upper'], 'bb_signal'] = -1  # Above upper band - potential sell
        
        return result
    
    @staticmethod
    def add_atr(df, window=14):
        """
        Add Average True Range (ATR) to the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            window (int): Window period for ATR calculation
            
        Returns:
            pandas.DataFrame: DataFrame with ATR
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate high-low, high-close, and low-close ranges
        result['hl'] = result['price'].shift(1).diff().abs()
        
        # Calculate ATR
        result['atr'] = result['hl'].rolling(window=window).mean()
        
        return result
    
    @staticmethod
    def add_momentum(df, window=10):
        """
        Add momentum indicator to the DataFrame
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            window (int): Window period for momentum calculation
            
        Returns:
            pandas.DataFrame: DataFrame with momentum
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate momentum
        result['momentum'] = result['price'].diff(window)
        
        # Add momentum signal
        result['momentum_signal'] = 0
        result.loc[result['momentum'] > 0, 'momentum_signal'] = 1
        result.loc[result['momentum'] < 0, 'momentum_signal'] = -1
        
        return result


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100)
    prices = np.random.normal(loc=100, scale=10, size=100).cumsum()
    
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.rand(100) * 1000000
    }, index=dates)
    
    # Add technical indicators
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)
    
    # Show results
    print(df_with_indicators.tail())