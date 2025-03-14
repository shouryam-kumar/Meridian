import requests
import pandas as pd
import os
from datetime import datetime, timedelta

class CryptoDataFetcher:
    """
    Class for fetching cryptocurrency data from various APIs.
    Currently supports CoinGecko API.
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        
    def get_available_coins(self, limit=100):
        """
        Get list of available cryptocurrencies
        
        Args:
            limit (int): Maximum number of coins to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame with coin information
        """
        url = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            coins = response.json()
            return pd.DataFrame(coins)[['id', 'symbol', 'name', 'market_cap', 'current_price']]
        else:
            print(f"Error fetching coins: {response.status_code}")
            return None
    
    def fetch_historical_data(self, coin_id, days=365, currency='usd'):
        """
        Fetch historical price data for a specific cryptocurrency
        
        Args:
            coin_id (str): Coin identifier (e.g., 'bitcoin')
            days (int or str): Number of days of data to retrieve or 'max'
            currency (str): Base currency for price data
            
        Returns:
            pandas.DataFrame: DataFrame with historical data
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': currency,
            'days': days,
            'interval': 'daily'
        }
        
        print(f"Fetching historical data for {coin_id}...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert price data to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data
            volumes = data['total_volumes']
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            df['volume'] = volume_df['volume']
            
            # Add market cap data
            market_caps = data['market_caps']
            market_cap_df = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])
            df['market_cap'] = market_cap_df['market_cap']
            
            # Format DataFrame
            df = df.drop('timestamp', axis=1)
            df = df.set_index('date')
            
            # Add daily returns (ensure this is always calculated)
            df['daily_return'] = df['price'].pct_change() * 100
            
            return df
        else:
            print(f"Error fetching data: {response.status_code}")
            return None
    
    def save_data(self, data, coin_id, directory='data'):
        """
        Save the dataframe to a CSV file
        
        Args:
            data (pandas.DataFrame): Data to save
            coin_id (str): Coin identifier
            directory (str): Directory to save the file
            
        Returns:
            str: Path to the saved file
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filename = f"{directory}/{coin_id}_historical_data.csv"
        data.to_csv(filename)
        print(f"Data saved to {filename}")
        
        return filename
    
    def get_coin_info(self, coin_id):
        """
        Get detailed information about a specific coin
        
        Args:
            coin_id (str): Coin identifier
            
        Returns:
            dict: Coin information
        """
        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching coin info: {response.status_code}")
            return None
            
    def get_multiple_coins_data(self, coin_ids, days=30):
        """
        Fetch historical data for multiple coins
        
        Args:
            coin_ids (list): List of coin identifiers
            days (int): Number of days of data to retrieve
            
        Returns:
            dict: Dictionary of DataFrames with historical data for each coin
        """
        result = {}
        for coin_id in coin_ids:
            result[coin_id] = self.fetch_historical_data(coin_id, days)
        
        return result


# Example usage
if __name__ == "__main__":
    fetcher = CryptoDataFetcher()
    
    # Get list of available coins
    coins = fetcher.get_available_coins(limit=10)
    print("Top 10 cryptocurrencies by market cap:")
    print(coins)
    
    # Fetch Bitcoin data
    btc_data = fetcher.fetch_historical_data('bitcoin', days=30)
    print(btc_data.head())
    
    # Save data
    fetcher.save_data(btc_data, 'bitcoin')