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
    
    def fetch_historical_data(self, coin, timeframe):
        """
        Fetch historical price data for a cryptocurrency
        
        Args:
            coin (str): Cryptocurrency symbol (e.g., 'bitcoin', 'ethereum')
            timeframe (str): Time period for historical data ('30d', '90d', '180d', '1y', 'max')
            
        Returns:
            pandas.DataFrame: DataFrame with historical price data
        """
        try:
            # Map timeframe to number of days
            days_map = {
                '30d': 30,
                '90d': 90,
                '180d': 180,
                '1y': 365,
                'max': 'max'
            }
            
            days = days_map.get(timeframe, 30)  # Default to 30 days if invalid timeframe
            
            # Construct API URL
            base_url = "https://api.coingecko.com/api/v3/coins"
            
            # Format the market_chart endpoint URL
            if days == 'max':
                url = f"{base_url}/{coin}/market_chart?vs_currency=usd&days=max"
            else:
                url = f"{base_url}/{coin}/market_chart?vs_currency=usd&days={days}&interval=daily"
            
            print(f"Fetching historical data for {coin}...")
            
            # Make the API request
            response = requests.get(url)
            
            # Check for rate limiting
            if response.status_code == 429:
                raise Exception("429: API rate limit exceeded. The free tier limit has been reached. Please wait 5-10 minutes before trying again.")
            
            # Check for other errors
            if response.status_code != 200:
                raise Exception(f"Error fetching data: {response.status_code}")
            
            # Parse the response JSON
            data = response.json()
            
            if not data or 'prices' not in data:
                return None
            
            # Process price data
            price_data = []
            volume_data = []
            
            for price_point, volume_point in zip(data['prices'], data['total_volumes']):
                timestamp = price_point[0]
                price = price_point[1]
                volume = volume_point[1]
                
                # Convert timestamp to datetime
                date = datetime.fromtimestamp(timestamp / 1000)
                
                price_data.append({
                    'date': date,
                    'price': price,
                    'volume': volume
                })
            
            # Create DataFrame
            df = pd.DataFrame(price_data)
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Add daily return
            df['daily_return'] = df['price'].pct_change()
            
            return df
            
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                raise Exception("429: API rate limit exceeded. The free tier limit has been reached. Please wait 5-10 minutes before trying again.")
            else:
                raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            # Propagate the error with clear message
            raise Exception(f"Error fetching data: {str(e)}")
    
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