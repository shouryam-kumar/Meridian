import requests

def test_coingecko_api():
    print("Testing CoinGecko API...")
    try:
        # Test the ping endpoint
        response = requests.get("https://api.coingecko.com/api/v3/ping")
        print(f"Ping response: {response.status_code}")
        print(response.json())
        
        # Test getting bitcoin data
        params = {
            'vs_currency': 'usd',
            'days': 7,
            'interval': 'daily'
        }
        response = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart", params=params)
        print(f"Bitcoin data response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Received {len(data['prices'])} price points")
        else:
            print(response.text)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_coingecko_api()