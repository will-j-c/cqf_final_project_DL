import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv('.env')

api_key = os.environ['AV_API']
base_url = f'https://www.alphavantage.co/query'
params = {
    'function': 'NEWS_SENTIMENT',
    'tickers': 'CRYPTO:ETH',
    'topics': 'blockchain',
    'limit': 1000,
    'apikey': api_key
}
# r = requests.get(base_url, params=params)
# print(api_key)
# print(r.json())
# feed = r.json()['feed']
with open('data.json') as f:
    feed = json.load(f)['feed']
arr = []

for item in feed:
    arr.append((item['time_published'], item['overall_sentiment_score']))

df = pd.DataFrame(arr, columns=['datetime', 'sentiment'])
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
# Group by hour
df = df.groupby(pd.Grouper(freq='H')).mean()
# Fill NaN with last sentiment
df.ffill(inplace=True)
# Output to csv
df.to_csv('data/sentiment.csv')