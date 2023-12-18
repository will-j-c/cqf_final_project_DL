import requests
import pandas as pd
import os
from dotenv import load_dotenv
import json
import datetime
import time

# Load environment variables
load_dotenv('.env')

api_key = os.environ['AV_API']
base_url = f'https://www.alphavantage.co/query'
start_date = datetime.datetime(2021, 12, 13, 1)
end_date = datetime.datetime(2023, 12, 12, 23)
params = {
    'function': 'NEWS_SENTIMENT',
    'tickers': 'CRYPTO:ETH',
    'topics': 'blockchain',
    'limit': 1000,
    'time_from': start_date.strftime('%Y%m%dT%H%M'),
    'apikey': api_key
}
delta = datetime.timedelta(seconds=1)
i = 1
arr = []
while datetime.datetime.strptime(params['time_from'], '%Y%m%dT%H%M') < end_date:
    try:
        # Make request
        r = requests.get(base_url, params=params)
        print(r.url)
        feed = r.json()['feed']
        # Extract required values
        for item in feed:
            # Append the values to the arr
            arr.append((item['time_published'], item['overall_sentiment_score']))
        # Set date to last record received plus 1 second
        print(feed[-1]['time_published'])
        new_date = datetime.datetime.strptime(
            feed[-1]['time_published'], '%Y%m%dT%H%M%S')
        params['time_from'] = (new_date + delta).strftime('%Y%m%dT%H%M')
        print(params['time_from'])
        time.sleep(5)
        if i == 25:
            break
    except Exception as e:
        print(e)
        print('Hits this ')
        break

df = pd.DataFrame(arr, columns=['datetime', 'sentiment'])
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
# Group by hour
df = df.groupby(pd.Grouper(freq='H')).mean()
# Fill NaN with last sentiment
df.ffill(inplace=True)
# Output to csv
df.to_csv('data/sentiment.csv')