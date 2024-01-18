import requests
import pandas as pd

r = requests.get('https://api.alternative.me/fng/?limit=0')
data = r.json()['data']
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
df.index = df['timestamp']
# df.drop(['timestamp', 'time_until_update'], axis=0, inplace=True)
df.to_csv('data/raw/crypto_greed_fear_index.csv')