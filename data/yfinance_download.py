import yfinance as yf
import sys

def download_ticker(ticker_str, period_str, interval_str):
    ticker = yf.Ticker(ticker_str)
    filename = f'{ticker_str}_{period_str}_{interval_str}.csv'
    history = ticker.history(period=period_str, interval=interval_str)
    # Check if some data was returned
    if len(history) != 0:
        history.to_csv(f'data/{filename}')
        print(f'Successfully downloaded and saved as {filename}')
    else:
        print('No data was returned')


if __name__ == '__main__':
    try:
        # Takes the user inputs from the command line
        ticker_input = str(sys.argv[1])
        period_input = str(sys.argv[2])
        interval_input = str(sys.argv[3])
        # Download and save data as a csv
        download_ticker(ticker_input, period_input, interval_input)
    except:
        print('Please input a valid ticker, time period and interval\nValid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max\nValid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo')
