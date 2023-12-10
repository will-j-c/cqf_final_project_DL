import yfinance as yf
import sys


def download_ticker(ticker_str, period_str):
    ticker = yf.Ticker(ticker_str)
    history = ticker.history(period=period_str)
    history.to_csv(f'data/{ticker_str}.csv')

if __name__ == '__main__':
    try:
        # Takes the user inputs from the command line
        ticker_input = str(sys.argv[1])
        period_input = str(sys.argv[2])
        # Download and save data as a csv
        download_ticker(ticker_input, period_input)
    except:
        print('Please input a valid ticker and time period')
