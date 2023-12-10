# Will Colgate CQF Final Assignment

### Installation

Python 3.10 is required.

The preferred installation method is by using [pipenv](https://pipenv.pypa.io/en/latest/) and simply using the following command.

`pipenv install`

This will automatically create a virtual environment and install teh necessary packages.

For `pip` users, you should create a virtual environment using `venv` and install the requirements.txt as normal.

Navigate to the project folder and create a virtual environment folder

`python3 -m venv venv`

Activate the virtual environment

`source venv/bin/activate`

Install the requirements as normal

`pip install -r requirements.txt`

### Files
#### download.py
You can download data using `yfinance` by running this script with the ticker, period and interval that you want to download.

For example, to download the daily data for Apple for past 5 years:

`python3 download.py AAPL 5y 1d`

Data will be saved in the data folder as a CSV file.