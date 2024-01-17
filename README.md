# CQF Final Assignment June 2023 Cohort

### Package Installation

Python 3.10 is required.

The preferred installation method is by using [pipenv](https://pipenv.pypa.io/en/latest/) and simply using the following command.

`pipenv install`

This will automatically create a virtual environment and install the necessary packages.

For `pip` users, you should create a virtual environment using `venv` and install the requirements.txt as normal.

Navigate to the project folder and create a virtual environment folder

`python3 -m venv venv`

Activate the virtual environment

`source venv/bin/activate`

Install the requirements as normal

`pip install -r requirements.txt`

### Workflow

##### Clone from GitHub

1. `git clone https://github.com/will-j-c/cqf_final_project_DL.git`
2. Installation following the instructions above
3. cd into working directory
4. Run `python3 src/clean_scale.py`

##### Fresh run of entire workflow

1. Install dependencies as per above
2. Run `python3 main.py`

Note that the entire script will take several hours to run, depending on the hardware that you are running it on.

### Known Package Instabilities and Fixes

`boruta_py` - The pypi package has some know issues around deprecated `numpy` types. If these issues occur, you may need to manually change `np.float`, `np.int` and `np.bool` to their native python equivalents and it should work. See [this](https://github.com/scikit-learn-contrib/boruta_py/issues/122) GitHub issue for further details.

`pandas-ta` - The requirements.txt file includes the cutting edge release of this package pulled from GitHub. This is due to the pypi package breaking when trying to use the Strategy class. See [this](https://github.com/twopirllc/pandas-ta?tab=readme-ov-file#latest-version) GitHub page.

### Main File Descriptions

##### report.ipynb

This is the main jupyter notebook report. It will run assuming the installation instructions have been followed or the instructor is running the code as submitted to the institute.

##### clean_scale.py

This script imports the raw data from `data/raw/Gemini_ETHUSD_1h.csv`, engineers features using `pandas-ta`, adds temporal data, cleans the dataframe and scales the data according to outliers.

##### feature_select_reduce.py

This script establishes several different pipelines dealing with collinearity, algorithmic feature selection and dimensionality reduction and tests these pipelines against a baseline `keras` model.

##### model_selection.py

This script defines 6 different models discussed in the report and tests them with various look back periods after running the pipeline chosen in `feature_select_reduce.py`.

##### model_tuning.py

This script takes the selected best model produced in `model_selection.py` and uses Keras Tuner to tune the hyperparameters. 

##### helpers.py

Contains a number of helper classes and functions that are used in various places.

##### plots.py

Contains code for various plots that have been reused in numerous places.

##### strategy.py

This file contains a BaseStrategyBacktest class, with standard functionality and plotting for any strategy and a LongOnlyBacktest class which inherits from the BaseStrategyBacktest but extends it with logic that processes the signals from the model predictions. The file was designed this way so that further strategies could be implemented in the future.

##### config.py

Contains all the standard imports to be imported into the notebook.

##### greed_fear_index.py

A script that calls the api for the fear and greed index, cleans the data and saves it to a csv.

##### download.py (not used but kept for reference)

You can download data using `yfinance` by running this script with the ticker, period and interval that you want to download.

For example, to download the daily data for Apple for past 5 years:

`python3 download.py AAPL 5y 1d`

Data will be saved in the data folder as a CSV file.