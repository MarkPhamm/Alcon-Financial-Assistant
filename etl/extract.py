# Import necessary libraries
import yfinance as yf
import pandas as pd
import requests

import os

import dotenv
dotenv.load_dotenv()

# Get the current working directory
current_dir = os.getcwd()
# Move up one level from the current directory
parent_dir = os.path.dirname(current_dir)
# Change directory into data directory
data_dir = os.path.join(parent_dir, 'data')


def get_income_statement(tickers, api_key):
    """
    Fetch income statement data for a list of tickers using the Financial Modeling Prep API.

    Args:
    tickers (list): A list of stock ticker symbols.
    api_key (str): Your API key for Financial Modeling Prep.

    Returns:
    pandas.DataFrame: A DataFrame containing the income statement data for all tickers.
    """
    all_data = []

    for ticker in tickers:
        # Define the endpoint with 10 years of data
        url = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey={api_key}'

        # Send a GET request to the API
        response = requests.get(url)

        # Parse the JSON data
        data = response.json()

        # Convert the data to a DataFrame and add a column for the ticker
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        all_data.append(df)

    # Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(all_data, ignore_index=True)
    
    return result_df

def get_balance_sheet(tickers, api_key):
    """
    Fetch balance sheet data for a list of tickers using the Financial Modeling Prep API.

    Args:
    tickers (list): A list of stock ticker symbols.
    api_key (str): Your API key for Financial Modeling Prep.

    Returns:
    pandas.DataFrame: A DataFrame containing the balance sheet data for all tickers.
    """
    all_data = []

    for ticker in tickers:
        url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=120&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        all_data.append(df)

    result_df = pd.concat(all_data, ignore_index=True)
    return result_df

def get_cash_flow_statement(tickers, api_key):
    """
    Fetch cash flow statement data for a list of tickers using the Financial Modeling Prep API.

    Args:
    tickers (list): A list of stock ticker symbols.
    api_key (str): Your API key for Financial Modeling Prep.

    Returns:
    pandas.DataFrame: A DataFrame containing the cash flow statement data for all tickers.
    """
    all_data = []

    for ticker in tickers:
        url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=120&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        all_data.append(df)

    result_df = pd.concat(all_data, ignore_index=True)
    return result_df

def get_key_metrics(tickers, api_key):
    """
    Fetch key metrics data for a list of tickers using the Financial Modeling Prep API.

    Args:
    tickers (list): A list of stock ticker symbols.
    api_key (str): Your API key for Financial Modeling Prep.

    Returns:
    pandas.DataFrame: A DataFrame containing the key metrics data for all tickers.
    """
    all_data = []

    for ticker in tickers:
        url = f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?limit=120&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        all_data.append(df)

    result_df = pd.concat(all_data, ignore_index=True)
    return result_df