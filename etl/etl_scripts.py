import os
from typing import List, Tuple

import pandas as pd

import extract
import transform
import load

def get_data_directory() -> str:
    """Get the path to the data directory."""
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, 'data')

def etl_process(tickers: List[str], api_key: str, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the ETL (Extract, Transform, Load) process for financial data.

    Args:
        tickers (List[str]): List of stock ticker symbols.
        api_key (str): API key for financial data retrieval.
        data_dir (str): Directory to save the processed data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Processed DataFrames for
        income statement, balance sheet, cash flow, and key metrics.
    """
    # Extract data
    key_metrics_df = extract.get_key_metrics(tickers, api_key)
    balance_sheet_df = extract.get_balance_sheet(tickers, api_key)
    cash_flow_df = extract.get_cash_flow_statement(tickers, api_key)
    income_statement_df = extract.get_income_statement(tickers, api_key)

    # Transform data
    dataframes = [key_metrics_df, balance_sheet_df, cash_flow_df, income_statement_df]
    transformed_dfs = [transform.rename_columns_for_business(df) for df in dataframes]

    # Load data
    file_names = ['key_metrics', 'balance_sheet', 'cash_flow', 'income_statement']
    for df, file_name in zip(transformed_dfs, file_names):
        load.load_data(df, data_dir, file_name)

    return tuple(transformed_dfs)

def main():
    tickers = [
        "ALC",   # Alcon Inc.
        "JNJ",   # Johnson & Johnson
        "BLCO"   # Bausch + Lomb Corporation
        # "ABT",   # Abbott Laboratories
        # "AFXXF", # Carl Zeiss Meditec AG (OTC)
        # "MDT",   # Medtronic PLC
        # "SYK",   # Stryker Corporation
        # "BSX",   # Boston Scientific Corporation
        # "NVS"    # Novartis AG
    ]

    api_key = os.environ.get('FMP_API_KEY')
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable is not set")
    data_dir = get_data_directory()
    income_statement_df, balance_sheet_df, cash_flow_df, key_metrics_df = etl_process(tickers, api_key, data_dir)
    print(income_statement_df.head())
    print("ETL process completed successfully.")

if __name__ == "__main__":
    main()
