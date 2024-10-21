import os
import sys
import time
import logging
from typing import List, Tuple

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extract
import transform
import load

import config as cfg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data_directory() -> str:
    """Get the path to the data directory."""
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    print(data_dir)
    return data_dir

def etl_process(tickers: List[str], data_dir: str):
    """
    Execute the ETL (Extract, Transform, Load) process for financial data.

    Args:
        tickers (List[str]): List of stock ticker symbols.
        api_key (str): API key for financial data retrieval.
        data_dir (str): Directory to save the processed data.

    Returns:
        None
    """
    start_time = time.time()
    logging.info("Starting ETL process")

    # Extract annual data
    logging.info("Extracting annual data")
    annually_income_statement_df = extract.get_income_statement(tickers = cfg.tickers, period = 'annually')
    annually_balance_sheet_df = extract.get_balance_sheet(tickers = cfg.tickers, period = 'annually')
    annually_cash_flow_df = extract.get_cashflow(tickers = cfg.tickers, period = 'annually')

    # Extract quarterly data
    logging.info("Extracting quarterly data")
    quarterly_income_statement_df = extract.get_income_statement(tickers = cfg.tickers, period = 'quarterly')
    quarterly_balance_sheet_df = extract.get_balance_sheet(tickers = cfg.tickers, period = 'quarterly')
    quarterly_cash_flow_df = extract.get_cashflow(tickers = cfg.tickers, period = 'quarterly')

    # Transform data
    logging.info("Transforming data")
    annual_dataframes = [annually_income_statement_df, annually_balance_sheet_df, annually_cash_flow_df]
    quarterly_dataframes = [quarterly_income_statement_df, quarterly_balance_sheet_df, quarterly_cash_flow_df]
    
    transformed_annual_dfs = [
        transform.add_custom_metrics(transform.add_quarter_and_year_columns(df)) for df in annual_dataframes
    ]
    
    transformed_quarterly_dfs = [
        transform.add_custom_metrics(transform.add_quarter_and_year_columns(df)) for df in quarterly_dataframes
    ]
    # Load data
    logging.info("Loading data")
    load.load_data(transformed_annual_dfs[0], data_dir, 'annually_income_statement')
    load.load_data(transformed_annual_dfs[1], data_dir, 'annually_balance_sheet')
    load.load_data(transformed_annual_dfs[2], data_dir, 'annually_cash_flow')
    
    load.load_data(transformed_quarterly_dfs[0], data_dir, 'quarterly_income_statement')
    load.load_data(transformed_quarterly_dfs[1], data_dir, 'quarterly_balance_sheet')
    load.load_data(transformed_quarterly_dfs[2], data_dir, 'quarterly_cash_flow')

    end_time = time.time()
    logging.info(f"ETL process completed in {end_time - start_time:.2f} seconds")

    return None

def main():
    tickers = cfg.tickers

    data_dir = get_data_directory()
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    etl_process(tickers, data_dir)
    logging.info("ETL process completed successfully.")

if __name__ == "__main__":
    main()
