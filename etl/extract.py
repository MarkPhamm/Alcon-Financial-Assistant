import yfinance as yf
import pandas as pd

def get_income_statement(tickers, period='annually'):
    """
    Fetch income statement data for a list of tickers using yfinance.

    Args:
    tickers (str or list): A single stock ticker symbol or a list of stock ticker symbols.
    period (str): 'annually' or 'quarterly' to fetch annual or quarterly income statement data.

    Returns:
    pandas.DataFrame: A DataFrame containing the income statement data for all tickers.
    """

    all_data = []

    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]

    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        if period == 'annually':
            income_statement = ticker.financials if ticker.financials is not None else pd.DataFrame()
        elif period == 'quarterly':
            income_statement = ticker.quarterly_financials if ticker.quarterly_financials is not None else pd.DataFrame()
        else:
            raise ValueError("Invalid period. Use 'annually' or 'quarterly'.")

        if not income_statement.empty:
            # Standardize date format
            if period == 'annually':
                income_statement.columns = pd.to_datetime(income_statement.columns).to_period('Y').to_timestamp('Y')
            else:
                income_statement.columns = pd.to_datetime(income_statement.columns).to_period('Q').to_timestamp('Q')
            
            # Transpose the DataFrame
            income_statement = income_statement.T
            
            # Reset index to make date a column
            income_statement.reset_index(inplace=True)
            income_statement.rename(columns={'index': 'Date'}, inplace=True)
            
            # Add ticker column
            income_statement['Symbol'] = ticker_symbol
            
            if period == 'annually':
                # Filter for the last 4 years
                four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
                income_statement = income_statement[income_statement['Date'] >= four_years_ago]
            
            all_data.append(income_statement)

    # Concatenate all DataFrames into a single DataFrame
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        
        # Include only the specified columns
        columns_to_include = ['Date', 'Symbol', 'Tax Effect Of Unusual Items', 'Tax Rate For Calcs',
                              'Normalized EBITDA',
                              'Net Income From Continuing Operation Net Minority Interest',
                              'Reconciled Depreciation', 'Reconciled Cost Of Revenue', 'EBITDA',
                              'EBIT', 'Net Interest Income', 'Interest Expense', 'Normalized Income',
                              'Net Income From Continuing And Discontinued Operation',
                              'Total Expenses', 'Diluted Average Shares', 'Basic Average Shares',
                              'Diluted EPS', 'Basic EPS', 'Diluted NI Availto Com Stockholders',
                              'Net Income Common Stockholders', 'Net Income',
                              'Net Income Including Noncontrolling Interests',
                              'Net Income Continuous Operations', 'Tax Provision', 'Pretax Income',
                              'Net Non Operating Interest Income Expense',
                              'Interest Expense Non Operating', 'Operating Income',
                              'Operating Expense', 'Research And Development',
                              'Selling General And Administration', 'Gross Profit', 'Cost Of Revenue',
                              'Total Revenue', 'Operating Revenue']
        result_df = result_df[columns_to_include]
        
        # Reorder columns to have Date and Symbol first
        cols = ['Date', 'Symbol'] + [col for col in result_df.columns if col not in ['Date', 'Symbol']]
        result_df = result_df[cols]
        
        # Sort by Date (descending) and Symbol
        result_df = result_df.sort_values(['Symbol', 'Date'], ascending=[True, False])
    else:
        result_df = pd.DataFrame()

    return result_df

def get_balance_sheet(tickers, period='annually'):
    """
    Fetch balance sheet data for a list of tickers using yfinance.

    Args:
    tickers (str or list): A single stock ticker symbol or a list of stock ticker symbols.
    period (str): 'annually' or 'quarterly' to fetch annual or quarterly balance sheet data.

    Returns:
    pandas.DataFrame: A DataFrame containing the balance sheet data for all tickers.
    """

    all_data = []

    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]

    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        if period == 'annually':
            balance_sheet = ticker.balance_sheet if ticker.balance_sheet is not None else pd.DataFrame()
        elif period == 'quarterly':
            balance_sheet = ticker.quarterly_balance_sheet if ticker.quarterly_balance_sheet is not None else pd.DataFrame()
        else:
            raise ValueError("Invalid period. Use 'annually' or 'quarterly'.")

        if not balance_sheet.empty:
            # Standardize date format
            if period == 'annually':
                balance_sheet.columns = pd.to_datetime(balance_sheet.columns).to_period('Y').to_timestamp('Y')
            else:
                balance_sheet.columns = pd.to_datetime(balance_sheet.columns).to_period('Q').to_timestamp('Q')
            
            # Transpose the DataFrame
            balance_sheet = balance_sheet.T
            
            # Reset index to make date a column
            balance_sheet.reset_index(inplace=True)
            balance_sheet.rename(columns={'index': 'Date'}, inplace=True)
            
            # Add ticker column
            balance_sheet['Symbol'] = ticker_symbol
            
            if period == 'annually':
                # Filter for the last 4 years
                four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
                balance_sheet = balance_sheet[balance_sheet['Date'] >= four_years_ago]
            
            all_data.append(balance_sheet)

    # Concatenate all DataFrames into a single DataFrame
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        
        # Include only the specified columns
        columns_to_include = ['Date', 'Symbol', 'Ordinary Shares Number', 'Share Issued',
                              'Tangible Book Value', 'Invested Capital', 'Working Capital',
                              'Net Tangible Assets', 'Common Stock Equity', 'Total Capitalization',
                              'Total Equity Gross Minority Interest', 'Stockholders Equity',
                              'Capital Stock', 'Common Stock',
                              'Total Liabilities Net Minority Interest',
                              'Total Non Current Liabilities Net Minority Interest',
                              'Current Liabilities', 'Payables', 'Accounts Payable', 'Total Assets',
                              'Total Non Current Assets', 'Other Non Current Assets', 'Net PPE',
                              'Accumulated Depreciation', 'Gross PPE',
                              'Machinery Furniture Equipment', 'Properties', 'Current Assets',
                              'Inventory', 'Accounts Receivable',
                              'Cash Cash Equivalents And Short Term Investments',
                              'Cash And Cash Equivalents']
        result_df = result_df[columns_to_include]
        
        # Reorder columns to have Date and Symbol first
        cols = ['Date', 'Symbol'] + [col for col in result_df.columns if col not in ['Date', 'Symbol']]
        result_df = result_df[cols]
        
        # Sort by Date (descending) and Symbol
        result_df = result_df.sort_values(['Symbol', 'Date'], ascending=[True, False])
    else:
        result_df = pd.DataFrame()

    return result_df

def get_cashflow(tickers, period='annually'):
    """
    Fetch cash flow data for a list of tickers using yfinance.

    Args:
    tickers (str or list): A single stock ticker symbol or a list of stock ticker symbols.
    period (str): 'annually' or 'quarterly' to fetch annual or quarterly cash flow data.

    Returns:
    pandas.DataFrame: A DataFrame containing the cash flow data for all tickers.
    """

    all_data = []

    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]

    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        if period == 'annually':
            cashflow = ticker.cashflow if ticker.cashflow is not None else pd.DataFrame()
        elif period == 'quarterly':
            cashflow = ticker.quarterly_cashflow if ticker.quarterly_cashflow is not None else pd.DataFrame()
        else:
            raise ValueError("Invalid period. Use 'annually' or 'quarterly'.")

        if not cashflow.empty:
            # Standardize date format
            if period == 'annually':
                cashflow.columns = pd.to_datetime(cashflow.columns).to_period('Y').to_timestamp('Y')
            else:
                cashflow.columns = pd.to_datetime(cashflow.columns).to_period('Q').to_timestamp('Q')
            
            # Transpose the DataFrame
            cashflow = cashflow.T
            
            # Reset index to make date a column
            cashflow.reset_index(inplace=True)
            cashflow.rename(columns={'index': 'Date'}, inplace=True)
            
            # Add ticker column
            cashflow['Symbol'] = ticker_symbol
            
            if period == 'annually':
                # Filter for the last 4 years
                four_years_ago = pd.Timestamp.now() - pd.DateOffset(years=4)
                cashflow = cashflow[cashflow['Date'] >= four_years_ago]
            
            all_data.append(cashflow)

    # Concatenate all DataFrames into a single DataFrame
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        
        # Include only the specified columns
        columns_to_include = ['Date', 'Symbol', 'Free Cash Flow', 'Capital Expenditure',
                              'End Cash Position', 'Beginning Cash Position', 'Changes In Cash',
                              'Financing Cash Flow', 'Net Issuance Payments Of Debt',
                              'Investing Cash Flow', 'Net PPE Purchase And Sale', 'Purchase Of PPE',
                              'Operating Cash Flow', 'Change In Working Capital',
                              'Change In Inventory', 'Change In Receivables', 'Other Non Cash Items',
                              'Depreciation And Amortization',
                              'Net Income From Continuing Operations']
        result_df = result_df[columns_to_include]
        
        # Reorder columns to have Date and Symbol first
        cols = ['Date', 'Symbol'] + [col for col in result_df.columns if col not in ['Date', 'Symbol']]
        result_df = result_df[cols]
        
        # Sort by Date (descending) and Symbol
        result_df = result_df.sort_values(['Symbol', 'Date'], ascending=[True, False])
    else:
        result_df = pd.DataFrame()

    return result_df

