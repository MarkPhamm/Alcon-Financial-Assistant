import pandas as pd

def add_timeframe_column(df, timeframe):
    """
    Add a 'Time Frame' column to the DataFrame with the specified value.

    Args:
    df (pandas.DataFrame): The input DataFrame.
    timeframe (str): The value to be added in the 'Time Frame' column.

    Returns:
    pandas.DataFrame: The DataFrame with the new 'Time Frame' column.
    """
    df['Time Frame'] = timeframe
    return df

def add_quarter_and_year_columns(df):
    """
    Add 'Quarter' and 'Year' columns to the DataFrame.

    Args:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with the new 'Quarter' and 'Year' columns.
""" 
    # Convert the 'date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract the quarter and year from the 'date' column
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    return df

def add_custom_metrics(df):
    if 'Current Assets' in df.columns and 'Current Liabilities' in df.columns:
        df['Current Ratio'] = (df['Current Assets'] / df['Current Liabilities']).round(2)
    return df

