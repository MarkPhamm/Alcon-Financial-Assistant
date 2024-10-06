import os
import pandas as pd


def load_data(df, data_dir, file_name):
    """
    Load a DataFrame to a CSV file in the specified data directory.

    Args:
    df (pandas.DataFrame): The DataFrame to be saved.
    data_dir (str): The directory where the CSV file will be saved.
    file_name (str): The name of the file to be saved (without extension).
    """

    # Create the full file path
    file_path = os.path.join(data_dir, f"{file_name}.csv")

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
