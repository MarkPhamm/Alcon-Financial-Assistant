import pandas as pd

def rename_columns_for_business(df):
    """
    Rename DataFrame columns for business use by adding spaces and capitalizing each word.
    
    Args:
    df (pandas.DataFrame): The input DataFrame with original column names.
    
    Returns:
    pandas.DataFrame: A DataFrame with renamed columns.
    """
    def format_column_name(col):
        # Check if the column name contains 'EBITDA'
        if 'ebitda' in col:
            return col.replace('_', ' ').title().replace('Ebitda', 'EBITDA')
        
        # Split the column name by underscores and capital letters
        words = []
        current_word = col[0]
        for char in col[1:]:
            if char.isupper() or char == '_':
                words.append(current_word)
                current_word = char if char != '_' else ''
            else:
                current_word += char
        words.append(current_word)
        
        # Capitalize each word and join with spaces
        return ' '.join(word.capitalize() for word in words if word)
    
    # Create a dictionary of old column names to new column names
    column_mapping = {col: format_column_name(col) for col in df.columns}
    
    # Rename the columns
    renamed_df = df.rename(columns=column_mapping)
    
    # Replace "E B I T D A" with "EBITDA" in all column names
    renamed_df.columns = [col.replace("E B I T D A", "EBITDA") for col in renamed_df.columns]
    
    return renamed_df