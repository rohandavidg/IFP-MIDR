import pandas as pd
import numpy as np

def expand_array_column(df, array_column_name):
    """Expands a column containing arrays into multiple separate feature columns."""
    if array_column_name not in df.columns:
        raise ValueError(f"Column '{array_column_name}' not found in DataFrame.")

    # Expand the array column into separate columns
    array_columns = df[array_column_name].apply(pd.Series)
    array_columns.columns = [f'{array_column_name}_value_{i+1}' for i in range(array_columns.shape[1])]

    # Drop the original array column and concatenate the expanded columns
    expanded_df = pd.concat([df.drop(columns=[array_column_name]), array_columns], axis=1)
    
    return expanded_df

def save_dataframe(df, file_path):
    """Saves a dataframe to a CSV file."""
    df.to_csv(file_path, index=False)

def load_dataframe(file_path):
    """Loads a CSV file as a dataframe."""
    return pd.read_csv(file_path)
