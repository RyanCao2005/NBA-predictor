'''
normalizes the data from three different csv files and combines them into normalized_data.csv
'''

import pandas as pd

# Loading CSV files into DataFrame
df1 = pd.read_csv('out20201_20826.csv')
df2 = pd.read_csv('stats.csv')
df3 = pd.read_csv('stats10726.csv')

# List of columns to fill NaN with 0 and normalize
columns_to_normalize = ['ast', 'blk', 'dreb', 'fg3_pct', 'fg3a', 'fg3m', 'fg_pct', 'fga', 'fgm',
                         'ft_pct', 'fta', 'ftm', 'min', 'oreb', 'pf', 'pts', 'reb', 'stl',
                         'turnover', 'game.home_team_score', 'game.visitor_team_score', 'player.height_feet', 'player.height_inches', 'player.weight_pounds']

# Filling NaN values with 0 and converting all columns to numeric
df1[columns_to_normalize] = df1[columns_to_normalize].apply(pd.to_numeric, errors='coerce').fillna(0)
df2[columns_to_normalize] = df2[columns_to_normalize].apply(pd.to_numeric, errors='coerce').fillna(0)
df3[columns_to_normalize] = df3[columns_to_normalize].apply(pd.to_numeric, errors='coerce').fillna(0)

# Defining the normalization function
def normalize_column(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return df  # Return the original DataFrame

    min_val = df[column_name].min()
    max_val = df[column_name].max()

    if min_val == max_val:
        # Handle the case where min and max are the same to avoid division by zero
        df[column_name] = 0.0
    else:
        df[column_name] = (df[column_name] - min_val) / (max_val - min_val)

    return df

# Apply normalization to each column in the list
for column in columns_to_normalize:
    df1 = normalize_column(df1, column)
    df2 = normalize_column(df2, column)
    df3 = normalize_column(df3, column)

# Saving the normalized DataFrame to a new CSV file after merging
result_df = pd.concat([df1, df2, df3], ignore_index=True)
result_df.to_csv('actually_normalized_data.csv', index=False)
