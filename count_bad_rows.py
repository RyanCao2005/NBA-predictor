import pandas as pd

# Load the CSV file
df = pd.read_csv('games_normalized.csv')

# Find rows where both scores are 0
bad_rows = df[(df['home_team_score'] == 0) & (df['visitor_team_score'] == 0)]

print(f"Total rows in dataset: {len(df)}")
print(f"Rows with both scores = 0: {len(bad_rows)}")
print(f"Percentage of bad data: {(len(bad_rows) / len(df)) * 100:.2f}%")

# Show the indices of bad rows (first 10)
if len(bad_rows) > 0:
    print(f"\nFirst 10 bad row indices: {bad_rows.index[:10].tolist()}")
    print(f"Last 10 bad row indices: {bad_rows.index[-10:].tolist()}") 