import pandas as pd

def clean_games_data():
    """
    Clean the games_normalized.csv file by removing rows where both 
    home_team_score and visitor_team_score are 0 (bad data).
    """
    print("Loading games_normalized.csv...")
    
    # Load the CSV file
    df = pd.read_csv('games_normalized.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original dataset has {len(df)} rows")
    
    # Check how many rows have 0 scores
    zero_scores = df[(df['home_team_score'] == 0) & (df['visitor_team_score'] == 0)]
    print(f"Rows with both scores = 0: {len(zero_scores)}")
    
    # Filter out rows where both scores are 0
    cleaned_df = df[~((df['home_team_score'] == 0) & (df['visitor_team_score'] == 0))]
    
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Cleaned dataset has {len(cleaned_df)} rows")
    print(f"Removed {len(df) - len(cleaned_df)} rows with 0 scores")
    
    # Save the cleaned data
    cleaned_df.to_csv('games_normalized_cleaned.csv', index=False)
    print("Saved cleaned data to 'games_normalized_cleaned.csv'")
    
    # Show some statistics about the scores
    print("\nScore statistics after cleaning:")
    print(f"Home team score range: {cleaned_df['home_team_score'].min():.6f} to {cleaned_df['home_team_score'].max():.6f}")
    print(f"Visitor team score range: {cleaned_df['visitor_team_score'].min():.6f} to {cleaned_df['visitor_team_score'].max():.6f}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = clean_games_data() 