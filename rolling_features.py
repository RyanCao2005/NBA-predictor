import pandas as pd
import numpy as np

def calculate_rolling_features_efficient(merged_data, window_size=5):
    """
    Calculate rolling average features for PPG and point differences using efficient vectorized operations.
    
    Args:
        merged_data: DataFrame with game data containing home_team_score, visitor_team_score, 
                    home_team_id, visitor_team_id, and date columns
        window_size: Number of previous games to include in rolling average (default: 5)
    
    Returns:
        DataFrame with added rolling features:
        - home_team_rolling_ppg: Rolling average of home team's points per game
        - visitor_team_rolling_ppg: Rolling average of visitor team's points per game
        - home_team_rolling_point_diff: Rolling average of home team's point differential
        - visitor_team_rolling_point_diff: Rolling average of visitor team's point differential
    """
    print(f"🔄 Calculating rolling averages with {window_size}-game window (efficient method)...")
    
    # Sort by date to ensure chronological order
    merged_data = merged_data.sort_values('date').copy()
    
    # Initialize rolling feature columns
    merged_data['home_team_rolling_ppg'] = 0.0
    merged_data['visitor_team_rolling_ppg'] = 0.0
    merged_data['home_team_rolling_point_diff'] = 0.0
    merged_data['visitor_team_rolling_point_diff'] = 0.0
    
    # Calculate rolling stats for each team using a more direct approach
    for team_id in merged_data['home_team_id'].unique():
        # Home games for this team
        home_mask = merged_data['home_team_id'] == team_id
        home_games = merged_data[home_mask].copy()
        home_games = home_games.sort_values('date')
        
        # Calculate rolling averages for home games
        home_games['home_team_rolling_ppg'] = home_games['home_team_score'].rolling(window=window_size, min_periods=1).mean().shift(1)
        home_games['home_team_rolling_point_diff'] = (home_games['home_team_score'] - home_games['visitor_team_score']).rolling(window=window_size, min_periods=1).mean().shift(1)
        
        # Update the main dataframe
        merged_data.loc[home_games.index, 'home_team_rolling_ppg'] = home_games['home_team_rolling_ppg']
        merged_data.loc[home_games.index, 'home_team_rolling_point_diff'] = home_games['home_team_rolling_point_diff']
        
        # Visitor games for this team
        visitor_mask = merged_data['visitor_team_id'] == team_id
        visitor_games = merged_data[visitor_mask].copy()
        visitor_games = visitor_games.sort_values('date')
        
        # Calculate rolling averages for visitor games
        visitor_games['visitor_team_rolling_ppg'] = visitor_games['visitor_team_score'].rolling(window=window_size, min_periods=1).mean().shift(1)
        visitor_games['visitor_team_rolling_point_diff'] = (visitor_games['visitor_team_score'] - visitor_games['home_team_score']).rolling(window=window_size, min_periods=1).mean().shift(1)
        
        # Update the main dataframe
        merged_data.loc[visitor_games.index, 'visitor_team_rolling_ppg'] = visitor_games['visitor_team_rolling_ppg']
        merged_data.loc[visitor_games.index, 'visitor_team_rolling_point_diff'] = visitor_games['visitor_team_rolling_point_diff']
    
    # Fill NaN values with reasonable defaults
    merged_data['home_team_rolling_ppg'] = merged_data['home_team_rolling_ppg'].fillna(0.5)
    merged_data['visitor_team_rolling_ppg'] = merged_data['visitor_team_rolling_ppg'].fillna(0.5)
    merged_data['home_team_rolling_point_diff'] = merged_data['home_team_rolling_point_diff'].fillna(0.0)
    merged_data['visitor_team_rolling_point_diff'] = merged_data['visitor_team_rolling_point_diff'].fillna(0.0)
    
    print("✅ Rolling features calculated successfully!")
    print(f"📊 Added features:")
    print(f"   - home_team_rolling_ppg (last {window_size} games)")
    print(f"   - visitor_team_rolling_ppg (last {window_size} games)")
    print(f"   - home_team_rolling_point_diff (last {window_size} games)")
    print(f"   - visitor_team_rolling_point_diff (last {window_size} games)")
    
    return merged_data

def calculate_rolling_features_advanced_efficient(merged_data, window_sizes=[3, 5, 10]):
    """
    Calculate rolling average features with multiple window sizes using efficient vectorized operations.
    
    Args:
        merged_data: DataFrame with game data
        window_sizes: List of window sizes to calculate (default: [3, 5, 10])
    
    Returns:
        DataFrame with rolling features for each window size
    """
    print(f"🔄 Calculating rolling averages with multiple windows: {window_sizes} (efficient method)...")
    
    # Sort by date to ensure chronological order
    merged_data = merged_data.sort_values('date').copy()
    
    # Calculate features for each window size
    for window_size in window_sizes:
        print(f"Processing window size: {window_size}")
        
        # Initialize columns for this window size
        merged_data[f'home_team_rolling_ppg_{window_size}'] = 0.0
        merged_data[f'visitor_team_rolling_ppg_{window_size}'] = 0.0
        merged_data[f'home_team_rolling_point_diff_{window_size}'] = 0.0
        merged_data[f'visitor_team_rolling_point_diff_{window_size}'] = 0.0
        
        # Calculate rolling stats for each team
        for team_id in merged_data['home_team_id'].unique():
            # Home games for this team
            home_mask = merged_data['home_team_id'] == team_id
            home_games = merged_data[home_mask].copy()
            home_games = home_games.sort_values('date')
            
            # Calculate rolling features for home games
            home_games[f'home_team_rolling_ppg_{window_size}'] = home_games['home_team_score'].rolling(window=window_size, min_periods=1).mean().shift(1)
            home_games[f'home_team_rolling_point_diff_{window_size}'] = (home_games['home_team_score'] - home_games['visitor_team_score']).rolling(window=window_size, min_periods=1).mean().shift(1)
            
            # Update the main dataframe
            merged_data.loc[home_games.index, f'home_team_rolling_ppg_{window_size}'] = home_games[f'home_team_rolling_ppg_{window_size}']
            merged_data.loc[home_games.index, f'home_team_rolling_point_diff_{window_size}'] = home_games[f'home_team_rolling_point_diff_{window_size}']
            
            # Visitor games for this team
            visitor_mask = merged_data['visitor_team_id'] == team_id
            visitor_games = merged_data[visitor_mask].copy()
            visitor_games = visitor_games.sort_values('date')
            
            # Calculate rolling features for visitor games
            visitor_games[f'visitor_team_rolling_ppg_{window_size}'] = visitor_games['visitor_team_score'].rolling(window=window_size, min_periods=1).mean().shift(1)
            visitor_games[f'visitor_team_rolling_point_diff_{window_size}'] = (visitor_games['visitor_team_score'] - visitor_games['home_team_score']).rolling(window=window_size, min_periods=1).mean().shift(1)
            
            # Update the main dataframe
            merged_data.loc[visitor_games.index, f'visitor_team_rolling_ppg_{window_size}'] = visitor_games[f'visitor_team_rolling_ppg_{window_size}']
            merged_data.loc[visitor_games.index, f'visitor_team_rolling_point_diff_{window_size}'] = visitor_games[f'visitor_team_rolling_point_diff_{window_size}']
        
        # Fill NaN values
        merged_data[f'home_team_rolling_ppg_{window_size}'] = merged_data[f'home_team_rolling_ppg_{window_size}'].fillna(0.5)
        merged_data[f'visitor_team_rolling_ppg_{window_size}'] = merged_data[f'visitor_team_rolling_ppg_{window_size}'].fillna(0.5)
        merged_data[f'home_team_rolling_point_diff_{window_size}'] = merged_data[f'home_team_rolling_point_diff_{window_size}'].fillna(0.0)
        merged_data[f'visitor_team_rolling_point_diff_{window_size}'] = merged_data[f'visitor_team_rolling_point_diff_{window_size}'].fillna(0.0)
    
    print("✅ Advanced rolling features calculated successfully!")
    print(f"📊 Added features for window sizes: {window_sizes}")
    
    return merged_data

# Keep the original functions for backward compatibility
def calculate_rolling_features(merged_data, window_size=5):
    """
    Legacy function - use calculate_rolling_features_efficient instead for better performance.
    """
    print("⚠️ Using legacy function. Consider using calculate_rolling_features_efficient for better performance.")
    return calculate_rolling_features_efficient(merged_data, window_size)

def calculate_rolling_features_advanced(merged_data, window_sizes=[3, 5, 10]):
    """
    Legacy function - use calculate_rolling_features_advanced_efficient instead for better performance.
    """
    print("⚠️ Using legacy function. Consider using calculate_rolling_features_advanced_efficient for better performance.")
    return calculate_rolling_features_advanced_efficient(merged_data, window_sizes)

# Example usage:
if __name__ == "__main__":
    # Load your data
    # merged_data = pd.read_csv("your_data.csv")
    
    # Calculate basic rolling features with 5-game window (efficient method)
    # merged_data = calculate_rolling_features_efficient(merged_data, window_size=5)
    
    # Or calculate advanced rolling features with multiple windows (efficient method)
    # merged_data = calculate_rolling_features_advanced_efficient(merged_data, window_sizes=[3, 5, 10])
    
    print("Rolling features module loaded successfully!") 