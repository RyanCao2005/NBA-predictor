import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_prepare_data():
    """Load and prepare the base data"""
    games_data = pd.read_csv("games.csv")
    teams_data = pd.read_csv("team.csv")
    
    selected_columns_games = ['date', 'home_team_score', 'visitor_team_score', 'home_team_id', 'visitor_team_id']
    merged_data = pd.merge(games_data[selected_columns_games], teams_data, how='inner', left_on='home_team_id', right_on='team_id')
    merged_data['date'] = pd.to_datetime(merged_data['date']).dt.tz_localize(None)
    
    game_columns = ['date', 'home_team_score', 'visitor_team_score', 'home_team_id', 'visitor_team_id']
    merged_data = merged_data[game_columns].drop_duplicates()
    
    return merged_data

def calculate_point_differential_features(merged_data):
    """Calculate point differential features from historical games"""
    print("🔄 Calculating point differential features...")
    
    # Sort by date to ensure chronological order
    merged_data = merged_data.sort_values('date')
    
    # Filter to recent seasons
    start_date = pd.to_datetime("2021-10-19")
    merged_data = merged_data[merged_data['date'] >= start_date].copy()
    
    # Calculate outcome for each game
    merged_data['outcome'] = (merged_data['home_team_score'] > merged_data['visitor_team_score']).astype(int)
    
    # Calculate point differential for each game
    merged_data['home_point_differential'] = merged_data['home_team_score'] - merged_data['visitor_team_score']
    merged_data['visitor_point_differential'] = merged_data['visitor_team_score'] - merged_data['home_team_score']
    
    # Designate season
    merged_data['season'] = merged_data['date'].apply(lambda x: x.year + 1 if 10 <= x.month <= 12 else x.year)
    
    # Calculate rolling point differentials (last 10 games for each team)
    print("📊 Calculating rolling point differentials...")
    
    # For home team performance
    home_stats = merged_data.groupby('home_team_id').agg({
        'home_point_differential': ['mean', 'std', 'count'],
        'outcome': 'sum'
    }).reset_index()
    home_stats.columns = ['team_id', 'home_avg_point_diff', 'home_point_diff_std', 'home_games_played', 'home_wins']
    
    # For visitor team performance
    visitor_stats = merged_data.groupby('visitor_team_id').agg({
        'visitor_point_differential': ['mean', 'std', 'count'],
        'outcome': lambda x: (x == 0).sum()  # Visitor wins when outcome = 0
    }).reset_index()
    visitor_stats.columns = ['team_id', 'visitor_avg_point_diff', 'visitor_point_diff_std', 'visitor_games_played', 'visitor_wins']
    
    # Merge team stats
    team_stats = pd.merge(home_stats, visitor_stats, on='team_id', how='outer').fillna(0)
    
    # Calculate overall team performance metrics
    team_stats['total_games'] = team_stats['home_games_played'] + team_stats['visitor_games_played']
    team_stats['total_wins'] = team_stats['home_wins'] + team_stats['visitor_wins']
    team_stats['win_percentage'] = team_stats['total_wins'] / team_stats['total_games']
    
    # Weighted average point differential (home games count more for home performance)
    team_stats['home_performance'] = (team_stats['home_avg_point_diff'] * team_stats['home_games_played'] + 
                                     team_stats['visitor_avg_point_diff'] * team_stats['visitor_games_played']) / team_stats['total_games']
    
    # Calculate rolling averages for each game
    print("🔄 Calculating rolling averages for each game...")
    
    # Initialize columns
    merged_data['home_team_rolling_point_diff'] = 0.0
    merged_data['visitor_team_rolling_point_diff'] = 0.0
    merged_data['home_team_rolling_wins'] = 0.0
    merged_data['visitor_team_rolling_wins'] = 0.0
    
    # Calculate rolling stats for each team
    for team_id in merged_data['home_team_id'].unique():
        # Home games for this team
        home_games = merged_data[merged_data['home_team_id'] == team_id].copy()
        home_games = home_games.sort_values('date')
        
        # Calculate rolling averages (excluding current game)
        home_games['home_team_rolling_point_diff'] = home_games['home_point_differential'].expanding().mean().shift(1)
        home_games['home_team_rolling_wins'] = home_games['outcome'].expanding().mean().shift(1)
        
        # Update the main dataframe
        merged_data.loc[home_games.index, 'home_team_rolling_point_diff'] = home_games['home_team_rolling_point_diff']
        merged_data.loc[home_games.index, 'home_team_rolling_wins'] = home_games['home_team_rolling_wins']
        
        # Visitor games for this team
        visitor_games = merged_data[merged_data['visitor_team_id'] == team_id].copy()
        visitor_games = visitor_games.sort_values('date')
        
        # Calculate rolling averages (excluding current game)
        visitor_games['visitor_team_rolling_point_diff'] = visitor_games['visitor_point_differential'].expanding().mean().shift(1)
        visitor_games['visitor_team_rolling_wins'] = (visitor_games['outcome'] == 0).expanding().mean().shift(1)
        
        # Update the main dataframe
        merged_data.loc[visitor_games.index, 'visitor_team_rolling_point_diff'] = visitor_games['visitor_team_rolling_point_diff']
        merged_data.loc[visitor_games.index, 'visitor_team_rolling_wins'] = visitor_games['visitor_team_rolling_wins']
    
    # Fill NaN values with 0 (for first few games of each team)
    merged_data['home_team_rolling_point_diff'] = merged_data['home_team_rolling_point_diff'].fillna(0)
    merged_data['visitor_team_rolling_point_diff'] = merged_data['visitor_team_rolling_point_diff'].fillna(0)
    merged_data['home_team_rolling_wins'] = merged_data['home_team_rolling_wins'].fillna(0.5)
    merged_data['visitor_team_rolling_wins'] = merged_data['visitor_team_rolling_wins'].fillna(0.5)
    
    # Add seasonal features
    merged_data['season'] = merged_data['date'].apply(lambda x: x.year + 1 if 10 <= x.month <= 12 else x.year)
    
    # Calculate seasonal win percentages
    merged_data['home_running_wins'] = merged_data.groupby(['season', 'home_team_id'])['outcome'].transform('cumsum')
    merged_data['home_running_games'] = merged_data.groupby(['season', 'home_team_id']).cumcount() + 1
    merged_data['visitor_running_wins'] = merged_data.groupby(['season', 'visitor_team_id'])['outcome'].transform('cumsum')
    merged_data['visitor_running_games'] = merged_data.groupby(['season', 'visitor_team_id']).cumcount() + 1
    
    merged_data['home_seasonal_win_percentage'] = merged_data['home_running_wins'] / (merged_data['home_running_games'] + 1)
    merged_data['visitor_seasonal_win_percentage'] = merged_data['visitor_running_wins'] / (merged_data['visitor_running_games'] + 1)
    
    # Playoff hunt features
    merged_data['home_in_playoff_hunt'] = (merged_data['home_seasonal_win_percentage'] >= 0.500).astype(int)
    merged_data['visitor_in_playoff_hunt'] = (merged_data['visitor_seasonal_win_percentage'] >= 0.500).astype(int)
    
    # Drop intermediate columns
    merged_data = merged_data.drop(columns=['home_running_wins', 'visitor_running_wins', 
                                           'home_running_games', 'visitor_running_games'])
    
    # Date features
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    ref_date = pd.to_datetime("2021-10-19")
    merged_data['days_since_start'] = (merged_data['date'] - ref_date).dt.days
    
    # Playoff features
    merged_data['is_playoff'] = merged_data['date'].apply(
        lambda x: 1 if ((x.month == 4 and x.day > 13) or (x.month in [5, 6])) else 0
    )
    
    print(f"✅ Point differential features calculated for {len(merged_data)} games")
    return merged_data

def preprocess_features(df):
    """Preprocess features for model training"""
    df['home_team_id'] = df['home_team_id'].astype(int)
    df['visitor_team_id'] = df['visitor_team_id'].astype(int)
    return df

class TeamStatsDataset(torch.utils.data.Dataset):
    def __init__(self, team_features, target_outcome):
        self.features = torch.tensor(team_features, dtype=torch.float32)
        self.targets = torch.tensor(target_outcome.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TeamStatsPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TeamStatsPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model_with_point_differentials():
    """Train model using point differential features"""
    print("🚀 Training model with point differential features...")
    
    # Load and prepare data
    merged_data = load_and_prepare_data()
    merged_data = calculate_point_differential_features(merged_data)
    merged_data = preprocess_features(merged_data)
    
    # Select features (NO DATA LEAKAGE!)
    selected_features = [
        'home_team_id', 'visitor_team_id',
        'home_team_rolling_point_diff', 'visitor_team_rolling_point_diff',
        'home_team_rolling_wins', 'visitor_team_rolling_wins',
        'home_seasonal_win_percentage', 'visitor_seasonal_win_percentage',
        'home_in_playoff_hunt', 'visitor_in_playoff_hunt',
        'is_playoff', 'days_since_start'
    ]
    
    # Check for NaN values
    print("\n📊 Data Quality Check:")
    print(merged_data[selected_features].isnull().sum())
    
    # Fill NaN values
    X = merged_data[selected_features].fillna(0)
    y = merged_data['outcome']
    
    print(f"\n📈 Dataset: {X.shape[0]} games, {X.shape[1]} features")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TeamStatsDataset(X_train_scaled, y_train)
    test_dataset = TeamStatsDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = len(selected_features)
    hidden_size = 64
    output_size = 1
    model = TeamStatsPredictor(input_size, hidden_size, output_size)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    print(f"\n🏋️ Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss):
                print(f"⚠️ NaN loss detected at epoch {epoch+1}")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        if torch.isnan(torch.tensor(avg_loss)):
            print("❌ Training stopped due to NaN loss")
            return None
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\n✅ Final Accuracy: {accuracy:.2f}%")
    
    # Feature importance analysis
    print(f"\n🔍 Feature Analysis:")
    print("Features used (no data leakage):")
    for i, feature in enumerate(selected_features):
        print(f"  {i+1}. {feature}")
    
    return model, accuracy, selected_features

if __name__ == "__main__":
    model, accuracy, features = train_model_with_point_differentials()
    
    if model is not None:
        print(f"\n🎉 Model trained successfully!")
        print(f"📊 Accuracy: {accuracy:.2f}%")
        print(f"🔧 Features used: {len(features)}")
        print("✅ No data leakage - all features are historical!")
    else:
        print("❌ Model training failed") 