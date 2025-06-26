import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and prepare data
def load_and_prepare_data():
    # Load datasets
    games_data = pd.read_csv("games.csv")
    teams_data = pd.read_csv("team.csv")
    
    # Define columns to be selected
    selected_columns_games = ['date', 'home_team_score', 'visitor_team_score', 'home_team_id', 'visitor_team_id']
    
    # Merge datasets
    merged_data = pd.merge(games_data[selected_columns_games], teams_data, how='inner', left_on='home_team_id', right_on='team_id')
    
    # Ensure 'date' column is in datetime format and sort by date
    merged_data['date'] = pd.to_datetime(merged_data['date']).dt.tz_localize(None)
    
    # Create a new dataframe with unique games
    game_columns = ['date', 'home_team_score', 'visitor_team_score', 'home_team_id', 'visitor_team_id']
    merged_data = merged_data[game_columns].drop_duplicates()
    
    return merged_data

def calculate_date_features(merged_data):
    # Ensure 'date' column is in datetime format and sort by date
    merged_data = merged_data.sort_values(by='date')
    
    # Filter out games prior to 2021-10-19
    start_date = pd.to_datetime("2021-10-19")
    merged_data = merged_data[merged_data['date'] >= start_date]
    
    # Outcome column where 1 = home win, 0 = visitor win
    merged_data['outcome'] = (merged_data['home_team_score'] > merged_data['visitor_team_score']).astype(int)
    
    # Designate a season for each game being played
    merged_data['season'] = merged_data['date'].apply(lambda x: x.year + 1 if 10 <= x.month <= 12 else x.year)
    
    # Group by home and visitor team IDs for efficiency
    merged_data['home_running_wins'] = merged_data.groupby(['season', 'home_team_id'])['outcome'].transform('cumsum')
    merged_data['home_running_games'] = merged_data.groupby(['season', 'home_team_id']).cumcount() + 1
    
    merged_data['visitor_running_wins'] = merged_data.groupby(['season', 'visitor_team_id'])['outcome'].transform('cumsum')
    merged_data['visitor_running_games'] = merged_data.groupby(['season', 'visitor_team_id']).cumcount() + 1
    
    # Compute win percentages
    merged_data['home_seasonal_win_percentage'] = merged_data['home_running_wins'] / (merged_data['home_running_games'] + 1)
    merged_data['visitor_seasonal_win_percentage'] = merged_data['visitor_running_wins'] / (merged_data['visitor_running_games'] + 1)
    
    # Playoff hunt: 1 if win percentage >= 50%, else 0
    merged_data['home_in_playoff_hunt'] = (merged_data['home_seasonal_win_percentage'] >= 0.500).astype(int)
    merged_data['visitor_in_playoff_hunt'] = (merged_data['visitor_seasonal_win_percentage'] >= 0.500).astype(int)
    
    # Drop intermediate cumulative columns
    merged_data = merged_data.drop(columns=['home_running_wins', 'visitor_running_wins', 
                                           'home_running_games', 'visitor_running_games'])
    
    # Compute season end date dynamically
    merged_data['season_end_date'] = merged_data['date'].apply(
        lambda x: pd.to_datetime(f"{x.year + 1}-04-15") if x.month >= 10 else pd.to_datetime(f"{x.year}-04-15")
    )
    
    # Define playoff games: only games after April 13 are considered playoff games
    merged_data['is_playoff'] = merged_data['date'].apply(
        lambda x: 1 if ((x.month == 4 and x.day > 13) or (x.month in [5, 6])) else 0
    )
    
    # Calculate days remaining in the season for non-playoff games only
    merged_data['days_remaining'] = np.where(
        merged_data['is_playoff'] == 0,
        (merged_data['season_end_date'] - merged_data['date']).dt.days,
        np.nan
    )
    
    # Compute late season weight based on days remaining
    merged_data['late_season_weight'] = np.maximum(0, 1 - (merged_data['days_remaining'] / 365))
    
    # Create separate late playoff weight features for home and visitor teams
    merged_data['home_late_playoff_weight'] = merged_data['late_season_weight'] * merged_data['home_in_playoff_hunt']
    merged_data['visitor_late_playoff_weight'] = merged_data['late_season_weight'] * merged_data['visitor_in_playoff_hunt']
    
    return merged_data

def preprocess_features(df):
    # Convert categorical features to numeric
    df['home_team_id'] = df['home_team_id'].astype(int)
    df['visitor_team_id'] = df['visitor_team_id'].astype(int)
    
    # Convert the 'date' column to numeric
    df['date'] = pd.to_datetime(df['date'])
    ref_date = pd.to_datetime("2021-10-19")
    df['days_since_start'] = (df['date'] - ref_date).dt.days
    
    return df

# Dataset class
class TeamStatsDataset(torch.utils.data.Dataset):
    def __init__(self, team_features, target_outcome):
        self.features = torch.tensor(team_features, dtype=torch.float32)
        self.targets = torch.tensor(target_outcome.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Fixed model class (removed sigmoid activation)
class TeamStatsPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TeamStatsPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # No sigmoid here - BCEWithLogitsLoss will handle it
        return x

def train_model():
    # Load and prepare data
    merged_data = load_and_prepare_data()
    merged_data = calculate_date_features(merged_data)
    merged_data = preprocess_features(merged_data)
    
    # FIXED: Remove data leakage features (home_team_score and visitor_team_score)
    selected_features = [
        'home_team_id', 'visitor_team_id', 
        'home_seasonal_win_percentage', 'visitor_seasonal_win_percentage', 
        'home_in_playoff_hunt', 'visitor_in_playoff_hunt', 'is_playoff', 
        'days_remaining', 'late_season_weight', 'home_late_playoff_weight', 'visitor_late_playoff_weight',
        'days_since_start'
    ]
    
    # Check for NaN values and handle them
    print("Checking for NaN values in features:")
    print(merged_data[selected_features].isnull().sum())
    
    # Fill NaN values with 0 for numeric features
    X = merged_data[selected_features].fillna(0)
    y = merged_data['outcome']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert the data into PyTorch Datasets
    train_dataset = TeamStatsDataset(X_train_scaled, y_train)
    test_dataset = TeamStatsDataset(X_test_scaled, y_test)
    
    # Create DataLoader for batching during model training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    input_size = len(selected_features)
    hidden_size = 64
    output_size = 1
    model = TeamStatsPredictor(input_size, hidden_size, output_size)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # This includes sigmoid internally
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            
            # Compute the loss
            loss = criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_count}!")
                print(f"Features shape: {features.shape}")
                print(f"Features range: {features.min():.4f} to {features.max():.4f}")
                print(f"Targets: {targets}")
                print(f"Outputs: {outputs}")
                return None
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        # Early stopping if loss is still NaN
        if torch.isnan(torch.tensor(avg_loss)):
            print("Training stopped due to NaN loss")
            return None
    
    return model, test_loader, scaler

if __name__ == "__main__":
    model, test_loader, scaler = train_model()
    
    if model is not None:
        # Evaluate the model
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
        print(f'Accuracy on test data: {accuracy:.2f}%')
    else:
        print("Model training failed due to NaN loss") 