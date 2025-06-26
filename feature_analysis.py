import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data (reusing your existing functions)
def load_and_prepare_data():
    games_data = pd.read_csv("games.csv")
    teams_data = pd.read_csv("team.csv")
    
    selected_columns_games = ['date', 'home_team_score', 'visitor_team_score', 'home_team_id', 'visitor_team_id']
    merged_data = pd.merge(games_data[selected_columns_games], teams_data, how='inner', left_on='home_team_id', right_on='team_id')
    merged_data['date'] = pd.to_datetime(merged_data['date']).dt.tz_localize(None)
    
    game_columns = ['date', 'home_team_score', 'visitor_team_score', 'home_team_id', 'visitor_team_id']
    merged_data = merged_data[game_columns].drop_duplicates()
    
    return merged_data

def calculate_date_features(merged_data):
    merged_data = merged_data.sort_values(by='date')
    start_date = pd.to_datetime("2021-10-19")
    merged_data = merged_data[merged_data['date'] >= start_date]
    
    merged_data['outcome'] = (merged_data['home_team_score'] > merged_data['visitor_team_score']).astype(int)
    merged_data['season'] = merged_data['date'].apply(lambda x: x.year + 1 if 10 <= x.month <= 12 else x.year)
    
    merged_data['home_running_wins'] = merged_data.groupby(['season', 'home_team_id'])['outcome'].transform('cumsum')
    merged_data['home_running_games'] = merged_data.groupby(['season', 'home_team_id']).cumcount() + 1
    merged_data['visitor_running_wins'] = merged_data.groupby(['season', 'visitor_team_id'])['outcome'].transform('cumsum')
    merged_data['visitor_running_games'] = merged_data.groupby(['season', 'visitor_team_id']).cumcount() + 1
    
    merged_data['home_seasonal_win_percentage'] = merged_data['home_running_wins'] / (merged_data['home_running_games'] + 1)
    merged_data['visitor_seasonal_win_percentage'] = merged_data['visitor_running_wins'] / (merged_data['visitor_running_games'] + 1)
    
    merged_data['home_in_playoff_hunt'] = (merged_data['home_seasonal_win_percentage'] >= 0.500).astype(int)
    merged_data['visitor_in_playoff_hunt'] = (merged_data['visitor_seasonal_win_percentage'] >= 0.500).astype(int)
    
    merged_data = merged_data.drop(columns=['home_running_wins', 'visitor_running_wins', 
                                           'home_running_games', 'visitor_running_games'])
    
    merged_data['season_end_date'] = merged_data['date'].apply(
        lambda x: pd.to_datetime(f"{x.year + 1}-04-15") if x.month >= 10 else pd.to_datetime(f"{x.year}-04-15")
    )
    
    merged_data['is_playoff'] = merged_data['date'].apply(
        lambda x: 1 if ((x.month == 4 and x.day > 13) or (x.month in [5, 6])) else 0
    )
    
    merged_data['days_remaining'] = np.where(
        merged_data['is_playoff'] == 0,
        (merged_data['season_end_date'] - merged_data['date']).dt.days,
        np.nan
    )
    
    merged_data['late_season_weight'] = np.maximum(0, 1 - (merged_data['days_remaining'] / 365))
    merged_data['home_late_playoff_weight'] = merged_data['late_season_weight'] * merged_data['home_in_playoff_hunt']
    merged_data['visitor_late_playoff_weight'] = merged_data['late_season_weight'] * merged_data['visitor_in_playoff_hunt']
    
    return merged_data

def preprocess_features(df):
    df['home_team_id'] = df['home_team_id'].astype(int)
    df['visitor_team_id'] = df['visitor_team_id'].astype(int)
    df['date'] = pd.to_datetime(df['date'])
    ref_date = pd.to_datetime("2021-10-19")
    df['days_since_start'] = (df['date'] - ref_date).dt.days
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

def analyze_features():
    print("🔍 FEATURE ANALYSIS FOR NBA GAME PREDICTION")
    print("=" * 50)
    
    # Load and prepare data
    merged_data = load_and_prepare_data()
    merged_data = calculate_date_features(merged_data)
    merged_data = preprocess_features(merged_data)
    
    # Current features
    selected_features = [
        'home_team_id', 'visitor_team_id', 
        'home_seasonal_win_percentage', 'visitor_seasonal_win_percentage', 
        'home_in_playoff_hunt', 'visitor_in_playoff_hunt', 'is_playoff', 
        'days_remaining', 'late_season_weight', 'home_late_playoff_weight', 'visitor_late_playoff_weight',
        'days_since_start'
    ]
    
    X = merged_data[selected_features].fillna(0)
    y = merged_data['outcome']
    
    print(f"📊 Dataset: {X.shape[0]} games, {X.shape[1]} features")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    print()
    
    # 1. CORRELATION ANALYSIS
    print("1️⃣ CORRELATION ANALYSIS")
    print("-" * 30)
    
    correlation_matrix = X.corr()
    print("Feature correlations with target:")
    target_correlations = correlation_matrix.abs().sort_values(ascending=False)
    for feature in selected_features:
        if feature in target_correlations.index:
            corr = target_correlations[feature]
            print(f"  {feature}: {corr:.4f}")
    
    # 2. RANDOM FOREST FEATURE IMPORTANCE
    print("\n2️⃣ RANDOM FOREST FEATURE IMPORTANCE")
    print("-" * 40)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature importance (Random Forest):")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 3. PCA ANALYSIS
    print("\n3️⃣ PCA ANALYSIS")
    print("-" * 20)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # Explained variance ratio
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print("Explained variance by components:")
    for i, (var, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
        print(f"  PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)")
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_components_95}")
    
    # 4. MODEL PERFORMANCE WITH DIFFERENT FEATURE SETS
    print("\n4️⃣ MODEL PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different feature sets
    feature_sets = {
        'All Features': selected_features,
        'Top 5 Important': feature_importance.head(5)['feature'].tolist(),
        'Top 3 Important': feature_importance.head(3)['feature'].tolist(),
        'High Correlation (>0.1)': [f for f in selected_features if abs(X[f].corr(y)) > 0.1]
    }
    
    results = {}
    
    for set_name, features in feature_sets.items():
        if len(features) == 0:
            continue
            
        print(f"\nTesting: {set_name} ({len(features)} features)")
        
        # Prepare data
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]
        
        # Scale
        scaler_subset = StandardScaler()
        X_train_scaled = scaler_subset.fit_transform(X_train_subset)
        X_test_scaled = scaler_subset.transform(X_test_subset)
        
        # Create datasets
        train_dataset = TeamStatsDataset(X_train_scaled, y_train)
        test_dataset = TeamStatsDataset(X_test_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train model
        model = TeamStatsPredictor(len(features), 64, 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Quick training (5 epochs for comparison)
        for epoch in range(5):
            model.train()
            for features_batch, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(features_batch)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                predictions.extend(predicted.numpy())
        
        accuracy = accuracy_score(y_test, predictions)
        results[set_name] = accuracy
        print(f"  Accuracy: {accuracy:.4f}")
    
    # 5. RECOMMENDATIONS
    print("\n5️⃣ RECOMMENDATIONS")
    print("-" * 20)
    
    print("Based on the analysis:")
    
    # Best performing feature set
    best_set = max(results, key=results.get)
    print(f"✅ Best performing feature set: {best_set} ({results[best_set]:.4f} accuracy)")
    
    # Most important features
    top_features = feature_importance.head(3)['feature'].tolist()
    print(f"🎯 Most important features: {', '.join(top_features)}")
    
    # Correlation insights
    high_corr_features = [f for f in selected_features if abs(X[f].corr(y)) > 0.1]
    print(f"📈 Features with high correlation to target: {', '.join(high_corr_features)}")
    
    print("\n💡 NEXT STEPS:")
    print("1. Focus on the top 3-5 most important features")
    print("2. Consider adding features related to:")
    print("   - Head-to-head history between teams")
    print("   - Rest days between games")
    print("   - Home/away performance splits")
    print("   - Recent form (last 5-10 games)")
    print("3. Test the model with the best feature set before adding more")
    
    return results, feature_importance, pca

if __name__ == "__main__":
    results, feature_importance, pca = analyze_features() 