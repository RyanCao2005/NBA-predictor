
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assuming you have loaded the player data
player_data = pd.read_csv('actually_normalized_data.csv') 

# Select relevant features and the target statistic you want to predict
selected_features = ['ast', 'blk', 'dreb', 'fg3_pct', 'fg3a', 'fg3m', 'fg_pct', 'fga', 'fgm',
                     'ft_pct', 'fta', 'ftm', 'min', 'oreb', 'pf', 'pts', 'reb', 'stl', 'turnover']
target_statistic = 'pts'  # Change this to the desired statistic

# Filter out rows with missing values in the target statistic
player_data = player_data.dropna(subset=[target_statistic])

# Select features and target
X = player_data[selected_features]
y = player_data[target_statistic]

# Normalize/Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a PyTorch Dataset for player stats prediction
class PlayerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create PyTorch datasets and data loaders
train_dataset = PlayerDataset(X_train, y_train)
test_dataset = PlayerDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple player stats prediction model
class PlayerStatsPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PlayerStatsPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = len(selected_features)
hidden_size = 64
output_size = 1
player_model = PlayerStatsPredictor(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(player_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    player_model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = player_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Evaluate the model on the test set
player_model.eval()
with torch.no_grad():
    all_predictions = []
    all_targets = []
    for inputs, targets in test_loader:
        predictions = player_model(inputs)
        all_predictions.extend(predictions.numpy())
        all_targets.extend(targets.numpy())

# Calculate MSE on the test set
mse = mean_squared_error(all_targets, all_predictions)
print(f'Mean Squared Error on Test Set: {mse:.4f}')

# Predict player stats
def predict_player_stats(player_name, selected_features, player_model):
    player_names = player_name.split()
    player_features = player_data[(player_data['player.first_name'] == player_names[0]) &
                                  (player_data['player.last_name'] == player_names[1])][selected_features].values
    player_features_tensor = torch.tensor(player_features, dtype=torch.float32)
    player_model.eval()
    with torch.no_grad():
        predicted_stat = player_model(player_features_tensor)
    return predicted_stat.numpy()[0][0]

predicted_stats = predict_player_stats('LeBron James', selected_features, player_model)
print(f'Predicted Stats for LeBron James: {predicted_stats:.2f}')