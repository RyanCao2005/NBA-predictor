# Copy this entire code block into a new cell in your BinaryClassifier.ipynb notebook

# FIXED VERSION: Corrected model training to prevent NaN loss
# List of selected features (REMOVED home_team_score and visitor_team_score to prevent data leakage)
selected_features = [
    'home_team_id', 'visitor_team_id', 
    'home_seasonal_win_percentage', 'visitor_seasonal_win_percentage', 
    'home_in_playoff_hunt', 'visitor_in_playoff_hunt', 'is_playoff', 
    'days_remaining', 'late_season_weight', 'home_late_playoff_weight', 'visitor_late_playoff_weight',
    'days_since_start'  # Added the date feature we created
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

# FIXED: Remove sigmoid from model since BCEWithLogitsLoss includes it
class TeamStatsPredictorFixed(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TeamStatsPredictorFixed, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Remove sigmoid - BCEWithLogitsLoss will handle it
        return x

# Initialize the model
input_size = len(selected_features)  # Number of input features
hidden_size = 64  # Choose a hidden size (adjustable)
output_size = 1  # Binary outcome (win or loss)
model = TeamStatsPredictorFixed(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # This includes sigmoid internally
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with NaN detection
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    batch_count = 0

    for features, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(features)  # Forward pass
        
        # Compute the loss
        loss = criterion(outputs, targets)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch+1}, batch {batch_count}!")
            print(f"Features shape: {features.shape}")
            print(f"Features range: {features.min():.4f} to {features.max():.4f}")
            print(f"Targets: {targets}")
            print(f"Outputs: {outputs}")
            break
        
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
        break

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