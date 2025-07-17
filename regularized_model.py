# REGULARIZATION TECHNIQUES FOR NBA PREDICTION MODEL

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RegularizedTeamStatsPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with smaller values (L2-like effect)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_with_regularization(X_train, y_train, X_test, y_test, regularization_type='l2'):
    """
    Train model with different regularization techniques
    """
    print(f"Training with {regularization_type.upper()} regularization...")
    
    # Model parameters
    input_size = X_train.shape[1]
    hidden_size = 64  # Smaller than your tuned model
    output_size = 1
    learning_rate = 0.001  # Lower learning rate
    
    # Create model
    if regularization_type == 'dropout':
        model = RegularizedTeamStatsPredictor(input_size, hidden_size, output_size, dropout=0.4)
    else:
        model = RegularizedTeamStatsPredictor(input_size, hidden_size, output_size, dropout=0.2)
    
    # Optimizer with L2 regularization (weight decay)
    if regularization_type == 'l2':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif regularization_type == 'dropout':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:  # No regularization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop with early stopping
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training
        for i in range(0, len(X_train), 32):
            batch_X = torch.FloatTensor(X_train[i:i+32])
            batch_y = torch.FloatTensor(y_train[i:i+32])
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_test), 32):
                batch_X = torch.FloatTensor(X_test[i:i+32])
                batch_y = torch.FloatTensor(y_test[i:i+32])
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = running_loss / (len(X_train) // 32)
        avg_val_loss = val_loss / (len(X_test) // 32)
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate final model
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            batch_X = torch.FloatTensor(X_test[i:i+32])
            batch_y = torch.FloatTensor(y_test[i:i+32])
            
            outputs = model(batch_X)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Final accuracy with {regularization_type.upper()}: {accuracy:.2f}%")
    
    return accuracy, model

# Example usage:
def compare_regularization_techniques():
    """
    Compare different regularization techniques
    """
    print("🔍 COMPARING REGULARIZATION TECHNIQUES")
    print("=" * 50)
    
    # You would use your actual data here
    # X_train, y_train, X_test, y_test = your_data
    
    results = {}
    
    # Test different regularization techniques
    for reg_type in ['none', 'l2', 'dropout']:
        # accuracy, model = train_with_regularization(X_train, y_train, X_test, y_test, reg_type)
        # results[reg_type] = accuracy
        print(f"Would test {reg_type} regularization")
    
    print("\n📊 REGULARIZATION COMPARISON:")
    for reg_type, acc in results.items():
        print(f"{reg_type.upper()}: {acc:.2f}%")

if __name__ == "__main__":
    compare_regularization_techniques() 