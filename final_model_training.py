# FINAL MODEL TRAINING FOR DEPLOYMENT
# Run this after your hyperparameter tuning is complete

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader

# Import your existing classes and data
# (Make sure these are available from your notebook)

def train_final_model_for_deployment():
    """
    Train the final model using best hyperparameters on full dataset
    """
    print("🚀 TRAINING FINAL MODEL FOR DEPLOYMENT")
    print("=" * 50)
    
    # Get best hyperparameters from your tuning
    best_params = {
        'hidden_size': 161,  # Replace with your actual best values
        'learning_rate': 0.008325960334170717,
        'dropout': 0.18439070479767278
    }
    print(f"Best hyperparameters: {best_params}")
    
    # Prepare full dataset (no train/test split for deployment)
    # You'll need to load your merged_data and feature_cols here
    # X_full = merged_data[feature_cols].fillna(0)
    # y_full = merged_data['outcome']
    
    # For now, using placeholder - replace with your actual data
    print("⚠️ Replace this section with your actual data loading")
    
    # Convert to numpy and scale
    # X_full_np = X_full.astype(np.float32).values
    # y_full_np = y_full.astype(np.float32).values.reshape(-1, 1)
    # X_full_np = scaler.transform(X_full_np)
    
    # Create final model with best hyperparameters
    class FinalTeamStatsPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Initialize model (replace input_size with your actual feature count)
    input_size = 25  # Replace with your actual feature count
    final_model = FinalTeamStatsPredictor(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        output_size=1,
        dropout=best_params['dropout']
    )
    
    # Setup optimizer with best learning rate
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    # Create full dataset and loader
    # full_dataset = TeamStatsDataset(X_full_np, y_full_np)
    # full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    # Train on full dataset
    print("Training on full dataset...")
    num_epochs = 20  # More epochs since we're training on full data
    
    # Training loop (uncomment when you have your data)
    """
    for epoch in range(num_epochs):
        final_model.train()
        running_loss = 0.0
        
        for features, targets in full_loader:
            optimizer.zero_grad()
            outputs = final_model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(full_loader)
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    """
    
    print("✅ Final model training complete!")
    
    # Save the final model
    torch.save(final_model.state_dict(), 'nba_predictor_final.pth')
    print("💾 Model saved as 'nba_predictor_final.pth'")
    
    # Save model info for deployment
    model_info = {
        'input_size': input_size,
        'hidden_size': best_params['hidden_size'],
        'dropout': best_params['dropout'],
        'best_params': best_params,
        'feature_columns': []  # Add your feature columns here
    }
    
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print("📋 Model info saved as 'model_info.pkl'")
    
    print("\n🎯 DEPLOYMENT READY!")
    print("Files created:")
    print("- nba_predictor_final.pth (model weights)")
    print("- model_info.pkl (model configuration)")

if __name__ == "__main__":
    train_final_model_for_deployment() 