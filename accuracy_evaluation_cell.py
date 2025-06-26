# EVALUATE MODEL ACCURACY
print("🔍 MODEL EVALUATION")
print("=" * 40)

# Set model to evaluation mode
model.eval()

# Initialize counters
correct = 0
total = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Evaluate on test data
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Calculate confusion matrix components
        for i in range(len(targets)):
            if targets[i] == 1 and predicted[i] == 1:
                true_positives += 1
            elif targets[i] == 0 and predicted[i] == 1:
                false_positives += 1
            elif targets[i] == 1 and predicted[i] == 0:
                false_negatives += 1
            elif targets[i] == 0 and predicted[i] == 0:
                true_negatives += 1

# Calculate metrics
accuracy = 100 * correct / total
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Display results
print(f"📊 ACCURACY: {accuracy:.2f}%")
print(f"🎯 PRECISION: {precision:.4f}")
print(f"📈 RECALL: {recall:.4f}")
print(f"⚖️ F1-SCORE: {f1_score:.4f}")
print()
print("📋 CONFUSION MATRIX:")
print(f"True Positives (Home Win Predicted Correctly): {true_positives}")
print(f"False Positives (Home Win Predicted Incorrectly): {false_positives}")
print(f"False Negatives (Away Win Predicted Incorrectly): {false_negatives}")
print(f"True Negatives (Away Win Predicted Correctly): {true_negatives}")
print()
print(f"📈 PREDICTION BREAKDOWN:")
print(f"Total Predictions: {total}")
print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {total - correct}")
print(f"Home Win Predictions: {true_positives + false_positives}")
print(f"Away Win Predictions: {true_negatives + false_negatives}")

# Optional: Save results to a variable for further analysis
evaluation_results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'confusion_matrix': {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }
}

print(f"\n✅ Evaluation complete! Model accuracy: {accuracy:.2f}%") 