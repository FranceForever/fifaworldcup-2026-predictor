#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================
# Part 1: Load and Prepare Data
# ============================
data_file = 'data/final_features_dataset.csv'
if not os.path.exists(data_file):
    print(f"Error: Final dataset not found at {data_file}. Run feature_engineering.py first.")
    exit(1)
df = pd.read_csv(data_file)

# Preserve team identity
team_info = df[['home_team', 'away_team']].copy()

# Create mappings for non-numeric columns (if not already done)
non_numeric_cols = ['home_team', 'away_team', 'tournament']
non_numeric_mappings = {}
for col in non_numeric_cols:
    unique_vals = df[col].unique()
    mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
    non_numeric_mappings[col] = mapping
    df[f"{col}_idx"] = df[col].map(mapping)

with open('non_numeric_mappings.pkl', 'wb') as f:
    pickle.dump(non_numeric_mappings, f)
print("Non-numeric column mappings saved to non_numeric_mappings.pkl")

# Define columns to exclude from features
exclude_cols = ['match_outcome', 'home_team', 'away_team', 'tournament', 'date', 'city', 'country', 'home_last_date', 'away_last_date']
feature_cols = [col for col in df.columns if col not in exclude_cols]
print("Using feature columns:", feature_cols)

try:
    X = df[feature_cols].values.astype(np.float32)
except Exception as e:
    print("Error converting features to float:", e)
    exit(1)

# Create a dummy target if needed (replace with your true target)
def create_dummy_outcome(row):
    if row['ranking_diff'] > 5:
        return 2  # Home win
    elif row['ranking_diff'] < -5:
        return 0  # Away win
    else:
        return 1  # Draw

if 'match_outcome' not in df.columns:
    df['match_outcome'] = df.apply(create_dummy_outcome, axis=1)

y = df['match_outcome'].values.astype(np.int64)
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# ============================
# Part 2: Create PyTorch Dataset and DataLoader
# ============================
class FootballDataset(Dataset):
    def __init__(self, features, targets):
        self.X = features
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = FootballDataset(X, y)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples.")

# ============================
# Part 3: Define the Enhanced Neural Network
# ============================
class FootballNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(FootballNN, self).__init__()
        # First layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Third layer with a residual connection from fc1 output
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Final output layer
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # First layer
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        # Second layer
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        # Third layer with residual connection from x1
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3 + x1[:, :x3.shape[1]])  # residual connection (slice to match dimensions)
        x3 = self.dropout(x3)
        
        output = self.fc4(x3)
        return output

input_dim = X.shape[1]
model = FootballNN(input_dim=input_dim)
print("Neural network defined with input dimension:", input_dim)

# Use CrossEntropyLoss and Adam optimizer with weight decay for regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Setup a learning rate scheduler: reduce LR on plateau of validation loss.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ============================
# Part 4: Training Loop
# ============================
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Training on device:", device)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            count += inputs.size(0)
    return total_loss / count, correct / count

best_val_loss = np.inf
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    train_loss = running_loss / len(train_dataset)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Step the scheduler based on validation loss
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Best model saved at epoch {epoch+1}")

# ============================
# Part 5: Evaluate on Test Set
# ============================
test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), 'final_model.pt')
print("Final model saved to final_model.pt")