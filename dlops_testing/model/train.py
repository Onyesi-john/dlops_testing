import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import mlflow
import mlflow.pytorch

from model import SimpleNN

# Load training data
data_path = "model/data.csv"

# Check if the file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file {data_path} not found. Please make sure the file exists.")

data = pd.read_csv(data_path)

# Print out the first few rows of data to verify it's being loaded
print(f"Data preview:\n{data.head()}")

X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

# Verify shape of input data
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Initialize model
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define example input for logging the model with signature
example_input = X[0].numpy()  # Convert tensor to numpy.ndarray

# Log model with input example for signature
mlflow.set_experiment("dlops_experiment")
with mlflow.start_run():
    # Train model
    for epoch in range(10):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log the loss for each epoch
        mlflow.log_metric("loss", loss.item(), step=epoch)
        
        # Print progress
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
    
    # Log the model and the final loss after training
    mlflow.pytorch.log_model(model, "model", input_example=example_input)
    mlflow.log_metric("final_loss", loss.item())

    # Check logging confirmation
    mlflow.log_param("epochs", 10)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "MSELoss")
    print("Training complete, model and metrics logged to MLflow.")
