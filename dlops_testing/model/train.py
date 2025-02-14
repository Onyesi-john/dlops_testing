import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import mlflow
import mlflow.pytorch
import os

from model import SimpleNN

# Load training data
data_path = "data.csv"

# Check if the file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file {data_path} not found. Please make sure the file exists.")

data = pd.read_csv(data_path)
X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

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
    print("Training complete, model and metrics logged to MLflow.")
