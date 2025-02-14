import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import mlflow
import mlflow.pytorch

from model import SimpleNN

# Load training data
data = pd.read_csv("data.csv")
X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

# Initialize model
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

# Log to MLflow
mlflow.set_experiment("dlops_experiment")
with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_metric("loss", loss.item())
