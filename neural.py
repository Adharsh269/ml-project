import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, output_size=1):
        super(TimeSeriesPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

total_samples = 1000
context_length = 30

# will replace after data proparation is done
X_train = torch.randn(total_samples, context_length) 
y_train = torch.randn(total_samples, 1)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TimeSeriesPredictor(input_size=context_length).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

print("Starting training...")
for epoch in range(epochs):
    model.train() 
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        predictions = model(batch_X)
        
        loss = loss_function(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}] | Average Loss: {avg_loss:.4f}")

print("Training complete!")

torch.save(model.state_dict(), "trained_predictor.pth")
print("Model saved to trained_predictor.pth")