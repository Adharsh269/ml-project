
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from transformer_model import TimeSeriesTransformer


# -----------------------------
# 1. Configuration
# -----------------------------
DATA_FILE = Path(__file__).parent / "synthetic_training_data.txt"

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

torch.manual_seed(42)


# -----------------------------
# 2. Load synthetic data
# -----------------------------

rows = []

with open(DATA_FILE, "r", newline="") as file:
    reader = csv.reader(file)

    for line_number, row in enumerate(reader, start=1):
        # Skip blank lines
        if not row or not any(value.strip() for value in row):
            continue

        # Skip comment/header lines
        if row[0].strip().startswith("#"):
            continue

        try:
            values = [float(value.strip()) for value in row]
        except ValueError as exc:
            raise ValueError(
                f"Invalid number on line {line_number}: {row}"
            ) from exc

        # 30 input values + 1 target value
        if len(values) != 31:
            raise ValueError(
                f"Line {line_number} has {len(values)} values; "
                "expected 31 (30 inputs + 1 target)."
            )

        rows.append(values)


data = torch.tensor(rows, dtype=torch.float32)

# First 30 numbers are inputs; last number is the target
X = data[:, :30].unsqueeze(-1)
y = data[:, 30].unsqueeze(-1)

print("Total samples:", len(data))
print("Input shape:", X.shape)
print("Target shape:", y.shape)


# -----------------------------
# 3. Create training/validation sets
# -----------------------------
dataset = TensorDataset(X, y)

if len(dataset) < 2:
    raise ValueError("Need at least two samples to train and validate.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# 4. Create model and optimizer
# -----------------------------
model = TimeSeriesTransformer()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# 5. Train the Transformer
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for inputs, targets in train_loader:
        # Forward pass: predict the next value
        predictions = model(inputs)

        # Compare predictions with true targets
        loss = criterion(predictions, targets)

        # Backpropagation and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)

    train_loss = total_train_loss / train_size

    # Evaluate on data not used for weight updates
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_val_loss += loss.item() * inputs.size(0)

    val_loss = total_val_loss / val_size

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train MSE: {train_loss:.6f} | "
        f"Validation MSE: {val_loss:.6f}"
    )


# -----------------------------
# 6. Save the trained model
# -----------------------------
model.eval()

torch.save(
    model.state_dict(),
    Path(__file__).parent / "trained_transformer.pth"
)

print("\nTraining complete.")
print("Saved model: trained_transformer.pth")
