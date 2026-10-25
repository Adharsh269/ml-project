
import csv
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, random_split
from transformer_model import TimeSeriesTransformer

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "synthetic_training_data.txt"
MODEL_FILE = BASE_DIR / "trained_transformer.pth"

# 1. Load the synthetic dataset
rows = []

with open(DATA_FILE, "r", newline="") as file:
    for row in csv.reader(file):
        if not row or not any(v.strip() for v in row):
            continue
        if row[0].strip().startswith("#"):
            continue

        values = [float(v.strip()) for v in row]

        if len(values) != 31:
            raise ValueError("Each data row must contain 31 values.")

        rows.append(values)

data = torch.tensor(rows, dtype=torch.float32)
X = data[:, :30].unsqueeze(-1)
y = data[:, 30].unsqueeze(-1)

dataset = TensorDataset(X, y)

# 2. Recreate the same validation split used during training
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 3. Load the trained model
model = TimeSeriesTransformer()
state_dict = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()


# 4. Evaluate all validation examples
transformer_errors = []
naive_errors = []
transformer_squared_errors = []
naive_squared_errors = []

with torch.no_grad():
    for inputs, actual in val_dataset:
        inputs = inputs.unsqueeze(0)
        actual_value = actual.item()

        prediction = model(inputs).item()
        naive_prediction = inputs[0, -1, 0].item()

        transformer_error = abs(prediction - actual_value)
        naive_error = abs(naive_prediction - actual_value)

        transformer_errors.append(transformer_error)
        naive_errors.append(naive_error)

        transformer_squared_errors.append(transformer_error ** 2)
        naive_squared_errors.append(naive_error ** 2)

n = len(val_dataset)

print(f"Validation samples: {n}")
print(f"\nTransformer MAE: "
      f"{sum(transformer_errors) / n:.4f}")
print(f"Naive MAE:       "
      f"{sum(naive_errors) / n:.4f}")

print(f"\nTransformer MSE: "
      f"{sum(transformer_squared_errors) / n:.4f}")
print(f"Naive MSE:       "
      f"{sum(naive_squared_errors) / n:.4f}")


# 5. Naive baseline: predict the last observed value
naive_prediction = inputs[0, -1, 0].item()

print("Last 5 context observations:")
print(inputs[0, -5:, 0].tolist())

print(f"\nActual next value:       {actual_value:.4f}")
print(f"Transformer prediction:  {prediction:.4f}")
print(f"Naive prediction:        {naive_prediction:.4f}")

print(f"\nTransformer absolute error: "
      f"{abs(actual_value - prediction):.4f}")
print(f"Naive absolute error:       "
      f"{abs(actual_value - naive_prediction):.4f}")
