
import torch
from transformer_model import TimeSeriesTransformer

model = TimeSeriesTransformer()

# Load the trained parameters
state_dict = torch.load(
    "trained_transformer.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# Display the parameter names and shapes
for name, parameter in model.named_parameters():
    print(name, parameter.shape)

print("\nModel loaded successfully!")
