import math
import torch.nn as nn
import torch

# Three observations, each represented by 4 features
x = torch.tensor([
    [
        [1.0, 0.0, 1.0, 0.0],  # Observation 1
        [0.0, 2.0, 0.0, 2.0],  # Observation 2
        [1.0, 1.0, 1.0, 1.0]   # Observation 3
    ]
])

B_S, S_l, D = x.shape

# Learnable projections to create Q, K, and V
W_Q = nn.Linear(D, D, bias=False)
W_K = nn.Linear(D, D, bias=False)
W_V = nn.Linear(D, D, bias=False)

Q = W_Q(x)
K = W_K(x)
V = W_V(x)
# Step 1: Calculate similarity scores
scores = Q @ K.transpose(-2, -1) / math.sqrt(D)

# Step 2: Convert scores into attention weights
attention_weights = torch.softmax(scores, dim=-1)

# Step 3: Combine the value vectors
output = attention_weights @ V

print("Input shape:", x.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("\nAttention scores:\n", scores)
print("\nAttention weights:\n", attention_weights)
print("\nOutput shape:", output.shape)
print("\nOutput:\n", output)