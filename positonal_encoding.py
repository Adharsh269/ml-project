import torch
import torch.nn as nn
import math

BATCH_SIZE = 32
SEQUENCE_LENGTH = 30
DIMENSIONS_MODEL = 64
HEADS = 4
LAYERS = 2

#Creating sample input
x = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIMENSIONS_MODEL)

#having the multi head attention
#mha = nn.MultiheadAttention(embed_dim=DIMENSIONS_MODEL, num_heads=HEADS, batch_first= True)
# output, attention_weights = mha(
#     query = x,
#     key = x,
#     value = x,
#     need_weights = True,
#     average_attn_weights = False
# )

#transformers encoding
encoder_layer = nn.TransformerEncoderLayer(
    d_model=DIMENSIONS_MODEL,
    nhead=HEADS,
    dim_feedforward=128,
    batch_first=True
)
#second encoder layer
transformer_encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=LAYERS
)
output = transformer_encoder(x)

print("Input shape:", x.shape)
print("Number of encoder layers:", LAYERS)
print("Output shape:", output.shape)
print("First observation representation:", output[0, 0])

# #Converting each observation to 64 features
# input_projection = nn.Linear(1, DIMENSIONS_MODEL)
# z = input_projection(x)

# #positional encoding
# position = torch.arange(SEQUENCE_LENGTH).unsqueeze(1)

# div_term = torch.exp(torch.arange(0, DIMENSIONS_MODEL, 2) * (-math.log(1000.0) / DIMENSIONS_MODEL))

# pe = torch.zeros(SEQUENCE_LENGTH, DIMENSIONS_MODEL)
# pe[:, 0::2] = torch.sin(position  * div_term)
# pe[:, 1::2] = torch.cos(position * div_term)

# #adding positional encoding to the input
# z = z +pe.unsqueeze(0)

# print("Input shape:", x.shape)
# print("After projection:", input_projection(x).shape)
# print("Positional encoding shape:", pe.shape)
# print("After positional encoding:", z.shape)