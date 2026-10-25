import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, sequence_length = 30, d_model = 64, num_heads = 4, num_layers =2, dim_feedforward = 128):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

        #converting each numerical data to 64 features
        self.input_projection = nn.Linear(1, d_model)

        #doing the position embeddings
        position = torch.arange(sequence_length).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(sequence_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store positional encoding as a non-trainable buffer
        self.register_buffer("positional_encoding", pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.prediction_head = nn.Linear(d_model, 1)

    def forward(self, x) :
        x = self.input_projection(x)
        x = x + self.positional_encoding
        x= self.transformer_encoder(x)
        x = x[:, -1, :]
        prediction = self.prediction_head(x)
        return prediction
if __name__ == "__main__":
    print("hi")
    batch_size = 32
    model = TimeSeriesTransformer()
    x = torch.randn(batch_size, 30, 1)
    prediction = model(x)
    print("Input shape:", x.shape)
    print("Prediction shape:", prediction.shape)
    print("First 5 predictions:")
    print(prediction[:5].detach())

    print("Total trainable parameters:",sum(p.numel() for p in model.parameters()if p.requires_grad))