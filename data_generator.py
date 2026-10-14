import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_synthetic_batch(batch_size, context_length, prediction_length=1):
    """
    Generates synthetic time series windows for in-context learning.
    Mixes sine waves, linear trends, and noise to simulate market movements.
    """
    total_length = context_length + prediction_length
    x = np.linspace(0, 10, total_length)
    
    batch_data = []
    
    for _ in range(batch_size):
        # Randomize the parameters for each sequence so the model learns the "math", not the specific wave
        amplitude = np.random.uniform(0.5, 2.0)
        frequency = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 2 * np.pi)
        trend_slope = np.random.uniform(-0.5, 0.5)
        noise_level = np.random.uniform(0.1, 0.5)
        
        # Combine wave, trend, and noise
        wave = amplitude * np.sin(frequency * x + phase)
        trend = trend_slope * x
        noise = np.random.normal(0, noise_level, total_length)
        
        sequence = wave + trend + noise
        batch_data.append(sequence)
        
    # Convert to PyTorch tensors
    batch_tensor = torch.tensor(np.array(batch_data), dtype=torch.float32)
    
    # Split into context window (input) and target (label)
    context = batch_tensor[:, :context_length]
    target = batch_tensor[:, context_length:]
    
    return context, target

# --- Test the Generator ---
if __name__ == "__main__":
    context_len = 30
    
    # Generate 1 sequence
    ctx, tgt = generate_synthetic_batch(batch_size=1, context_length=context_len)
    
    # Plot the result to visualize what the model will see
    plt.figure(figsize=(10, 4))
    plt.plot(range(context_len), ctx[0].numpy(), label="Context Window (Input)", color='blue')
    plt.plot([context_len], tgt[0].numpy(), marker='o', color='red', label="Target (To Predict)")
    plt.title("Sample Synthetic Sequence for ICL Training")
    plt.legend()
    plt.show()