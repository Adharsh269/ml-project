import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_synthetic_batch(batch_size, context_length, prediction_length=1):
    total_length = context_length + prediction_length
    x = np.linspace(0, 10, total_length)
    
    batch_data = []
    
    for _ in range(batch_size):
        # Randomize the parameters for each sequence
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

if __name__ == "__main__":
    total_samples = 100  
    context_len = 30
    filename = "synthetic_training_data.txt"
    
    print(f"Generating {total_samples} synthetic sequences...")
    
    inputs, targets = generate_synthetic_batch(batch_size=total_samples, context_length=context_len)
    
    inputs_np = inputs.numpy()
    targets_np = targets.numpy()
    
    combined_data = np.hstack((inputs_np, targets_np))
    
    header_text = f"Columns 1 to {context_len}: Context Window (Input), Column {context_len + 1}: Target (To Predict)"
    np.savetxt(filename, combined_data, delimiter=",", fmt="%.4f", header=header_text)
    
    print(f"Success! Data saved to: {filename}")
    
    # plt.figure(figsize=(10, 4))
    # plt.plot(range(context_len), inputs[0].numpy(), label="Context Window (Input)", color='blue')
    # plt.plot([context_len], targets[0].numpy(), marker='o', color='red', label="Target (To Predict)")
    # plt.title("Sample #1 from Saved Text File Dataset")
    # plt.legend()
    # plt.show()