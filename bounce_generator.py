# Mean Reversion (The "Bounce"):
# Market Meaning: Simulates a stock taking a massive hit, bouncing off a support level, and 
# slowly settling back into a stable range.The Math: A damped harmonic oscillator. 
# You multiply a sine wave by an exponential decay function (e^{-d*x} *sin(kx)). 
# The swings start huge and get progressively smaller.



import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_bounce_batch(batch_size, context_length, prediction_length=1):
    """
    Generates time series windows featuring mean reversion (Damped Harmonic Oscillator)
    that can trigger at a random point in the sequence.
    """
    total_length = context_length + prediction_length
    x = np.linspace(0, 10, total_length)
    
    batch_data = []
    
    for _ in range(batch_size):
        # 1. Base parameters
        baseline = np.random.uniform(-1.0, 1.0)     
        noise_level = np.random.uniform(0.1, 0.3)
        
        # 2. Pick a random start time for the bounce shock
        bounce_start_idx = np.random.randint(int(total_length * 0.1), int(total_length * 0.6))
        
        # 3. ADJUSTED: Realistic Damped Oscillator Parameters
        amplitude = np.random.uniform(1.0, 2.5)     # Reduced: Smaller, more realistic initial shock
        decay_rate = np.random.uniform(1.0, 3.0)    # Increased: Faster settling time
        frequency = np.random.uniform(10.0, 25.0)   # Drastically Increased: Rapid, tight panic swings
        phase = np.random.uniform(0, np.pi)         
        
        # 4. The Math: Apply the bounce ONLY after the start index
        bounce = np.zeros(total_length)
        
        # We must shift the time axis for the bounce so the math (t=0) starts right at the shock
        x_shifted = x[bounce_start_idx:] - x[bounce_start_idx]
        
        # Apply exponential decay to the sine wave using the shifted time
        bounce[bounce_start_idx:] = amplitude * np.exp(-decay_rate * x_shifted) * np.sin(frequency * x_shifted + phase)
        
        noise = np.random.normal(0, noise_level, total_length)
        
        sequence = baseline + bounce + noise
        batch_data.append(sequence)
        
    batch_tensor = torch.tensor(np.array(batch_data), dtype=torch.float32)
    context = batch_tensor[:, :context_length]
    target = batch_tensor[:, context_length:]
    
    return context, target

if __name__ == "__main__":
    ctx, tgt = generate_bounce_batch(batch_size=1, context_length=30)
    
    # plt.figure(figsize=(10, 4))
    # plt.plot(range(30), ctx[0].numpy(), label="Context (Bounce)", color='purple')
    # plt.plot([30], tgt[0].numpy(), marker='o', color='red', label="Target")
    # plt.title("Isolated Pattern: Mean Reversion Starting at Random Index")
    # plt.legend()
    # plt.show()