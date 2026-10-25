# Mean Reversion (The "Bounce"):
# Market Meaning: Simulates a stock taking a massive hit, bouncing off a support level, and 
# slowly settling back into a stable range.The Math: A damped harmonic oscillator. 
# You multiply a sine wave by an exponential decay function (e^{-d*x} *sin(kx)). 
# The swings start huge and get progressively smaller.



import numpy as np
import torch_test
import matplotlib.pyplot as plt

def generate_bounce_batch(batch_size, context_length, prediction_length=1):

    total_length = context_length + prediction_length
    x = np.linspace(0, 10, total_length)
    
    batch_data = []
    
    for _ in range(batch_size):
        baseline = np.random.uniform(-1.0, 1.0)     
        noise_level = np.random.uniform(0.1, 0.3)
        
        bounce_start_idx = np.random.randint(int(total_length * 0.1), int(total_length * 0.6))
        
        amplitude = np.random.uniform(1.0, 2.5)     
        decay_rate = np.random.uniform(1.0, 3.0)   
        frequency = np.random.uniform(10.0, 25.0)  
        phase = np.random.uniform(0, np.pi)         
        
        bounce = np.zeros(total_length)
        
        x_shifted = x[bounce_start_idx:] - x[bounce_start_idx]
        
        bounce[bounce_start_idx:] = amplitude * np.exp(-decay_rate * x_shifted) * np.sin(frequency * x_shifted + phase)
        
        noise = np.random.normal(0, noise_level, total_length)
        
        sequence = baseline + bounce + noise
        batch_data.append(sequence)
        
    batch_tensor = torch_test.tensor(np.array(batch_data), dtype=torch_test.float32)
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