# Sudden Shocks (Gaps, Drops, and Flash Crashes):
# Market Meaning: Simulates overnight earnings reports, sudden news drops, or flash crashes.
# The Math: A Heaviside step function. You pick a random time index t and suddenly add or 
# ubtract a massive constant value for all points after t.


import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_shock_batch(batch_size, context_length, prediction_length=1):
    """
    Generates time series windows featuring sudden market shocks (Heaviside step function).
    """
    total_length = context_length + prediction_length
    x = np.linspace(0, 10, total_length)
    
    batch_data = []
    
    for _ in range(batch_size):
        # 1. Base market drift and noise (no complex waves here, just trend and noise)
        trend_slope = np.random.uniform(-0.3, 0.3)
        noise_level = np.random.uniform(0.1, 0.3)
        
        trend = trend_slope * x
        noise = np.random.normal(0, noise_level, total_length)
        
        # 2. The Sudden Shock
        shock = np.zeros(total_length)
        shock_index = np.random.randint(int(total_length * 0.2), int(total_length * 0.8))
        shock_magnitude = np.random.uniform(2.0, 5.0)
        shock_direction = np.random.choice([-1, 1]) 
        
        # Apply the step function
        shock[shock_index:] = shock_magnitude * shock_direction
        
        sequence = trend + noise + shock
        batch_data.append(sequence)
        
    batch_tensor = torch.tensor(np.array(batch_data), dtype=torch.float32)
    context = batch_tensor[:, :context_length]
    target = batch_tensor[:, context_length:]
    
    return context, target

if __name__ == "__main__":
    ctx, tgt = generate_shock_batch(batch_size=1, context_length=30)