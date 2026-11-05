import numpy as np
import torch
import matplotlib.pyplot as plt
import time

class RealisticMonthlySimulator:
    def __init__(self, num_assets=5000, total_months=20000):
        """
        Scaled up for Apple Silicon: 
        Simulates 100 Million data points instantly using vectorized numpy.
        """
        self.num_assets = num_assets
        self.total_months = total_months
        self.timeline = self._generate_quant_matrix()
        
    def _generate_quant_matrix(self):
        print(f"Generating Matrix: {self.num_assets} assets x {self.total_months} months...")
        start_time = time.time()
        
        mu = np.random.uniform(0.002, 0.01, self.num_assets) 
        
        volatility = np.zeros((self.num_assets, self.total_months))
        volatility[:, 0] = np.random.uniform(0.03, 0.08, self.num_assets) 
        
        vol_mean, vol_reversion, vol_noise = 0.05, 0.05, 0.01
        jump_lambda, jump_mean, jump_vol = 0.02, -0.15, 0.10     
        
        log_returns = np.zeros((self.num_assets, self.total_months))
        
        for t in range(1, self.total_months):
            dv = vol_reversion * (vol_mean - volatility[:, t-1]) + np.random.normal(0, vol_noise, self.num_assets)
            volatility[:, t] = np.clip(volatility[:, t-1] + dv, 0.02, 0.20) 
            
            diffusion = np.random.normal(0, volatility[:, t], self.num_assets)
            
            jump_occurs = np.random.poisson(jump_lambda, self.num_assets)
            jump_sizes = jump_occurs * np.random.normal(jump_mean, jump_vol, self.num_assets)
            
            drift = mu - (0.5 * volatility[:, t]**2)
            log_returns[:, t] = drift + diffusion + jump_sizes

        start_prices = np.random.uniform(50.0, 150.0, (self.num_assets, 1))
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        prices = start_prices * np.exp(cumulative_log_returns)
        
        print(f"Matrix generation complete in {time.time() - start_time:.2f} seconds.")
        return prices

    def get_batch(self, batch_size, context_length, prediction_length=1):
        total_window = context_length + prediction_length
        batch_data = []
        
        max_start_idx = self.total_months - total_window
        asset_indices = np.random.randint(0, self.num_assets, size=batch_size)
        time_indices = np.random.randint(0, max_start_idx, size=batch_size)
        
        for i in range(batch_size):
            asset_idx = asset_indices[i]
            t_idx = time_indices[i]
            window = self.timeline[asset_idx, t_idx : t_idx + total_window]
            
            # Normalize
            window_normalized = window / window[0] 
            batch_data.append(window_normalized)
            
        batch_tensor = torch.tensor(np.array(batch_data), dtype=torch.float32)
        return batch_tensor[:, :context_length], batch_tensor[:, context_length:]

# ---------------------------------------------------------
# Execution Block: Generate and Save Massive Dataset
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Build the massive universe (100 million data points)
    simulator = RealisticMonthlySimulator(num_assets=5000, total_months=20000)
    
    # 2. Extract a massive training dataset
    total_samples = 100000
    ctx_len = 30
    filename = "synthetic_quant_data.pt"
    
    print(f"\nSlicing {total_samples} random samples...")
    slice_start = time.time()
    inputs, targets = simulator.get_batch(batch_size=total_samples, context_length=ctx_len)
    print(f"Slicing complete in {time.time() - slice_start:.2f} seconds.")
    
    # 3. Save directly as a PyTorch binary file (Lightning fast on M2)
    print(f"Saving tensors to {filename}...")
    torch.save({
        'inputs': inputs,
        'targets': targets
    }, filename)
    
    print(f"Success! {total_samples} sequences saved to {filename}.")
    # simulator = RealisticMonthlySimulator(num_assets=100, total_months=10000)
    
    # ctx_len = 30 # 30 months = 2.5 years
    # ctx, tgt = simulator.get_batch(batch_size=4, context_length=ctx_len)
    
    # fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    # for i in range(4):
    #     axs[i].plot(range(ctx_len), ctx[i].numpy(), color='darkblue', linewidth=1.5, label="Context (30 Months)")
    #     axs[i].plot([ctx_len], tgt[i].numpy(), marker='X', markersize=8, color='red', label="Target (Month 31)")
    #     axs[i].set_title(f"Quant-Grade Monthly Simulation #{i+1} (Normalized)")
    #     axs[i].grid(True, alpha=0.3)
    #     axs[i].legend(loc="upper left")
        
    # plt.tight_layout()
    # plt.show()