import numpy as np
import torch
import time
import concurrent.futures
from fbm import FBM

def generate_single_fbm_path(total_months):
    """
    Generates a single multi-decade stock path using FBM.
    """
    # 1. Randomly assign this specific stock a "personality"
    hurst_exponent = np.random.uniform(0.3, 0.75) 
    
    # 2. Generate the fractional path using Davies-Harte (FFT-based math)
    f = FBM(n=total_months - 1, hurst=hurst_exponent, length=1, method='daviesharte')
    fractional_noise = f.fbm()
    
    # 3. Scale it to realistic monthly stock movements
    volatility_scalar = np.random.uniform(0.05, 0.15)
    log_returns = fractional_noise * volatility_scalar
    
    # 4. Add a slight long-term drift (0.2% to 0.8% monthly)
    drift = np.linspace(0, np.random.uniform(0.002, 0.008) * total_months, total_months)
    
    # 5. Convert to exponentially compounding prices
    start_price = np.random.uniform(50.0, 150.0)
    
    return start_price * np.exp(log_returns + drift)


class FractionalMarketSimulator:
    def __init__(self, num_assets=5000, total_months=20000):
        self.num_assets = num_assets
        self.total_months = total_months
        self.timeline = self._generate_fbm_matrix()
        
    def _generate_fbm_matrix(self):
        print(f"Generating FBM Matrix: {self.num_assets} assets x {self.total_months} months")
        start_time = time.time()
        
        # We will store the paths in a standard NumPy matrix
        prices = np.zeros((self.num_assets, self.total_months))
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Map the function to the number of assets we want
            # This returns an iterator of generated paths as they finish
            results = executor.map(generate_single_fbm_path, [self.total_months] * self.num_assets)
            
            # Unpack the parallel results directly into our matrix
            for i, path in enumerate(results):
                prices[i, :] = path
                
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
            
            # Strictly normalize the data (Prices -> Percentages)
            window_normalized = window / window[0] 
            batch_data.append(window_normalized)
            
        batch_tensor = torch.tensor(np.array(batch_data), dtype=torch.float32)
        return batch_tensor[:, :context_length], batch_tensor[:, context_length:]

# ---------------------------------------------------------
# Execution Block: Generate and Save Massive Dataset
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Build the massive universe (100 million data points distributed across all cores)
    simulator = FractionalMarketSimulator(num_assets=5000, total_months=20000)
    
    # 2. Extract a massive training dataset
    total_samples = 100000
    ctx_len = 30
    filename = "fractional_quant_data.pt"
    
    print(f"\nSlicing {total_samples} random samples...")
    slice_start = time.time()
    inputs, targets = simulator.get_batch(batch_size=total_samples, context_length=ctx_len)
    print(f"Slicing complete in {time.time() - slice_start:.2f} seconds.")
    
    # 3. Save directly as a PyTorch binary file
    print(f"Saving tensors to {filename}...")
    torch.save({
        'inputs': inputs,
        'targets': targets
    }, filename)
    
    print(f"Success! {total_samples} sequences saved to {filename}.")