#!/usr/bin/env python3
"""
Quick test to verify GPU training works without device mismatch errors.
"""

import torch
import numpy as np
from pathlib import Path

# Check if file exists
gpu_trainer = Path("ns_toy_vae_pyro_trainer_GPU.py")
if not gpu_trainer.exists():
    print("❌ GPU trainer not found!")
    exit(1)

# Import the fixed trainer
import sys
sys.path.insert(0, str(gpu_trainer.parent))
from ns_toy_vae_pyro_trainer_GPU import NSDiscontinuousPyroVAE, preprocess_data

print("="*70)
print("GPU TRAINER QUICK TEST")
print("="*70)

# Check GPU availability
if torch.cuda.is_available():
    device = 'cuda'
    print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n⚠️  No GPU detected, using CPU")
    device = 'cpu'

# Load small sample of data
data_file = Path("ns_toy_discontinuous.npy")
if not data_file.exists():
    print(f"\n❌ Data file not found: {data_file}")
    exit(1)

print(f"\nLoading data: {data_file}")
data = np.load(data_file)
print(f"  Data shape: {data.shape}")

# Preprocess
print("\nPreprocessing...")
data_clean, data_mean, data_std = preprocess_data(data[:1000], max_gradient=10.0, max_value=5.0)
data_tensor = torch.tensor(data_clean, dtype=torch.float32)

# Create small model
print(f"\nCreating VAE on {device}...")
vae = NSDiscontinuousPyroVAE(
    input_dim=data.shape[1],
    latent_dim=8,  # Smaller for quick test
    device=device
)

# Move data to device
data_tensor = data_tensor.to(device)
print(f"✓ Data on {data_tensor.device}")
print(f"✓ Model on {next(vae.parameters()).device}")

# Test single forward pass
print("\nTesting forward pass...")
try:
    with torch.no_grad():
        # Test encode
        mu, logvar = vae.encode(data_tensor[:10])
        print(f"  ✓ Encode: mu shape {mu.shape}, device {mu.device}")
        
        # Test decode
        recon = vae.decode(mu)
        print(f"  ✓ Decode: recon shape {recon.shape}, device {recon.device}")
        
        # Test model (Pyro)
        print(f"\nTesting Pyro model/guide...")
        import pyro
        pyro.clear_param_store()
        
        # Test model
        batch = data_tensor[:32]
        vae.model(batch)
        print(f"  ✓ Model forward pass successful")
        
        # Test guide  
        vae.guide(batch)
        print(f"  ✓ Guide forward pass successful")
        
        print("\n" + "="*70)
        print("SUCCESS: All device checks passed! ✓")
        print("="*70)
        print("\nGPU training should work now. Run:")
        print("  python ns_toy_vae_pyro_trainer_GPU.py --data ns_toy_discontinuous.npy --epochs 100")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  Device mismatch still present. Check the error above.")
    exit(1)

