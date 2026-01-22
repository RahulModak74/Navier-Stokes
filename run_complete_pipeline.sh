# 1. Generate discontinuous data
python3 ns_toy_data_generator.py --samples 20000 --output ns_toy_discontinuous.npy

# 2. Train with Pyro (Bayesian priors!)
python3 ns_toy_vae_pyro_trainer_GPU.py --data ns_toy_discontinuous.npy

# 3. Visualize the learned discontinuities
python3 ns_toy_visualizer.py --model ns_toy_pyro_vae_GPU.pth --data ns_toy_discontinuous.npy
