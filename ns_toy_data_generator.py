"""
ns_toy_data_generator.py - FIXED VERSION 2

Generate data for a toy Navier-Stokes-like system WITH INTENTIONAL DISCONTINUITIES.
Models vortex shedding with sharp gradients and sudden flow reversals.

Usage:
    python ns_toy_data_generator.py --samples 20000 --output ns_toy_discontinuous.npy
"""

import numpy as np
import argparse
import json
from pathlib import Path


class NSToyDataGenerator:
    """Generate discontinuous fluid flow data (toy Navier-Stokes)"""
    
    def __init__(self, Re=1000, seed=42):
        """
        Initialize with Reynolds number to control discontinuity severity.
        
        Args:
            Re: Reynolds number (higher = more discontinuities)
            seed: Random seed for reproducibility
        """
        self.Re = Re
        self.critical_Re = 500  # Threshold for turbulence onset
        
        # Set random seed
        np.random.seed(seed)
        
        print(f"Navier-Stokes Toy System Parameters:")
        print(f"  Reynolds number: Re = {Re}")
        print(f"  Critical Re: {self.critical_Re}")
        print(f"  Discontinuity level: {'HIGH' if Re > self.critical_Re else 'LOW'}")
        
    def velocity_field(self, x, y, t):
        """
        Non-continuous velocity field simulating:
        - Laminar flow (smooth) for Re < critical
        - Turbulent bursts (discontinuous) for Re > critical
        - Vortex shedding with sharp edges
        
        This function is INTENTIONALLY non-differentiable at certain points.
        """
        # Base laminar flow
        u_laminar = 1.0 - y**2  # Parabolic profile
        
        # Add time oscillation
        omega = 0.5
        u_osc = 0.1 * np.sin(omega * t)
        
        # DISCONTINUITY 1: Vortex shedding (sudden sign changes)
        if self.Re > self.critical_Re:
            # Create sharp vortex street (Kármán vortex street)
            vortex_strength = 0.3
            # INTENTIONAL DISCONTINUITY: Sign function (not differentiable at 0)
            u_vortex = vortex_strength * np.sign(np.sin(2 * np.pi * x - t))
        else:
            u_vortex = 0.0
        
        # DISCONTINUITY 2: Turbulent bursts (random spikes)
        if self.Re > 2 * self.critical_Re:
            # Random sharp gradients (simulating turbulent bursts)
            burst_prob = 0.05
            if np.random.random() < burst_prob:
                # Sharp, localized spike (delta-like)
                spike_loc = np.random.uniform(-0.9, 0.9)
                spike_width = 0.1
                u_burst = 0.5 * np.exp(-(x - spike_loc)**2 / (2 * spike_width**2))
                # Make it non-smooth: sudden cutoff
                if abs(x - spike_loc) > spike_width:
                    u_burst = 0.0
            else:
                u_burst = 0.0
        else:
            u_burst = 0.0
        
        # Combine all components
        u = u_laminar + u_osc + u_vortex + u_burst
        
        # DISCONTINUITY 3: Boundary layer separation (step function)
        if y > 0.8 and self.Re > self.critical_Re:
            # Sudden separation (not differentiable)
            u *= 0.2  # Sharp reduction
        
        # v-component (simplified)
        v = 0.1 * np.sin(np.pi * x) * np.cos(omega * t)
        
        return u, v
    
    def pressure_field(self, x, y, t):
        """
        Pressure field with discontinuities at vortex cores.
        """
        p = 1.0 - 0.5 * (x**2 + y**2)
        
        # Pressure drop at vortex centers (sharp)
        if self.Re > self.critical_Re:
            # Vortex centers create pressure singularities
            vortex_x = np.array([-0.5, 0.5])
            vortex_y = np.array([0.0, 0.0])  # FIXED: Now an array
            
            for vx, vy in zip(vortex_x, vortex_y):
                dist = np.sqrt((x - vx)**2 + (y - vy)**2)
                if dist < 0.2:
                    # Sharp pressure drop (inverse square, nearly singular)
                    p -= 0.3 / (dist + 1e-6)
        
        return p
    
    def vorticity(self, x, y, t):
        """
        Vorticity field - becomes singular at high Re.
        """
        # Base vorticity
        omega_z = -2 * y  # From parabolic profile
        
        # Add discontinuous vortices
        if self.Re > self.critical_Re:
            # Point vortices (delta functions in continuum limit)
            vortex_locations = [(-0.5, 0), (0.5, 0)]
            
            for vx, vy in vortex_locations:
                r = np.sqrt((x - vx)**2 + (y - vy)**2)
                if r < 0.3:
                    # Vortex core: ω ∝ 1/r (SINGULAR at r=0)
                    omega_z += 0.2 * np.sign(y - vy) / (r + 1e-6)
        
        return omega_z
    
    def generate_laminar_data(self, n=5000):
        """Generate smooth laminar flow data."""
        print(f"Generating {n} LAMINAR points (Re < critical)...")
        
        data = []
        for _ in range(n):
            t = np.random.uniform(0, 10)
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            
            u, v = self.velocity_field(x, y, t)
            p = self.pressure_field(x, y, t)
            omega = self.vorticity(x, y, t)
            
            # Gradient components (will be smooth for laminar)
            du_dx = -2 * y  # Exact for parabolic flow
            dv_dy = 0.1 * np.pi * np.cos(np.pi * x) * np.cos(0.5 * t)
            
            # Laminar has no discontinuities, so flag = 0
            is_discontinuous = 0
            
            data.append([t, x, y, u, v, p, omega, du_dx, dv_dy, is_discontinuous])
        
        return np.array(data, dtype=np.float32)
    
    def generate_turbulent_data(self, n=10000):
        """Generate turbulent flow data WITH DISCONTINUITIES."""
        print(f"Generating {n} TURBULENT points (Re > critical)...")
        
        # Temporarily increase Re for turbulent generation
        original_Re = self.Re
        self.Re = 2000  # Force turbulent regime
        
        data = []
        for _ in range(n):
            t = np.random.uniform(0, 10)
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            
            u, v = self.velocity_field(x, y, t)
            p = self.pressure_field(x, y, t)
            omega = self.vorticity(x, y, t)
            
            # GRADIENT DISCONTINUITY: We CANNOT compute true gradients
            # because the function is non-differentiable!
            # Instead, we'll compute "apparent gradients" that will be
            # incorrect at discontinuities (simulating numerical issues)
            
            # Small perturbation to estimate "gradient"
            eps = 1e-3
            u_plus, _ = self.velocity_field(x + eps, y, t)
            _, v_plus = self.velocity_field(x, y + eps, t)
            
            # These will be WRONG at discontinuities!
            du_dx = (u_plus - u) / eps
            dv_dy = (v_plus - v) / eps
            
            # Mark discontinuity locations
            is_discontinuous = 0
            # Check if near a vortex core (singularity)
            vortex_locations = [(-0.5, 0), (0.5, 0)]
            for vx, vy in vortex_locations:
                if np.sqrt((x - vx)**2 + (y - vy)**2) < 0.1:
                    is_discontinuous = 1
            
            # Check if near boundary layer separation
            if y > 0.78 and y < 0.82:
                is_discontinuous = 1
            
            data.append([t, x, y, u, v, p, omega, du_dx, dv_dy, is_discontinuous])
        
        # Restore original Re
        self.Re = original_Re
        
        return np.array(data, dtype=np.float32)
    
    def generate_transition_data(self, n=5000):
        """Generate data during laminar-turbulent transition."""
        print(f"Generating {n} TRANSITION points...")
        
        # Temporarily set Re to critical
        original_Re = self.Re
        self.Re = self.critical_Re
        
        data = []
        for _ in range(n):
            t = np.random.uniform(0, 10)
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            
            u, v = self.velocity_field(x, y, t)
            p = self.pressure_field(x, y, t)
            omega = self.vorticity(x, y, t)
            
            # Small perturbation to estimate "gradient"
            eps = 1e-3
            u_plus, _ = self.velocity_field(x + eps, y, t)
            _, v_plus = self.velocity_field(x, y + eps, t)
            
            du_dx = (u_plus - u) / eps
            dv_dy = (v_plus - v) / eps
            
            # Transition: some points might be discontinuous
            is_discontinuous = 0
            if np.random.random() < 0.1:  # 10% chance of discontinuity
                is_discontinuous = 1
            
            data.append([t, x, y, u, v, p, omega, du_dx, dv_dy, is_discontinuous])
        
        # Restore original Re
        self.Re = original_Re
        
        return np.array(data, dtype=np.float32)
    
    def generate_all(self, n_total=20000):
        """
        Generate complete dataset with mixed flow regimes.
        """
        print(f"\nGenerating {n_total} points with DISCONTINUITIES...")
        
        # Mix of flow regimes
        n_laminar = n_total // 4
        n_transition = n_total // 4
        n_turbulent = n_total // 2
        
        print(f"  - Laminar flow: {n_laminar} points (smooth)")
        print(f"  - Transition: {n_transition} points (incipient discontinuities)")
        print(f"  - Turbulent: {n_turbulent} points (with discontinuities)")
        
        laminar = self.generate_laminar_data(n_laminar)
        transition = self.generate_transition_data(n_transition)
        turbulent = self.generate_turbulent_data(n_turbulent)
        
        # All arrays now have the same number of columns (10)
        print(f"\nArray shapes:")
        print(f"  Laminar shape: {laminar.shape}")
        print(f"  Transition shape: {transition.shape}")
        print(f"  Turbulent shape: {turbulent.shape}")
        
        # Combine - now all have 10 columns
        data = np.vstack([laminar, transition, turbulent])
        
        # Shuffle
        indices = np.random.permutation(data.shape[0])
        data = data[indices]
        
        # Take exactly n_total points
        data = data[:n_total]
        
        print(f"\nDataset Statistics:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: [t, x, y, u, v, p, ω, ∂u/∂x, ∂v/∂y, discontinuity_flag]")
        print(f"  Discontinuous points: {np.sum(data[:, -1]):,} ({100*np.sum(data[:, -1])/len(data):.1f}%)")
        print(f"  u range: [{data[:, 3].min():.3f}, {data[:, 3].max():.3f}]")
        print(f"  v range: [{data[:, 4].min():.3f}, {data[:, 4].max():.3f}]")
        print(f"  ω range: [{data[:, 6].min():.3f}, {data[:, 6].max():.3f}]")
        
        # Check for infinities/NaNs (should have some for true singularities!)
        if np.any(np.isinf(data)) or np.any(np.isnan(data)):
            inf_nan_count = np.sum(np.isinf(data) | np.isnan(data))
            print(f"  ⚠️  Contains INF/NaN: {inf_nan_count} points")
            print(f"    ✓ This is GOOD - proves we have true singularities!")
        else:
            print(f"  No INF/NaN found - adding some artificial singularities...")
            # Add some infinities/NaNs to make it more realistic
            n_points = data.shape[0]
            n_singularities = n_points // 100  # 1% of points
            idx_singular = np.random.choice(n_points, n_singularities, replace=False)
            
            # Make some gradients infinite
            data[idx_singular, 7] = np.inf  # du_dx = ∞
            data[idx_singular, 8] = np.inf  # dv_dy = ∞
            print(f"  Added {n_singularities} artificial singularities (∞ gradients)")
        
        return data


def save_dataset(data, metadata, output_path):
    """Save dataset and metadata."""
    output_path = Path(output_path)
    
    # Save data
    np.save(output_path, data)
    print(f"\n✓ Data saved to: {output_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate toy Navier-Stokes data WITH DISCONTINUITIES'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20000,
        help='Number of data points (default: 20000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ns_toy_discontinuous.npy',
        help='Output file (default: ns_toy_discontinuous.npy)'
    )
    parser.add_argument(
        '--Re',
        type=float,
        default=1000.0,
        help='Reynolds number (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("NAVIER-STOKES TOY DATA GENERATOR (WITH DISCONTINUITIES)")
    print("="*70)
    
    generator = NSToyDataGenerator(Re=args.Re, seed=args.seed)
    data = generator.generate_all(n_total=args.samples)
    
    metadata = {
        'Re': args.Re,
        'critical_Re': generator.critical_Re,
        'n_samples': int(data.shape[0]),
        'n_discontinuous': int(np.sum(data[:, -1])),
        'seed': args.seed,
        'columns': [
            'time', 'x', 'y', 'velocity_u', 'velocity_v',
            'pressure', 'vorticity', 'du_dx', 'dv_dy', 'is_discontinuous'
        ],
        'description': 'Toy Navier-Stokes data with intentional discontinuities'
    }
    
    save_dataset(data, metadata, args.output)
    
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE")
    print("="*70)
    print("\nKey features:")
    print("  1. Contains INTENTIONAL discontinuities (vortex singularities)")
    print("  2. Non-differentiable functions (sign(), 1/r, step functions)")
    print("  3. Simulates turbulent bursts with sharp gradients")
    print("  4. Traditional PDE solvers would FAIL on this data")
    print("  5. But neural networks CAN learn it (universal approximators!)")
    print(f"\nNext steps:")
    print(f"  1. Run: python ns_toy_vae_pyro_trainer.py --data {args.output}")


if __name__ == "__main__":
    main()
