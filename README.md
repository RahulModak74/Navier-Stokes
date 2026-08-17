# Bayesian Neural Manifolds for Discontinuous Flow Fields

**A Bayesian VAE that learns a low-dimensional latent representation of a rough, discontinuity-laden synthetic flow field — and reconstructs it where naive gradient-based methods break down.**

Part of the **Modak–Walawalkar (MW) Framework**, previously applied to Bayesian General Relativity.
See [github.com/RahulModak74/mw-framework](https://github.com/RahulModak74/mw-framework).

---

## What this is (and what it is not)

This repository is a **computational demonstration**, not a proof. It shows that a
variational autoencoder with physics-informed likelihood terms can learn a compact
latent structure for a synthetic flow field that contains deliberately non-differentiable
features (sign-function vortex streets, compact-support bursts, 1/r cores), and can
reconstruct that field after the singular values have been made finite through
robust preprocessing.

**It does not:**

- solve the incompressible Navier–Stokes equations,
- resolve or bear on the Clay Millennium Problem,
- compute solutions in a Banach space `W^{1,p}` (that framing is *conceptual
  motivation only* — see [Framing](#framing-and-honesty) — and is not implemented
  in the code),
- train on real flow data.

The data is a hand-constructed analytic field designed to be rough, not the output
of a PDE solver. The value here is the **method** — physics constraints expressed as
Bayesian likelihood factors rather than weighted penalties — demonstrated on a
controlled, adversarial-by-design toy problem.

---

## The idea

Classical finite-difference and spectral methods assume smoothness: they differentiate
the field, so a genuine singularity (|∇u| → ∞ at a vortex core) produces `NaN`/`Inf`
and the method fails. This repo takes a different route:

- Represent the rough field as a set of scalar samples.
- Make the singular values finite and information-preserving through robust
  normalization + signed-log compression (ordering and relative magnitude survive;
  raw `Inf` does not enter training).
- Learn a **low-dimensional latent manifold** of the field with a VAE.
- Express physical constraints (incompressibility, a momentum proxy, vorticity,
  pressure) as **Bayesian likelihood factors** — `pyro.sample(..., obs=...)` — so that
  heavy-tailed distributions (Laplace, StudentT) can accommodate sharp gradients
  instead of fighting them.

```
Rough field samples  →  Encoder q(z|x)  →  z ∈ ℝ⁴  →  Decoder p(x|z)  →  reconstruction
                                              ↑
                              physics as likelihood factors
                              (Normal / Laplace / StudentT)
```

---

## Physics as likelihood factors (not penalties)

A conventional physics-informed loss adds weighted penalty terms:

```
loss = ||x - data||² + λ₁·||∇·u||² + λ₂·||residual||²
```

Here the same constraints are **observed sample sites** in a Pyro model, so the
"tightness" of each constraint is a distribution scale rather than a hand-tuned λ,
and the tail weight encodes how much sharpness the constraint tolerates:

```python
pyro.sample("incompressibility_obs", dist.Normal(0, 0.5),  obs=divergence)   # smooth
pyro.sample("momentum_obs",          dist.Laplace(0, 0.5), obs=accel)         # moderate tails
pyro.sample("vorticity_obs",         dist.StudentT(2.0, 0, 1), obs=omega)     # heavy tails
pyro.sample("pressure_obs",          dist.Normal(0, 1),   obs=p_field)
```

### Honest limitations of the current physics terms

These are real and worth stating up front — they are the next things to fix, not
things to paper over:

1. **The constraints read the decoder's own reconstructed columns.** `divergence`
   is `x_recon[:,7] + x_recon[:,8]` — i.e. the model's *outputs* for ∂u/∂x and ∂v/∂y,
   not derivatives of the reconstructed `u, v`. So incompressibility is a constraint on
   two output columns, not a constraint that actually ties the velocity field to its
   own divergence. Wiring this correctly (decoder emits `u, v`; derivatives via
   autograd) is the single highest-value improvement and is on the roadmap.
2. **The momentum term is a proxy.** The true advection term `(u·∇)u` needs `∂u/∂y`,
   which the 10-column data does not carry. The code uses `u·∂u/∂x + v·∂v/∂y`, the
   closest the current data supports. It is not a momentum residual.

Treat the four factors as **auxiliary regularizers with physically-motivated tail
behavior**, which is what they currently are, rather than as enforced conservation laws.

---

## Actual configuration

| Item | Value |
|---|---|
| Data representation | flat table of scalar samples, **10 columns** |
| Columns | `[t, x, y, u, v, p, ω, ∂u/∂x, ∂v/∂y, is_discontinuous]` |
| Latent dimension | **4** (must be `< input_dim = 10` for a real bottleneck) |
| Data points | 20,000 (mixed laminar / transition / turbulent) |
| Reynolds number | 1000 (generator default) |
| Framework | PyTorch + Pyro (SVI, Trace_ELBO) |

> Note: an earlier build used `latent_dim=16`, which is *larger* than the 10-column
> input and defeats the bottleneck. The current code fixes this to 4. If you see 16
> quoted anywhere, that is the superseded configuration.

---

## Quick start

```bash
git clone https://github.com/RahulModak74/bayesian-ns-solver.git
cd bayesian-ns-solver
pip install torch numpy pyro-ppl matplotlib scipy
```

For GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step by step

```bash
# 1. Generate the discontinuous toy field
python3 ns_toy_data_generator.py --samples 20000 --output ns_toy_discontinuous.npy

# 2. Train the Pyro VAE (GPU if available)
python3 ns_toy_vae_pyro_trainer_GPU.py --data ns_toy_discontinuous.npy
#    → saves ns_toy_pyro_vae_GPU.pth
#    (CPU variant: ns_toy_vae_pyro_trainer.py → ns_toy_pyro_vae_FIXED.pth)

# 3. Visualize
python3 ns_toy_visualizer.py --model ns_toy_pyro_vae_GPU.pth --data ns_toy_discontinuous.npy
```

---

## Reading the results honestly

The trainer prints the only numbers that decide whether the model learned anything.
**Do not report a bare MSE** — report it against the baseline:

```
Reconstruction MSE:            <value>
Predict-the-mean baseline MSE: <value>
Ratio MSE/baseline:            <value>   (< 1.0 = the model beats predicting the mean)
latent KL per-dim:             [...]     (dims with std < 0.1 and KL ~ 0 are dead)
```

A reconstruction MSE means nothing without the **ratio to the predict-the-mean
baseline**. If the ratio is near or above 1.0, the reconstruction claim does not hold,
regardless of how small the raw MSE looks. Likewise, check the per-dimension KL: if
most latent dimensions are dead, the "manifold" is lower-dimensional than advertised.

### On the interpolation panel

The visualizer shows a latent interpolation curve. A nonlinear interpolation curve is
a generic property of *any* nonlinear VAE decoder on *any* dataset — it is **not**
evidence of Banach geometry or of anything specific to Navier–Stokes. It is included as
a qualitative sanity check, not as proof of manifold structure.

---

## Framing and honesty

The MW framing here is: **think in terms of a learned manifold and Bayesian inference,
rather than differentiating a field you have assumed to be smooth.** That is a genuine
and defensible methodological stance, and it is what the code demonstrates on a toy
problem.

Everything beyond that — Banach `W^{1,p}` weak solutions, computational
constructivism as a resolution of existence, "solving intractable PDEs" — is
**aspirational motivation, not implemented result.** The weak-form functional
`∫∫[u·∂ₜφ + u⊗u:∇φ − p∇·φ + ν∇u:∇φ] = 0` does not appear in the code and no `W^{1,p}`
norm is computed. We keep the motivation because it points at where the work is going;
we label it clearly so no reader mistakes it for what has been done.

We do not claim, and this repository does not support any claim, about the Clay
Millennium Problem.

---

## Roadmap

- **Wire one physics constraint correctly.** Decoder emits `(u, v)` at collocation
  points; compute `∇·u` via autograd; place the incompressibility factor on *that*.
  One genuinely-enforced constraint is worth more than four decorative ones.
- **Held-out evaluation.** Every metric is currently computed on training samples.
  Add a proper train/test split and report the ratio on held-out data.
- **Test the "discontinuities are learnable" claim quantitatively.** Fit a classifier
  on the latent means predicting `is_discontinuous`; report AUC. Right now this claim
  is asserted, not measured.
- **Richer data.** Carry `∂u/∂y` so the true advection term is expressible, or move to
  a grid representation where derivatives are well-defined.

---

## Citation

```bibtex
@software{modak_bayesian_ns_manifold_2025,
  title  = {Bayesian Neural Manifolds for Discontinuous Flow Fields:
            A Physics-Informed VAE Demonstration (MW Framework)},
  author = {Modak, Rahul and Walawalkar, Rahul},
  year   = {2025},
  url    = {https://github.com/RahulModak74/bayesian-ns-solver},
  note   = {Computational demonstration; not a claim on the Clay Millennium Problem}
}
```

## Acknowledgments

Pyro (probabilistic programming), PyTorch (deep learning infrastructure).

---

**License:** MIT · **Status:** Active development · **Version:** 1.0.0
