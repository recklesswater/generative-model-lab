# 04 - SE(3) diffusion: frames, equivariance, and guidance

Protein backbones are not vectors. A residue is a **rigid frame** $T_i = (R_i, t_i)$ with
$R_i \in SO(3)$, and a global rotation $Q$ acts as $(R_i, t_i) \mapsto (Q R_i, Q t_i)$.
Any model of structures has to respect that action, and the cheapest way to prove it does is a
test you can run in a second:

```bash
python 04_se3_diffusion/equivariance_test.py     # group utilities + architectural equivariance
python 04_se3_diffusion/run_demo.py              # train, generate, steer with a property
```

## What is here

| File | Content |
| --- | --- |
| `so3.py` | `hat`/`vee`, exponential and logarithmic maps, Haar-random rotations, tangent-space noise |
| `frames.py` | toy helix backbones, the global action, invariant features (numpy + torch) |
| `model.py` | denoiser that consumes invariants and emits body-frame vectors |
| `diffusion_se3.py` | corruption, training loss, DDIM-style sampling with a guidance hook |
| `equivariance_test.py` | assertions: group utilities, and $f(Q \cdot T) = Q \cdot f(T)$ |
| `run_demo.py` | trains, generates, and steers the end-to-end distance of the chain |

## The three ideas, in the order they matter

1. **Noise must respect the group.** Adding Gaussians to the nine entries of a rotation matrix
   leaves $SO(3)$ immediately - the test `R^T R == I after tangent-space noise` is what that
   failure looks like when written down. Here the noise is drawn in the tangent space and mapped
   back with the exponential map, so the corrupted frames are always valid rotations
   (error $\sim 10^{-15}$).
2. **Equivariance can be built in, not learned.** The denoiser sees only invariants (relative
   translations expressed in each residue's own frame, and relative rotations) and emits only
   body-frame vectors. The world-frame output is reconstructed at the end, so the equivariance
   error is float32 precision ($\sim 10^{-8}$) *before any training*. A generic MLP on flattened
   coordinates fails this test immediately; that failure is the reason tensor-product networks
   (e3nn-style, with Clebsch-Gordan coefficients) exist.
3. **Conditioning is a gradient on the clean estimate.** `DistanceGuide` defines a property
   (the end-to-end distance should be some value), differentiates it with respect to the
   predicted clean structure, and nudges each reverse step. Motif scaffolding, binder interfaces
   and symmetry constraints are the same mechanism with a more informative score function.

## Honest caveats

* The corruption process is **not** a mathematically exact SE(3) diffusion: translations follow a
  standard variance-preserving diffusion, while rotations receive a one-shot isotropic
  tangent-space perturbation. The mechanics being demonstrated (tangent-space noise, equivariant
  denoiser, guided sampling) are the same; the interpolation on the group is simplified.
  Replacing it with a proper geodesic forward kernel is the first item on the roadmap.
* The toy data are helices with jitter, not proteins. Nothing here should be read as a statement
  about real backbone statistics - the smallest interesting test would be to condition on a real
  motif from a PDB fragment.

## Outputs

* `figures/04_frames.png` - training vs generated backbones, consecutive and end-to-end distance
  distributions, and the effect of the distance guide.

## Next

* Proper geodesic (IGSO3-style) forward kernel, then re-run the same equivariance test as a
  regression guard.
* Replace the MLP with a small tensor-product network and keep the MLP as the ablation that
  quantifies how much accuracy the symmetry is worth.
* Add a **motif-scaffolding** task: fix the frames of a fragment and diffuse the rest, which is
  the closest toy version of what RFdiffusion does in practice.
