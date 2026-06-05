# FDTD Dispersive Materials: Feynman Oscillators → Schneider Implementation

## Overview

Two tasks, implemented in **Python** (NumPy + Matplotlib). Both should be in a single well-structured codebase.

---

## Task 1: Derive Drude & Lorentz Permittivity from Feynman's Oscillator Models

### Goal
Show — analytically and numerically — that Feynman's two oscillator models (Vol I, Ch 23 & 31) produce exactly the Drude and Lorentz permittivity functions. Produce side-by-side plots proving they are identical.

### Physics: The Two Models

**Model A — Free Electron / Drude (no spring, damping only)**

Feynman's simplest case: a charged particle with only a damping force (no restoring spring). The equation of motion is:

```
m * x_ddot + m * γ * x_dot = q * E₀ * exp(-iωt)
```

This is effectively a **first-order** system in velocity. Solving in the frequency domain:

```
x̃(ω) = (q * E₀) / (m * (-ω² - iγω))
```

The polarization P = N*q*x̃ gives:

```
ε_Drude(ω) = ε_∞ - ωp² / (ω² + iγω)
```

where `ωp² = N*q²/(ε₀*m)` is the plasma frequency squared.

**Model B — Bound Electron / Lorentz (spring + damping)**

Feynman's driven damped harmonic oscillator (Ch 23 mechanics, applied to optics in Ch 31):

```
m * x_ddot + m * γ * x_dot + m * ω₀² * x = q * E₀ * exp(-iωt)
```

This is a genuine **second-order** system with resonance at ω₀. Solving:

```
ε_Lorentz(ω) = ε_∞ + ωp² / (ω₀² - ω² - iγω)
```

**Key insight to demonstrate:** Setting ω₀ = 0 in the Lorentz model collapses it exactly to the Drude model. The spring constant = 0 means no restoring force → free electrons → metals/plasmas.

### Implementation Details

1. **Derive symbolically** (use comments/docstrings to show the algebra step by step in the code).

2. **Plot the permittivity** for both models:
   - Use realistic parameters for **gold** (Drude): `ωp = 1.37e16 rad/s`, `γ = 4.05e13 rad/s`, `ε_∞ = 1.0`
   - Use example parameters for a **dielectric** (Lorentz): `ωp = 1.0e16 rad/s`, `ω₀ = 6.0e15 rad/s`, `γ = 2.0e14 rad/s`, `ε_∞ = 1.0`
   - Frequency range: `1e14` to `3e16 rad/s` (covers IR through UV)

3. **Produce 4 subplots** (2×2):
   - Top-left: Re(ε) for Drude — label key features (plasma frequency where Re(ε)=0, negative region)
   - Top-right: Im(ε) for Drude — show the loss peak
   - Bottom-left: Re(ε) for Lorentz — label the resonance, anomalous dispersion region
   - Bottom-right: Im(ε) for Lorentz — show absorption peak at ω₀

4. **Verification plot**: Take the Lorentz formula, set ω₀ = 0, and overlay it on the Drude plot to prove they are identical. Use dashed vs solid lines.

5. **Additional plot**: Show the transition from Lorentz → Drude by plotting Re(ε) for decreasing values of ω₀ (e.g., ω₀ = ω₀_full, 0.75*ω₀, 0.5*ω₀, 0.25*ω₀, 0) on the same axes. This visually demonstrates the resonance peak flattening out and becoming the Drude response.

### Style
- Clean, well-commented code
- Each function should have a docstring explaining the physics
- Print a summary of the derivation steps to console

---

## Task 2: 1D FDTD with Drude and Lorentz Materials (Schneider Style)

### Goal
Implement a 1D FDTD simulation that propagates a pulse into both a Drude material (metal-like) and a Lorentz material (dielectric resonance), using the **Auxiliary Differential Equation (ADE)** method exactly as described in Schneider's Chapter 10.

### Reference: Schneider's Book
- Book: "Understanding the FDTD Method" by John B. Schneider (free at https://eecs.wsu.edu/~schneidj/ufdtd/)
- Chapter 10: "Dispersive Material" — specifically sections 10.2–10.4
- Schneider uses C with structs, but we translate to Python with classes/dicts
- Schneider's coding style: modular, clean, minimal — each update step is explicit and readable

### The ADE Method (How Schneider Does It)

The key idea: instead of doing a convolution in time (expensive), we introduce auxiliary variables (the polarization current J) and derive update equations for them alongside the standard E and H updates.

#### Standard (non-dispersive) FDTD update in 1D:

```
H^{q+1/2}[k+1/2] = H^{q-1/2}[k+1/2] + (Δt/(μ₀*Δx)) * (E^q[k] - E^q[k+1])

E^{q+1}[k] = E^q[k] + (Δt/(ε₀*Δx)) * (H^{q+1/2}[k-1/2] - H^{q+1/2}[k+1/2])
```

#### For Drude material — ADE update:

The polarization current J satisfies the first-order ODE:
```
dJ/dt + γ*J = ε₀ * ωp² * E
```

Discretize using semi-implicit (average E at q and q+1):
```
J^{q+1} = ((1 - γΔt/2) / (1 + γΔt/2)) * J^q  +  (ε₀*ωp²*Δt / (1 + γΔt/2)) * (E^{q+1} + E^q) / 2
```

But E^{q+1} is unknown! So we substitute into Ampere's law and solve simultaneously. Define coefficients:

```
β_D = (1 - γ*Δt/2) / (1 + γ*Δt/2)
κ_D = (ε₀ * ωp² * Δt/2) / (1 + γ*Δt/2)
```

Then the **modified E update** in the Drude region becomes:
```
E^{q+1}[k] = (1/(ε_∞ + κ_D*Δt/(2*ε₀))) * [
    ε_∞ * E^q[k]
    + (Δt/(ε₀*Δx)) * (H^{q+1/2}[k-1/2] - H^{q+1/2}[k+1/2])
    - (Δt/ε₀) * (1+β_D)/2 * J^q
    + κ_D*Δt/(2*ε₀) * E^q[k]
]
```

Wait — let me be more precise. Schneider formulates it as follows. Ampere's law with polarization current:

```
ε₀ * ε_∞ * (E^{q+1} - E^q)/Δt = curl(H)^{q+1/2} - J^{q+1/2}
```

where `J^{q+1/2} = (J^{q+1} + J^q) / 2`.

Substituting the J update and solving for E^{q+1}:

The result is a coupled system. The cleanest way (Schneider's approach) is:

**Step 1:** Store `E_tmp = E^q` (save old E)

**Step 2:** Update E provisionally using curl(H) and old J:
```
E^{q+1}[k] = C_a * E^q[k] + C_b * (H[k-1/2] - H[k+1/2]) - C_c * J^q[k]
```
where C_a, C_b, C_c are precomputed coefficients that account for the semi-implicit coupling.

**Step 3:** Update J using both E^{q+1} and E_tmp:
```
J^{q+1}[k] = β_D * J^q[k] + κ_D * (E^{q+1}[k] + E_tmp[k])
```

The coefficients for the Drude E-field update are:
```
C_a = (2*ε₀*ε_∞ - κ_D*Δt) / (2*ε₀*ε_∞ + κ_D*Δt)     [Note: for Drude, sign is different!]
C_b = (2*Δt) / ((2*ε₀*ε_∞ + κ_D*Δt)*Δx)
C_c = (2*Δt*(β_D + 1)) / (2*(2*ε₀*ε_∞ + κ_D*Δt))
```

**IMPORTANT NOTE FOR IMPLEMENTER:** The exact coefficient expressions may differ slightly between Drude and Lorentz due to the sign of E^q in the J update equation. Please verify by re-deriving from the continuous equations. Schneider notes this explicitly: "the only difference is the sign of the old electric field associated with the polarization current."

#### For Lorentz material — ADE update:

The polarization P satisfies the second-order ODE:
```
d²P/dt² + γ*dP/dt + ω₀²*P = ε₀ * ωp² * E
```

Define `J = dP/dt` (polarization current). Then we have the system:
```
dJ/dt + γ*J + ω₀²*P = ε₀ * ωp² * E
dP/dt = J
```

This requires **two auxiliary arrays**: both J[k] and P[k] at every dispersive cell.

Discretize (central differences, semi-implicit for E):
```
(J^{q+1} - J^q)/Δt + γ*(J^{q+1} + J^q)/2 + ω₀²*(P^{q+1} + P^q)/2 = ε₀*ωp²*(E^{q+1} + E^q)/2

(P^{q+1} - P^q)/Δt = (J^{q+1} + J^q)/2
```

From the second equation: `P^{q+1} = P^q + Δt*(J^{q+1} + J^q)/2`

Substitute into the first and solve for J^{q+1}, then get P^{q+1}.

Define coefficients:
```
α_L = (2 - ω₀²*Δt²) / (2 + γ*Δt + ω₀²*Δt²/2)   -- wait, need to be careful
```

Actually, the Lorentz ADE coefficients are more involved. The cleanest formulation:

```
D1 = (4 + 2*γ*Δt + ω₀²*Δt²)            [denominator]
β_J = (4 - 2*γ*Δt - ω₀²*Δt²) / D1       [coefficient for old J]  -- VERIFY SIGN
β_P = (-4*ω₀²*Δt) / D1                    [coefficient for old P] -- VERIFY
κ_L = (2*ε₀*ωp²*Δt) / D1                  [coefficient for E]
```

Then:
```
J^{q+1} = β_J * J^q + β_P * P^q + κ_L * (E^{q+1} + E^q)
P^{q+1} = P^q + Δt/2 * (J^{q+1} + J^q)
```

And the E update for Lorentz cells follows the same structure as Drude but with κ_L replacing κ_D.

**CRITICAL: The implementer should re-derive these coefficients from scratch to make sure signs and factors of 2 are correct. Start from the continuous ODEs, discretize with central differences, and solve the coupled system. This is the most error-prone part.**

### Simulation Setup

```
Grid:
  - Size: 600 cells
  - Δx: 10 nm (suitable for optical frequencies)
  - Δt: Courant limit = Δx / (2*c) [use safety factor 0.99]
  - Time steps: 3000-4000

Source:
  - Type: TFSF (Total-Field/Scattered-Field) boundary at cell 80
  - Waveform: Gaussian pulse, width ~30 time steps, delay ~80 time steps
  - This gives broadband excitation covering the frequency range of interest

Materials:
  - Cells 0–299: Free space (ε = 1)
  - Cells 300–449: Dispersive material (either Drude or Lorentz)
  - Cells 450–599: Free space

  Run TWO simulations:
    1. Drude slab with gold-like parameters: ωp = 1.37e16, γ = 4.05e13, ε_∞ = 1.0
    2. Lorentz slab with: ωp = 1.0e16, ω₀ = 6.0e15, γ = 2.0e14, ε_∞ = 2.25

Boundaries:
  - Simple first-order ABC (absorbing boundary condition) on both ends
  - Or: second-order ABC if straightforward to implement
  - See Schneider Chapter 6 for implementation

Recording:
  - Record E-field at a point BEFORE the slab (reflected field monitor, e.g., cell 150)
  - Record E-field at a point AFTER the slab (transmitted field monitor, e.g., cell 500)
  - Record E-field snapshots every 50 time steps for animation/waterfall plot
```

### Output Plots

1. **Snapshots**: Show E-field across the entire grid at 4–6 key time steps:
   - Before pulse hits slab
   - Pulse entering slab
   - Pulse inside slab (show dispersion/absorption)
   - Pulse exiting slab + reflected pulse
   - Shade the slab region in the background

2. **Time-domain signals**: Plot the reflected and transmitted E-field vs time for both Drude and Lorentz cases (4 curves total, 2 subplots).

3. **Frequency-domain reflection coefficient**:
   - Take FFT of incident, reflected, and transmitted pulses
   - Compute R(ω) = |FFT(reflected)| / |FFT(incident)|
   - Compute T(ω) = |FFT(transmitted)| / |FFT(incident)|
   - Overlay with the **analytical** R and T from Fresnel equations using the Drude/Lorentz ε(ω)
   - This is the key validation: FDTD result must match analytical theory

4. **Permittivity validation**: On a separate plot, show the permittivity ε(ω) used in the simulation (from Task 1 functions) and annotate the frequency range covered by the Gaussian pulse.

### Code Structure (Schneider Style)

Schneider organizes his code into modular functions. Follow this pattern:

```python
# === Grid class/dict ===
class Grid:
    """Holds all field arrays, material arrays, and grid parameters."""
    ez: np.ndarray       # E-field
    hy: np.ndarray       # H-field
    # For Drude cells:
    jz_drude: np.ndarray # polarization current (1 auxiliary variable)
    # For Lorentz cells:
    jz_lorentz: np.ndarray  # polarization current
    pz_lorentz: np.ndarray  # polarization (2 auxiliary variables!)
    
    # Coefficient arrays (precomputed):
    ca, cb: np.ndarray   # standard E update coefficients
    cc: np.ndarray       # coupling to J
    da, db: np.ndarray   # H update coefficients
    
    # Dispersive coefficients per cell:
    beta_j, beta_p, kappa: np.ndarray

# === Update functions ===
def update_h(g: Grid):
    """Magnetic field update — identical to standard FDTD."""

def update_e(g: Grid):
    """Electric field update — modified for dispersive cells."""
    # Step 1: save old E
    # Step 2: standard-looking E update (but with modified coefficients in dispersive region)
    # Step 3: update J (and P for Lorentz) using new E and old E

def apply_abc(g: Grid):
    """First-order absorbing boundary condition."""

def apply_tfsf(g: Grid, time_step: int):
    """Total-field/scattered-field source injection."""

def gaussian_pulse(time_step, delay, width):
    """Gaussian pulse source function."""

# === Main time-stepping loop ===
for q in range(max_steps):
    update_h(g)
    apply_tfsf(g, q)  # correct H at TFSF boundary
    update_e(g)
    apply_tfsf(g, q)  # correct E at TFSF boundary
    apply_abc(g)
    # record monitors
```

### Validation Checklist

After implementation, verify:
- [ ] Energy conservation: for lossless case (γ→0), |R|² + |T|² ≈ 1
- [ ] Drude: below plasma frequency, wave should be mostly reflected (metal behavior)
- [ ] Lorentz: strong absorption near ω₀, anomalous dispersion visible
- [ ] Setting ω₀ = 0 in Lorentz code reproduces Drude results exactly
- [ ] Reflection coefficient matches Fresnel analytical prediction
- [ ] No late-time instability (run for extra 1000 steps and check fields don't blow up)

---

## File Organization

```
fdtd_dispersive/
├── permittivity.py          # Task 1: Drude & Lorentz permittivity functions + derivation plots
├── fdtd_1d.py               # Task 2: The 1D FDTD engine with ADE
├── run_simulation.py         # Task 2: Sets up and runs both Drude & Lorentz simulations
├── analytical.py             # Fresnel coefficients for a dispersive slab (for validation)
├── plots.py                  # All plotting functions
└── main.py                   # Entry point: runs everything, produces all figures
```

---

## Notes for Claude Code

1. **Coefficient derivation is critical.** The most common bug in dispersive FDTD is wrong signs or factors of 2 in the ADE coefficients. Re-derive from scratch: start with the continuous ODE, apply central-difference in time, average E semi-implicitly, substitute into Ampere's law, and solve for E^{q+1}. Do this separately for Drude and Lorentz.

2. **Schneider's key observation:** The Drude and Lorentz E-update equations are almost identical — the only difference is the sign of the old E-field term in the J update, and the Lorentz case has an extra P array. Structure the code to exploit this similarity.

3. **Stability:** The Courant number must satisfy `c*Δt/Δx ≤ 1` (in 1D). Use `Δt = 0.99 * Δx / c`. For dispersive materials, there can be additional stability constraints — if the simulation blows up, try reducing Δt by a factor of 2.

4. **Units:** Work in SI throughout. Typical values: Δx = 10 nm = 1e-8 m, frequencies in rad/s.

5. **Source reference:** Schneider's book is freely available at https://eecs.wsu.edu/~schneidj/ufdtd/ — Chapter 10 covers dispersive materials with the ADE method. Chapters 3–4 cover the basic 1D FDTD setup. The code style there is C, but the logic translates directly to Python.