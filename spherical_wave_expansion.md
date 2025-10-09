# Spherical Wave Expansion in AntennaPattern

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation in AntennaPattern](#implementation-in-antpy)
4. [Algorithm Details](#algorithm-details)
5. [Usage Guide](#usage-guide)
6. [File Format Specifications](#file-format-specifications)
7. [References](#references)

---

## Introduction

Spherical Wave Expansion (SWE) is a rigorous mathematical technique for representing electromagnetic fields radiated by antennas as a weighted sum of vector spherical wave functions. This representation enables:

- **Near-field to far-field transformations**: Calculate far-field patterns from near-field measurements
- **Field extrapolation**: Determine fields at any point outside the minimum sphere
- **Pattern analysis**: Decompose complex radiation patterns into fundamental modes
- **Antenna diagnostics**: Identify dominant radiation mechanisms
- **Software interoperability**: Exchange pattern data with commercial tools (TICRA GRASP, CST, FEKO)

The AntennaPattern implementation follows the formulations established by Hansen (1988) and Ludwig (1971), with compatibility for TICRA GRASP format specifications.

---

## Mathematical Foundation

### Vector Spherical Wave Functions

Any electromagnetic field radiated by sources of finite extent can be expressed as:

```
E(r,θ,φ) = k√ζ Σ Σ Σ Q_smn F_smn(r,θ,φ)
              s=1 m  n
```

where:
- `k = 2π/λ` is the wavenumber
- `ζ = 377Ω` is the free-space impedance
- `Q_smn` are complex mode coefficients
- `F_smn` are vector spherical wave functions
- `s ∈ {1,2}` denotes mode type (TE/TM)
- `m ∈ [-n, n]` is the azimuthal mode index
- `n ∈ [1, N]` is the polar mode index

### Mode Types

**TE Modes (s=1, Transverse Electric):**
- Also called M-modes or magnetic modes
- Electric field is purely transverse (no radial component)
- Represent magnetic current sources

**TM Modes (s=2, Transverse Magnetic):**
- Also called N-modes or electric modes  
- Have radial electric field component
- Represent electric current sources

### Spherical Wave Functions (Hansen Convention)

The vector spherical wave functions in Hansen's formulation are:

```
F_smn(r,θ,φ) = (1/√(2π√(n(n+1)))) × (-m/|m|)^m × (m'_mn or n'_mn)
```

where the elementary functions are:

**For s=1 (TE modes):**
```
m'_mn(r,θ,φ) = h_n^(2)(kr) × [
    0 in r-direction,
    -(1/sinθ) dP̄_n^|m|(cosθ)/dθ × e^(imφ) in θ-direction,
    -(im/sinθ) P̄_n^|m|(cosθ) × e^(imφ) in φ-direction
]
```

**For s=2 (TM modes):**
```
n'_mn(r,θ,φ) = [
    n(n+1)/(kr) h_n^(2)(kr) P̄_n^|m|(cosθ) × e^(imφ) in r-direction,
    (1/kr) d/d(kr){kr h_n^(2)(kr)} dP̄_n^|m|(cosθ)/dθ × e^(imφ) in θ-direction,
    (im/(kr sinθ)) d/d(kr){kr h_n^(2)(kr)} P̄_n^|m|(cosθ) × e^(imφ) in φ-direction
]
```

### Normalized Associated Legendre Polynomials

The normalized associated Legendre functions are:

```
P̄_n^m(cosθ) = √[(2n+1)/2 × (n-m)!/(n+m)!] × P_n^m(cosθ)
```

**Important Convention Note:**
- SciPy's `lpmv` function includes the Condon-Shortley phase factor `(-1)^m`
- Hansen's formulation omits this factor
- Conversion required: `P̄_Hansen = (-1)^m × P̄_scipy`

### Spherical Hankel Functions

The spherical Hankel function of the second kind:

```
h_n^(2)(kr) = j_n(kr) - i×y_n(kr)
```

where `j_n` and `y_n` are spherical Bessel functions.

**Recurrence relations:**
```
h_{n-1}^(2)(kr) + h_{n+1}^(2)(kr) = (2n+1)/(kr) × h_n^(2)(kr)

d/d(kr){kr h_n^(2)(kr)} = kr h_{n-1}^(2)(kr) - n h_n^(2)(kr)
```

**Far-field asymptotic form (kr >> 1):**
```
h_n^(2)(kr) → (-i)^(n+1) × e^(-ikr)/(kr)
```

### Mode Truncation

The maximum polar mode index N is determined by the radius of the minimum sphere `r_0` that encloses the antenna:

```
N = kr_0 + C × (kr_0)^(1/3)
```

where:
- `C = 3.6` for -60dB truncation accuracy (recommended)
- `C = 5.0` for -80dB truncation accuracy
- `C = 10` for legacy implementations

The azimuthal index is limited by: `|m| ≤ min(n, M)`

### Power Distribution

The radiated power contained in mode n is:

```
P_rad(n) = (1/2) Σ_s Σ_m |Q_smn|²
```

Total radiated power:

```
P_total = (1/2) Σ_n Σ_s Σ_m |Q_smn|²
```

---

## Implementation in AntPy

### Architecture Overview

The SWE implementation in AntennaPattern consists of several key components:

```
spherical_expansion.py
├── Pattern preparation (extension, downsampling)
├── Legendre polynomial computation (fast cached)
├── Coefficient extraction (FFT-accelerated)
├── Field reconstruction (near/far field)
└── Power analysis and mode truncation

ant_io.py
├── TICRA .sph format I/O
├── Coefficient file management
└── Format conversion utilities

pattern.py
├── calculate_spherical_modes() - Main user interface
├── evaluate_nearfield_sphere() - Near-field evaluation
└── SWE coefficient storage
```

### Key Design Decisions

**1. Far-Field Extraction:**
The implementation extracts Q-coefficients directly from far-field angular patterns, not near-field spherical surfaces. This design choice:
- Matches typical antenna measurement scenarios
- Avoids numerical issues with Hankel function evaluations
- Simplifies the extraction process
- Maintains compatibility with standard antenna pattern formats

**2. Adaptive Grid Optimization:**
The code automatically:
- Extends partial patterns to full sphere coverage
- Converts central patterns to sided coordinate systems
- Downsamples oversampled grids to optimal Nyquist spacing
- Reduces computation time by 50-80% for high-resolution patterns

**3. Fast Legendre Computation:**
Uses optimized recurrence relations with logarithmic normalization to avoid overflow:
```python
log_norm = 0.5 × (log(2n+1) - log(2) + log((n-m)!) - log((n+m)!))
P̄_n^m = exp(log_norm) × P_n^m_unnormalized
```

**4. FFT Acceleration:**
Leverages FFT for azimuthal integration:
- Computes φ-integration in frequency domain
- Reduces complexity from O(N_θ × N_φ × N × M) to O(N_θ × N_φ log N_φ + N_θ × N × M)
- Provides 5-10× speedup for typical patterns

---

## Algorithm Details

### Coefficient Extraction from Far-Field Patterns

The extraction uses orthogonality of spherical wave functions:

```
Q_smn = ∫∫ E_tan(θ,φ) · K*_smn(θ,φ) sinθ dθ dφ
```

where `K_smn` are the far-field pattern functions:

**For s=1 (TE):**
```
K_1mn = √(2/(n(n+1))) × (-m/|m|)^m × (-i)^(n+1) × e^(imφ) × [
    (im/sinθ) P̄_n^|m|(cosθ) θ̂ - dP̄_n^|m|/dθ φ̂
]
```

**For s=2 (TM):**
```
K_2mn = √(2/(n(n+1))) × (-m/|m|)^m × (-i)^n × e^(imφ) × [
    dP̄_n^|m|/dθ θ̂ + (im/sinθ) P̄_n^|m|(cosθ) φ̂
]
```

### Implementation Steps

**Step 1: Pattern Preparation**
```python
theta, phi, e_theta, e_phi = prepare_pattern_for_swe(
    pattern_obj, N, noise_floor_db=-60, downsample_factor=1.5
)
```

This function:
1. Checks coordinate system (converts central → sided if needed)
2. Extends partial patterns to full sphere (θ ∈ [0°, 180°])
3. Calculates required sampling: Δθ ≤ π/N, Δφ ≤ 2π/N
4. Downsamples if current resolution exceeds requirements
5. Returns optimized grid

**Step 2: Legendre Polynomial Caching**
```python
legendre_cache = precompute_legendre_fast(N, M, THETA)
```

Computes all required `P̄_n^m(cosθ)` and `dP̄_n^m/dθ` using:
1. Recurrence relations for sequential computation
2. Log-gamma functions to prevent overflow
3. Dictionary caching for O(1) lookup

**Step 3: FFT-Accelerated Integration**
```python
Q_coefficients = extract_q_coefficients_fft(
    e_theta, e_phi, legendre_cache, M, N, k, radius,
    theta_rad, phi_rad, THETA, sin_theta
)
```

Algorithm:
```
1. FFT(E_θ) and FFT(E_φ) once over φ
2. For each (s, n, m):
   a. Compute basis function K*_smn(θ)
   b. Get FFT[E] for this m-index
   c. Integrate: ∫ (E_θ K*_θ + E_φ K*_φ) sinθ dθ
   d. Multiply by Δφ for φ-integration
   e. Apply normalization and phase correction
3. Return Q[s, m+M, n-1]
```

**Step 4: Mode Power Analysis**
```python
power_per_n = calculate_mode_power_distribution(Q_coefficients, M, N)
```

Computes:
```
P(n) = Σ_s Σ_m |Q_smn|²
```

Used for:
- Convergence checking (adaptive radius selection)
- Mode truncation decisions
- Diagnostic analysis

### Field Reconstruction

**Far-Field Evaluation:**
```python
E_theta, E_phi = evaluate_farfield_from_modes(
    Q_coefficients, M, N, k, theta, phi
)
```

Uses asymptotic form:
```
E = √(ζk/4π) Σ_smn Q_smn × (-i)^(n+s) × [pattern_function]
```

**Near-Field Evaluation:**
```python
E_r, E_theta, E_phi = evaluate_field_from_modes(
    Q_coefficients, M, N, k, r, theta, phi
)
```

Uses full spherical wave functions with Hankel function evaluation.

---

## Usage Guide

### Basic Workflow

**1. Calculate SWE Coefficients:**
```python
import numpy as np
from antenna_pattern import AntennaPattern

# Load or create antenna pattern
pattern = AntennaPattern.from_ffs_file('antenna.ffs')

# Calculate spherical modes (adaptive method)
swe_data = pattern.calculate_spherical_modes(
    radius=None,           # Auto-estimate from pattern
    frequency=None,        # Use first frequency
    adaptive=True,         # Automatic radius optimization
    power_threshold=0.99,  # Retain 99% of power
    convergence_threshold=0.01  # Max 1% in high modes
)

# Access results
Q_coefficients = swe_data['Q_coefficients']  # [2, 2*M+1, N]
N = swe_data['N']                           # Polar mode index
M = swe_data['M']                           # Azimuthal mode index
radius = swe_data['radius']                 # Final radius used
power = swe_data['mode_power']              # Total power
```

**2. Export to TICRA Format:**
```python
from antenna_pattern.ant_io import write_ticra_sph

write_ticra_sph(
    swe_data, 
    'antenna_modes.sph',
    program_tag='AntPy',
    id_string='My Antenna SWE'
)
```

**3. Reconstruct Pattern:**
```python
from antenna_pattern.ant_io import reconstruct_pattern_from_swe

# Define reconstruction grid
theta_angles = np.linspace(-180, 180, 361)
phi_angles = np.linspace(0, 360, 73)

# Reconstruct far-field
reconstructed_pattern = reconstruct_pattern_from_swe(
    swe_data,
    theta_angles=theta_angles,
    phi_angles=phi_angles
)

# Compare with original
pattern.plot_cuts(cuts=[0, 90])
reconstructed_pattern.plot_cuts(cuts=[0, 90])
```

**4. Evaluate Near-Field:**
```python
# On spherical surface
nearfield = pattern.evaluate_nearfield_sphere(
    radius=0.10,  # meters
    theta_points=np.linspace(0, 180, 181),
    phi_points=np.linspace(0, 360, 37)
)

E_theta = nearfield['E_theta']
E_phi = nearfield['E_phi']
```

### Advanced Usage

**Manual Radius Selection:**
```python
# Specify radius explicitly (disable adaptive)
swe_data = pattern.calculate_spherical_modes(
    radius=0.15,      # meters
    adaptive=False    # Use fixed radius
)
```

**Mode Truncation Analysis:**
```python
# Extract power distribution
power_per_n = swe_data['power_per_n']
cumulative_power = np.cumsum(power_per_n) / np.sum(power_per_n)

# Plot modal content
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(power_per_n)+1), power_per_n, 'o-')
plt.xlabel('Mode Index n')
plt.ylabel('Modal Power')
plt.title('Spherical Mode Power Distribution')
plt.grid(True)
plt.show()
```

**Custom Grid Optimization:**
```python
from antenna_pattern.spherical_expansion import prepare_pattern_for_swe

# Prepare pattern with custom settings
theta_opt, phi_opt, e_theta_opt, e_phi_opt = prepare_pattern_for_swe(
    pattern,
    N=50,                    # Target mode index
    noise_floor_db=-70,      # Extension noise floor
    downsample_factor=1.5    # Keep 1.5× Nyquist
)
```

### Validation and Diagnostics

**Check Convergence:**
```python
if swe_data['converged']:
    print(f"Converged after {swe_data['iterations']} iterations")
    print(f"Final radius: {swe_data['radius']:.4f} m")
    print(f"Power retained: {swe_data['power_retained_fraction']*100:.2f}%")
else:
    print("Warning: Did not converge - increase max_iterations")
```

**Power Distribution Analysis:**
```python
# Check if power is concentrated in low modes (good)
power_in_first_10 = np.sum(power_per_n[:10]) / np.sum(power_per_n)
power_in_last_10 = np.sum(power_per_n[-10:]) / np.sum(power_per_n)

print(f"Power in first 10 modes: {power_in_first_10*100:.1f}%")
print(f"Power in last 10 modes: {power_in_last_10*100:.1f}%")

if power_in_last_10 > 0.01:
    print("Warning: Significant power in high modes - may need larger radius")
```

**Pattern Comparison Metrics:**
```python
# Compare original vs reconstructed
theta_idx = pattern.theta_angles >= 0  # Front hemisphere

orig_gain = pattern.realized_gain()[0, theta_idx, :]
recon_gain = reconstructed_pattern.realized_gain()[0, theta_idx, :]

rms_error = np.sqrt(np.mean((orig_gain - recon_gain)**2))
max_error = np.max(np.abs(orig_gain - recon_gain))

print(f"RMS error: {rms_error:.3f} dB")
print(f"Max error: {max_error:.3f} dB")
```

---

## File Format Specifications

### TICRA .sph Format

The TICRA spherical wave coefficient format consists of:

**Header Records (8 lines):**
```
1. PRGTAG:     Program identification string
2. IDSTRG:     User description string
3. NTHE, NPHI, NMAX, MMAX:  Control integers
4-8.           Dummy placeholder records
```

**Control Parameters:**
- `NTHE`: Number of θ samples over 360° (must be even, ≥ 4)
- `NPHI`: Number of φ samples over 360° (must be ≥ 3)
- `NMAX`: Maximum polar index n (N in implementation)
- `MMAX`: Maximum azimuthal index |m| (M in implementation)

**Mode Coefficient Records:**

Organized by |m| from 0 to MMAX:

```
For each |m|:
  Record m.1:  m_value, POWERM
  Record m.2:  Coefficients for this |m|
```

For m=0:
```
Q_101, Q_201    (n=1)
Q_102, Q_202    (n=2)
...
Q_10N, Q_20N    (n=N)
```

For m≠0:
```
Q_1(-m)m, Q_2(-m)m, Q_1(+m)m, Q_2(+m)m    (n=m)
Q_1(-m)(m+1), Q_2(-m)(m+1), Q_1(+m)(m+1), Q_2(+m)(m+1)    (n=m+1)
...
```

**Coefficient Format:**
Each coefficient is written as 4 floats: `Re(Q_s(-m)n) Im(Q_s(-m)n) Re(Q_s(+m)n) Im(Q_s(+m)n)`

**Normalization Convention:**
```
Q'_smn = (1/√(8π)) × Q*_smn
```

where * denotes complex conjugation.

**Power Normalization:**
When total radiated power = 4π watts, then:
```
Σ |Q'_smn|² = 1
```

### Internal .npz Format

AntennaPattern also supports native NumPy format for efficient storage:

```python
# Save coefficients
from antenna_pattern.ant_io import save_swe_coefficients

save_swe_coefficients(
    swe_data,
    'coefficients.npz',
    metadata={'antenna': 'Patch', 'frequency_ghz': 10.0}
)

# Load coefficients
from antenna_pattern.ant_io import load_swe_coefficients

loaded_swe = load_swe_coefficients('coefficients.npz')
```

---

## Performance Considerations

### Computational Complexity

**Coefficient Extraction:**
- Legendre computation: O(N²M)
- FFT operations: O(N_θ N_φ log N_φ)
- Integration loop: O(N_θ NM)
- Total: O(N_θ N_φ log N_φ + N_θ N²M)

**Field Reconstruction:**
- Per evaluation point: O(N²M)
- For grid of P points: O(PN²M)

### Typical Performance

On modern hardware (Intel i7, 32GB RAM):

| Grid Size | N | M | Extraction Time | Reconstruction Time |
|-----------|---|---|----------------|-------------------|
| 181×73    | 30| 30| 2-3 seconds    | 1-2 seconds       |
| 361×73    | 50| 50| 5-8 seconds    | 3-5 seconds       |
| 361×361   | 50| 50| 15-25 seconds  | 8-15 seconds      |
| 721×361   | 100| 100| 60-120 seconds | 30-60 seconds   |

### Memory Requirements

Approximate memory usage:
```
Pattern storage:     8 × N_θ × N_φ × N_freq bytes
Q-coefficients:      16 × (2M+1) × N bytes (complex64)
Legendre cache:      16 × (2M+1) × N × N_θ bytes
```

For N=M=50, N_θ=361, N_φ=73:
- Coefficients: ~80 KB
- Cache: ~29 MB
- Pattern: ~0.4 MB

---

## Limitations and Considerations

### Numerical Accuracy

**Sources of Error:**
1. **Truncation error**: Finite N limits accuracy
2. **Integration error**: Discrete θ,φ sampling
3. **Machine precision**: ~15 digits for float64
4. **Legendre overflow**: Controlled by log-gamma approach

**Recommended Practices:**
- Use N ≥ kr₀ + 5 for high accuracy
- Ensure pattern sampling meets Nyquist: Δθ ≤ 180°/N
- Extend patterns to full sphere before extraction
- Check power convergence: last 10% of modes should contain <1% power

### Physical Validity

The SWE representation is valid for:
- **Source-free region**: r > r₀ (minimum sphere radius)
- **Linear media**: Free space or homogeneous lossless medium
- **Time-harmonic fields**: Single frequency analysis

Not applicable for:
- Near-field inside r₀
- Non-linear antenna problems
- Transient/time-domain analysis
- Inhomogeneous media

### Pattern Requirements

**Minimum Requirements:**
- Full-sphere coverage: θ ∈ [0°, 180°]
- Reasonable φ sampling: At least 3 points (Nyquist: >2M+1)
- Adequate θ sampling: Nyquist requirement Δθ ≤ π/N
- Continuous phase: No phase wrapping artifacts

**Common Issues:**
- Partial sphere patterns → extend with noise floor
- Undersampled patterns → aliasing in high modes
- Phase discontinuities → spurious modes
- Missing back hemisphere → reconstruction errors

---

## Troubleshooting

### Issue: Poor Reconstruction Accuracy

**Symptoms:** Reconstructed pattern differs significantly from input

**Possible Causes:**
1. **Insufficient modes**: N too small for antenna size
   - Solution: Increase radius or use higher power_threshold

2. **Undersampled input**: Pattern grid too coarse
   - Solution: Interpolate input to finer grid or verify Nyquist

3. **Incomplete sphere**: Missing back hemisphere data
   - Solution: Check extension parameters, increase noise_floor_db

4. **Phase errors**: Incorrect phase reference or wrapping
   - Solution: Verify phase continuity, check coordinate system

### Issue: Non-Convergence in Adaptive Mode

**Symptoms:** max_iterations reached, convergence=False

**Possible Causes:**
1. **Antenna too large**: Needs larger radius
   - Solution: Increase initial_radius or max_iterations

2. **Complex pattern**: Requires many modes
   - Solution: Increase convergence_threshold slightly

3. **Pattern artifacts**: Noise or measurement errors
   - Solution: Apply smoothing, check data quality

### Issue: Excessive Computation Time

**Possible Causes:**
1. **Oversampled pattern**: Grid finer than needed
   - Solution: Enable downsampling (downsample_factor=1.5-2.0)

2. **Very high N**: Large antenna or tight accuracy
   - Solution: Reduce power_threshold slightly (0.99 → 0.98)

3. **Inefficient grid**: Non-optimal θ,φ spacing
   - Solution: Let prepare_pattern_for_swe optimize grid

---

## References

### Primary Sources

1. **Hansen, J. E. (Ed.)** (1988). *Spherical Near-Field Antenna Measurements*. Peter Peregrinus Ltd., London. (Reprinted 2008 by IET)
   - Definitive reference for spherical wave theory
   - Chapter 2: Mathematical formulation
   - Appendix A: Normalized Legendre functions
   - Forms basis for TICRA GRASP implementation

2. **Ludwig, A. C.** (1971). "Near-Field Far-Field Transformations Using Spherical-Wave Expansions." *IEEE Transactions on Antennas and Propagation*, vol. 19, no. 2, pp. 214-220.
   - First practical numerical implementation
   - Orthogonality integration formulas
   - Validation against measured data

3. **Jensen, F., & Frandsen, A.** (2004). "On the Number of Modes in Spherical Wave Expansions." *Proceedings of AMTA*.
   - Mode truncation formulas
   - Power distribution analysis
   - Optimal N selection criteria

### Supporting References

4. **Davidson, D. B., & Sutinjo, A.** (2024). "Spherical Mode Reconstruction Test." *Radio Science*, Supporting Information.
   - Hansen convention clarifications
   - Phase convention comparisons
   - Validation examples

5. **Mahfouz, A. M., & Kishk, A. A.** (2024). "MATLAB-Based Fast Vector Spherical Wave Expansion Implementation." *IEEE APS Symposium*, pp. 1333-1334.
   - Fast recurrence algorithms
   - Computational optimization techniques
   - Validation benchmarks

6. **TICRA Engineering Consultants** (2020). *GRASP Technical Description*. Section 4.7: Spherical Wave Expansion.
   - File format specifications
   - Q-mode formulation
   - Software implementation details

### Mathematical References

7. **Stratton, J. A.** (1941). *Electromagnetic Theory*. McGraw-Hill, New York.
   - Vector spherical harmonics foundation
   - Legendre polynomial properties

8. **Abramowitz, M., & Stegun, I. A.** (1972). *Handbook of Mathematical Functions*. Dover, New York.
   - Special function definitions
   - Numerical computation methods

### Online Resources

9. **NIST Digital Library of Mathematical Functions**
   - URL: https://dlmf.nist.gov/
   - Section 10.47: Spherical Bessel Functions
   - Section 14: Legendre Functions

10. **SciPy Documentation**
    - scipy.special.lpmv: Associated Legendre polynomials
    - scipy.special.spherical_jn, spherical_yn: Spherical Bessel functions

---

## Appendix: Convention Comparison

### Phase Factor Conventions

Different implementations use different phase conventions for the term (-m/|m|)^m:

| Source | m=0 | m>0 | m<0 | Notes |
|--------|-----|-----|-----|-------|
| Hansen | 1 | (-1)^m | 1 | GRASP/TICRA standard |
| Davidson | 1 | (-1)^m | 1 | Matches Hansen |
| Stratton | 1 | (-1)^m | (-1)^m | Symmetric form |
| SciPy lpmv | Includes (-1)^m in P_n^m definition | Requires correction |

### Legendre Normalization

**Unnormalized:**
```
P_n^m(x) = (±1)^m (1-x²)^(m/2) d^m/dx^m [P_n(x)]
```

**Fully Normalized (Hansen):**
```
P̄_n^m(x) = √[(2n+1)/2 × (n-m)!/(n+m)!] × P_n^m(x)
```

Note: Some sources include (-1)^m in P_n^m definition, others don't. Verify carefully when comparing implementations.

---

*Document Version: 1.0*  
*Last Updated: 2025*  
*AntennaPattern SWE Implementation*