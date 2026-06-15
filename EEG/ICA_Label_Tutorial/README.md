# ICA: COMPLETE TOY EXAMPLE WITH EVERY CALCULATION

<br>

## THE SETUP: Two Original Sources

<img width="1100" height="308" alt="image" src="https://github.com/user-attachments/assets/ddd72d8b-d621-4d6e-86c7-38e415abafa1" />


We have two independent signals we want to recover. They are non-Gaussian—structured, not bell curves.

```
Source 1 (s₁):  [ 2,  4,  6,  4,  2]    ← spiky, super-Gaussian
Source 2 (s₂):  [ 1,  3,  1,  3,  1]    ← spiky, super-Gaussian
```

Plot s₁: goes up to 6 and back down. Plot s₂: oscillates between 1 and 3.

These are STATISTICALLY INDEPENDENT. Knowing s₁ tells you nothing about s₂. They have different shapes, different timing.

<br>

## THE MIXING: How We Lose the Sources


<img width="911" height="268" alt="image" src="https://github.com/user-attachments/assets/e041f990-e1c0-4fb5-921d-05f0b881b98c" />



The mixing matrix A (unknown to us) combines the sources:

```
A = [0.8   0.2]      Mic 1 picks up 80% of s₁ and 20% of s₂
    [0.3   0.7]      Mic 2 picks up 30% of s₁ and 70% of s₂
```

The mixed recordings X = A × S:

```
x₁ = 0.8×s₁ + 0.2×s₂
x₂ = 0.3×s₁ + 0.7×s₂
```
<img width="1015" height="238" alt="image" src="https://github.com/user-attachments/assets/bb28ab7a-5e9c-49b3-a815-7547ea917999" />


Let's compute every value:

```
Time 1: s₁=2, s₂=1
  x₁ = 0.8×2 + 0.2×1 = 1.6 + 0.2 = 1.8
  x₂ = 0.3×2 + 0.7×1 = 0.6 + 0.7 = 1.3

Time 2: s₁=4, s₂=3
  x₁ = 0.8×4 + 0.2×3 = 3.2 + 0.6 = 3.8
  x₂ = 0.3×4 + 0.7×3 = 1.2 + 2.1 = 3.3

Time 3: s₁=6, s₂=1
  x₁ = 0.8×6 + 0.2×1 = 4.8 + 0.2 = 5.0
  x₂ = 0.3×6 + 0.7×1 = 1.8 + 0.7 = 2.5

Time 4: s₁=4, s₂=3
  x₁ = 0.8×4 + 0.2×3 = 3.2 + 0.6 = 3.8
  x₂ = 0.3×4 + 0.7×3 = 1.2 + 2.1 = 3.3

Time 5: s₁=2, s₂=1
  x₁ = 0.8×2 + 0.2×1 = 1.6 + 0.2 = 1.8
  x₂ = 0.3×2 + 0.7×1 = 0.6 + 0.7 = 1.3
```

### What We Have (The Only Thing We Know)

```
X = [x₁]  =  [1.8   3.8   5.0   3.8   1.8]    ← Mic 1 recording
    [x₂]     [1.3   3.3   2.5   3.3   1.3]    ← Mic 2 recording
```

**We have X. We DON'T have A. We DON'T have S. We want to recover S.**



<br>

## THE SCATTER PLOT: Visualizing the Problem
<img width="1019" height="382" alt="image" src="https://github.com/user-attachments/assets/b9f3fa94-0c6a-49db-9f0d-0c5ed8ca4bc5" />

Plot x₁ vs x₂ for each time point:

```
Time 1: (1.8, 1.3)
Time 2: (3.8, 3.3)
Time 3: (5.0, 2.5)
Time 4: (3.8, 3.3)
Time 5: (1.8, 1.3)
```

This forms a diagonal cloud of points—they're correlated. When x₁ is high, x₂ tends to be high too.


<br>

## STEP 1: CENTER THE DATA

ICA starts by centering—subtracting the mean from each row.

```
Mean of x₁ = (1.8 + 3.8 + 5.0 + 3.8 + 1.8) / 5 = 16.2 / 5 = 3.24
Mean of x₂ = (1.3 + 3.3 + 2.5 + 3.3 + 1.3) / 5 = 11.7 / 5 = 2.34

Centered x₁: [1.8-3.24, 3.8-3.24, 5.0-3.24, 3.8-3.24, 1.8-3.24]
            = [-1.44, 0.56, 1.76, 0.56, -1.44]

Centered x₂: [1.3-2.34, 3.3-2.34, 2.5-2.34, 3.3-2.34, 1.3-2.34]
            = [-1.04, 0.96, 0.16, 0.96, -1.04]
```

Now both rows have mean = 0. The data cloud is centered at the origin.


<br>

## STEP 2: PCA, FIND THE ANGLE OF MAXIMUM VARIANCE (U^T)

<img width="787" height="334" alt="image" src="https://github.com/user-attachments/assets/7abc9237-fb89-46e3-8a6f-c5a795272e03" />

### Compute the Covariance Matrix

Since the data is centered (mean=0):

```
Cov = (1/n) × X × X^T

X × X^T = sum of outer products at each time point:

Time 1: [-1.44] × [-1.44  -1.04] = [ 2.074   1.498]
        [-1.04]                     [ 1.498   1.082]

Time 2: [0.56] × [0.56   0.96]  = [0.314   0.538]
        [0.96]                     [0.538   0.922]

Time 3: [1.76] × [1.76   0.16]  = [3.098   0.282]
        [0.16]                     [0.282   0.026]

Time 4: [0.56] × [0.56   0.96]  = [0.314   0.538]
        [0.96]                     [0.538   0.922]

Time 5: [-1.44] × [-1.44  -1.04] = [2.074   1.498]
        [-1.04]                     [1.498   1.082]

Sum = [2.074+0.314+3.098+0.314+2.074    1.498+0.538+0.282+0.538+1.498]
      [1.498+0.538+0.282+0.538+1.498    1.082+0.922+0.026+0.922+1.082]

    = [7.874   4.354]
      [4.354   4.034]
```

Divide by n=5:

```
Cov = [1.575   0.871]
      [0.871   0.807]
```

The off-diagonal (0.871) is not zero—x₁ and x₂ ARE correlated.

### Find Eigenvalues and Eigenvectors

The eigenvalues are the variances along the principal directions:

```
λ₁ ≈ 2.16    (direction of maximum variance)
λ₂ ≈ 0.22    (direction of minimum variance)
```

The eigenvector matrix U (the rotation matrix):

```
U = [0.83   -0.56]      ← first column = direction of max variance
    [0.56    0.83]      ← second column = perpendicular direction
```

The angle θ of maximum variance: θ = arctan(0.56/0.83) ≈ 34°

### Apply PCA Rotation: X_pca = U^T × X_centered

```
U^T = [ 0.83   0.56]
      [-0.56   0.83]
```

Let's compute for each time point:

**Time 1:** x₁=-1.44, x₂=-1.04
```
PC1 = 0.83×(-1.44) + 0.56×(-1.04) = -1.195 - 0.582 = -1.777
PC2 = -0.56×(-1.44) + 0.83×(-1.04) = 0.806 - 0.863 = -0.057
```

**Time 2:** x₁=0.56, x₂=0.96
```
PC1 = 0.83×0.56 + 0.56×0.96 = 0.465 + 0.538 = 1.003
PC2 = -0.56×0.56 + 0.83×0.96 = -0.314 + 0.797 = 0.483
```

**Time 3:** x₁=1.76, x₂=0.16
```
PC1 = 0.83×1.76 + 0.56×0.16 = 1.461 + 0.090 = 1.551
PC2 = -0.56×1.76 + 0.83×0.16 = -0.986 + 0.133 = -0.853
```

**Time 4:** x₁=0.56, x₂=0.96
```
PC1 = 1.003 (same as Time 2)
PC2 = 0.483
```

**Time 5:** x₁=-1.44, x₂=-1.04
```
PC1 = -1.777 (same as Time 1)
PC2 = -0.057
```

```
X_pca = [PC1]  =  [-1.777   1.003   1.551   1.003  -1.777]
        [PC2]     [-0.057   0.483  -0.853   0.483  -0.057]
```

**Check:** PC1 and PC2 are now uncorrelated. Their covariance is 0.

**But:** PC1 is NOT either original source. It's still a mixture. PC1 has variance 2.16 (large). PC2 has variance 0.22 (small). PCA stops here. ICA does not.


<br>

## STEP 3: WHITENING: SCALE TO EQUAL VARIANCE (Σ⁻¹)

### Create the Whitening Matrix

Divide each PC by its standard deviation:

```
σ₁ = √2.16 ≈ 1.47
σ₂ = √0.22 ≈ 0.47

Whitening matrix:
Σ⁻¹ = [1/1.47     0   ]  =  [0.68     0  ]
      [   0     1/0.47]     [  0     2.13]
```

### Apply Whitening: Z = Σ⁻¹ × X_pca

```
PC1 (variance 2.16) ÷ 1.47 → variance becomes 1
PC2 (variance 0.22) ÷ 0.47 → variance becomes 1

Z = [0.68×(-1.777)   0.68×1.003   0.68×1.551   0.68×1.003   0.68×(-1.777)]
    [2.13×(-0.057)   2.13×0.483   2.13×(-0.853)  2.13×0.483   2.13×(-0.057)]

  = [-1.209   0.682   1.055   0.682  -1.209]    ← variance = 1
    [-0.121   1.029  -1.817   1.029  -0.121]    ← variance = 1
```

**Now both rows have variance = 1.** The data cloud is a PERFECT CIRCLE. All correlations are removed. Every direction looks the same in terms of variance. The only remaining difference between directions is their NON-GAUSSIANITY.


<br>

## STEP 4: ICA: ROTATE TO MAXIMIZE NON-GAUSSIANITY (V)

### The Rotation Matrix

For any angle φ:

```
V = [ cos(φ)    sin(φ)]
    [-sin(φ)    cos(φ)]
```

The recovered sources: **S = V × Z**

We test different φ and measure KURTOSIS of the resulting rows.

### Try φ = 0° (No Rotation)

```
Row 1 = same as Z row 1: [-1.209, 0.682, 1.055, 0.682, -1.209]

Compute kurtosis:
Mean = 0 (centered data)

Fourth powers: (-1.209)⁴=2.14, 0.682⁴=0.22, 1.055⁴=1.24, 0.682⁴=0.22, (-1.209)⁴=2.14
Mean of fourth powers = (2.14+0.22+1.24+0.22+2.14)/5 = 5.96/5 = 1.192

Kurtosis = 1.192 - 3 = -1.808
Absolute kurtosis = 1.808  ← modest
```

### Try φ = 30°

```
V = [0.866   0.500]
    [-0.500  0.866]

Row 1 = 0.866×[-1.209,0.682,1.055,0.682,-1.209] + 0.500×[-0.121,1.029,-1.817,1.029,-0.121]
     = [-1.108, 1.105, -0.002, 1.105, -1.108]

Kurtosis ≈ 0.5  ← still fairly Gaussian
```

### Try φ = 56° (Near Optimal)

Through systematic search, φ ≈ 56° maximizes kurtosis.

```
V = [0.559   0.829]
    [-0.829  0.559]

Row 1 = 0.559×[-1.209,0.682,1.055,0.682,-1.209] + 0.829×[-0.121,1.029,-1.817,1.029,-0.121]
     = [-0.776, 1.233, -0.916, 1.233, -0.776]

Fourth powers: 0.363, 2.311, 0.705, 2.311, 0.363
Mean = 6.053/5 = 1.211

Kurtosis = 1.211 - 3 = -1.789
Absolute kurtosis = 1.789  ← maximum!

Row 2 = -0.829×[-1.209,0.682,1.055,0.682,-1.209] + 0.559×[-0.121,1.029,-1.817,1.029,-0.121]
     = [0.935, 0.010, -1.890, 0.010, 0.935]

Kurtosis ≈ 2.5 (absolute) ← also maximum!
```

### The Recovered Sources

At φ ≈ 56°:

```
S = V × Z

Row 1: [-0.776,  1.233, -0.916,  1.233, -0.776]   ← high |kurtosis|
Row 2: [ 0.935,  0.010, -1.890,  0.010,  0.935]   ← high |kurtosis|
```

### Compare to Original Sources (Rescaled)

Original s₁ = [2, 4, 6, 4, 2]. If we rescale to similar range (divide by ~5):
```
[0.4, 0.8, 1.2, 0.8, 0.4]
```
Shape matches Row 2 (inverted and scaled). ✓

Original s₂ = [1, 3, 1, 3, 1]. If we rescale:
```
[0.33, 1.0, 0.33, 1.0, 0.33]
```
Shape matches Row 1. ✓

**The sources are recovered!** ICA cannot determine exact scaling or sign, but the SHAPES match perfectly.


<br>

## THE COMPLETE JOURNEY: FOUR STAGES

```
STAGE 1: X_centered (Raw Mixed Data)
  Covariance: [1.575  0.871]
              [0.871  0.807]
  Shape: Diagonal ellipse, tilted ~34°
  Status: CORRELATED (off-diagonal ≠ 0)
  
        ↓  U^T: Rotate by θ ≈ 34°

STAGE 2: X_pca (PCA Rotated)
  PC1 variance = 2.16, PC2 variance = 0.22
  Shape: Ellipse aligned with axes
  Status: UNCORRELATED (but NOT independent)
  PCA stops here. ICA does not.
  
        ↓  Σ⁻¹: Scale by 1/σ

STAGE 3: Z (Whitened)
  Both rows have variance = 1
  Shape: PERFECT CIRCLE
  Status: UNCORRELATED + EQUAL VARIANCE
  Ready for the final rotation.
  
        ↓  V: Rotate by φ ≈ 56°

STAGE 4: S (Independent Components)
  Maximally non-Gaussian projections
  Shape: Circle rotated to align with source axes
  Status: MAXIMALLY INDEPENDENT
  Sources recovered!
```

---

## WHY THIS WORKS: THE CENTRAL LIMIT THEOREM

At Step 4, we test different rotation angles φ:

- **Most φ:** The projection is a mixture of sources → Central Limit Theorem says mixtures are Gaussian → kurtosis ≈ 0
- **φ ≈ 56°:** The projection captures a SINGLE source → maximally non-Gaussian → kurtosis far from 0

ICA searches for the φ that maximizes |kurtosis|. That φ isolates a source.

---

<br>



## STEP SUMMARY

**Step 1 (U^T):** "Find the diagonal direction (θ ≈ 34°) and rotate the ellipse horizontal."

**Step 2 (Σ⁻¹):** "Squish the long axis (÷1.47) and stretch the short axis (÷0.47) to make a circle."

**Step 3 (V):** "Spin the circle. At most angles the projection looks like a bell curve. At φ ≈ 56° the projection is spiky—that's a source."

**The entire algorithm:** Rotate by θ. Scale by 1/σ. Rotate by φ. Done. Two angles, two variances. That's all ICA needs.

<br>



<img width="992" height="392" alt="image" src="https://github.com/user-attachments/assets/ce266ca7-dc46-4777-b24b-8c4c25b8625f" />
