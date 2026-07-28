# Functional Connectivity Features — From Absolute Zero

---

## The Problem: How Do Brain Regions Talk to Each Other?

You have fMRI data from many subjects. Each subject's brain is divided into 414 regions (using an atlas). For each region, you have a time series—how active that region was at each time point during the scan.

You want to know: **Which brain regions are CONNECTED?** Which ones rise and fall together?

---

## The Big Picture

```
414 brain regions × T time points per subject
         │
         ▼
For each subject, compute the CORRELATION between every pair of regions
         │
         ▼
You get (414 × 413) / 2 = 85,491 numbers per subject
Each number = how strongly two regions are connected
         │
         ▼
Add phenotype data (age, sex, diagnosis)
         │
         ▼
One big table: subjects × (85,491 features + phenotype columns)
```

---

## Toy Example — 3 Brain Regions, 4 Time Points

---

### Step 1: The fMRI Data

One subject. 3 brain regions. 4 time points.

```
Time:         t1   t2   t3   t4
Region 1:      2    4    6    4      ← motor cortex
Region 2:      1    2    3    2      ← thalamus
Region 3:      5    3    1    3      ← visual cortex
```

Each row is one brain region's activity over time. Each column is one moment in the scan.

---

### Step 2: Compute Correlation Between Every Pair of Regions

Correlation measures: "Do these two time series go up and down TOGETHER?"

---

**Region 1 vs. Region 2:**

```
Region 1: [2, 4, 6, 4]    mean = 4
Region 2: [1, 2, 3, 2]    mean = 2
```

Center both (subtract means):

```
Region 1: [-2, 0, 2, 0]
Region 2: [-1, 0, 1, 0]
```

Correlation = (sum of products) / (sqrt(sum of squares of R1) × sqrt(sum of squares of R2))

```
Sum of products: (-2)(-1) + 0×0 + 2×1 + 0×0 = 2 + 0 + 2 + 0 = 4

Sum of squares R1: (-2)² + 0² + 2² + 0² = 4 + 0 + 4 + 0 = 8
Sum of squares R2: (-1)² + 0² + 1² + 0² = 1 + 0 + 1 + 0 = 2

Correlation = 4 / sqrt(8 × 2) = 4 / sqrt(16) = 4/4 = 1.0
```

**Correlation = 1.0 → Perfectly connected.** Region 1 and Region 2 move EXACTLY together.

---

**Region 1 vs. Region 3:**

```
Region 1: [2, 4, 6, 4]    mean = 4
Region 3: [5, 3, 1, 3]    mean = 3
```

Centered:
```
Region 1: [-2, 0, 2, 0]
Region 3: [2, 0, -2, 0]
```

```
Sum of products: (-2)(2) + 0×0 + 2×(-2) + 0×0 = -4 + 0 - 4 + 0 = -8

Sum of squares R1: 8
Sum of squares R3: 2² + 0² + (-2)² + 0² = 4 + 0 + 4 + 0 = 8

Correlation = -8 / sqrt(8 × 8) = -8/8 = -1.0
```

**Correlation = −1.0 → Perfectly ANTI-connected.** When Region 1 goes up, Region 3 goes DOWN.

---

**Region 2 vs. Region 3:**

```
Region 2: [1, 2, 3, 2]    mean = 2
Region 3: [5, 3, 1, 3]    mean = 3
```

Centered:
```
Region 2: [-1, 0, 1, 0]
Region 3: [2, 0, -2, 0]
```

```
Sum of products: (-1)(2) + 0×0 + 1×(-2) + 0×0 = -2 + 0 - 2 + 0 = -4
Sum of squares R2: 2, Sum of squares R3: 8
Correlation = -4 / sqrt(16) = -4/4 = -1.0
```

**Correlation = −1.0 → Perfectly anti-connected.**

---

### Step 3: The Connectivity Matrix

For 3 regions, we have 3×2/2 = 3 unique pairs:

```
R1-R2: +1.0    ← strongly connected
R1-R3: -1.0    ← strongly anti-connected
R2-R3: -1.0    ← strongly anti-connected
```

---

### Step 4: Vectorize — Flatten Into One Row

Instead of a 3×3 matrix, we flatten the upper triangle into a list:

```
Subject 1 features: [R1-R2=1.0, R1-R3=-1.0, R2-R3=-1.0]
```

For 414 regions, we get (414×413)/2 = **85,491 features per subject.**

---

### Step 5: Add Phenotype Data

```
Subject ID | Age | Sex | Diagnosis | R1-R2 | R1-R3 | R2-R3 | ... (85,491 more)
   001     | 45  |  M  |     1     |  1.0  | -1.0  | -1.0  | ...
   002     | 32  |  F  |     0     |  0.3  |  0.5  | -0.2  | ...
   ...
```

Each row = one subject. Columns = connectivity features + phenotype info.

---

## What the Code Does — Step by Step

### 1. `load_fmri_data(path)`

Loads the fMRI data from an NPZ file. The data has shape `(subjects, time_points, regions)` or `(subjects, regions, time_points)`.

**Input:** File path.  
**Output:** 3D numpy array — subjects × regions × time.

---

### 2. `load_phenotype_data(path, required_cols)`

Loads the CSV with subject info (age, sex, diagnosis).

**Input:** File path, list of required column names.  
**Output:** DataFrame with subject ID, age, sex, and target column (diagnosis).

---

### 3. `load_atlas_labels(path, region_count)`

Loads the brain atlas—a CSV with 414 region names like "Motor_Cortex_L", "Thalamus_R", etc.

**Input:** File path, expected number of regions (414).  
**Output:** List of 414 region names. Verifies the count matches.

---

### 4. `compute_connectivity_features(fmri_data, region_count)`

This is the core. For EACH subject, it computes the correlation between every pair of brain regions.

Using `ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)`:

- **kind="correlation":** Use Pearson correlation.
- **vectorize=True:** Flatten the matrix into a 1D array (one row per subject).
- **discard_diagonal=True:** Don't include self-connections (R1-R1 = always 1.0, useless).
- **standardize=False:** Don't z-score the time series (correlation already handles scaling).

**Input:** 3D fMRI data (subjects × regions × time).  
**Output:** 2D array — `(subjects × 85,491 features)`. Each feature is one region-pair correlation.

---

### 5. `generate_feature_names(atlas_labels)`

Creates human-readable names for each feature:

```
"Motor_Cortex_L - Thalamus_R"
"Motor_Cortex_L - Visual_Cortex_L"
...
```

**Input:** List of 414 region names.  
**Output:** List of 85,491 strings, one per region pair.

---

### 6. `create_connectivity_dataset(...)`

Orchestrates everything:

1. Load fMRI data.
2. Load phenotype data.
3. Load atlas labels.
4. Compute connectivity features (correlations).
5. Generate feature names.
6. Add subject IDs.
7. Merge with phenotype data.
8. Rename the target column to "Label".
9. Save as CSV.

**Output:** One big CSV — subjects × (phenotype columns + 85,491 connectivity features).

---

## Why This Matters

This transforms raw fMRI time series into a format that machine learning models can use. Instead of "here's 414 wiggly lines per subject," you get "here's how strongly each pair of brain regions is connected." These connectivity patterns can then be used to predict diagnosis, age, or other clinical variables.

---

## One Sentence

**Functional connectivity takes fMRI time series from 414 brain regions, computes the correlation between every pair of regions for each subject, and produces a table where each row is a subject and each column is a region-pair connection strength—85,491 features that capture how the brain is wired.**
