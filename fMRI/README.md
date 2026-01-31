# Machine Learning on Resting-State fMRI Connectivity

<img width="448" height="371" alt="fMRI - Neuro AI" src="https://github.com/user-attachments/assets/3e139f24-034f-47a5-9d3a-f97a6694b288" />


This example demonstrates the **standard pipeline for applying machine learning to resting-state functional MRI (rs-fMRI) data** to study brain connectivity. It is designed to help students understand how neuroimaging data is transformed into a format suitable for classification or regression tasks—**without focusing on any specific disorder**.



---

## 1. Resting-State fMRI (rs-fMRI) Basics
- Participants lie quietly in the scanner with eyes closed (or fixated), **not performing any task**.
- The goal is to measure **spontaneous, low-frequency fluctuations** in blood oxygen level–dependent (BOLD) signals.
- These fluctuations reflect **intrinsic functional networks**—brain regions that "talk" to each other even at rest.
- Data are typically acquired using **echo-planar imaging (EPI)**, which enables fast whole-brain coverage.

---

## 2. From Brain Images to Features

### Step 1: Preprocessing
Raw fMRI data undergo standard preprocessing:
- Motion correction
- Slice-timing correction
- Spatial normalization to a standard brain template
- Nuisance regression (e.g., motion parameters, white matter, CSF)
- Bandpass filtering (to retain low-frequency BOLD signals)

### Step 2: Parcellation
The brain is divided into **regions of interest (ROIs)** using an atlas (e.g., DiFuMo with 64 ROIs).  
→ Each ROI represents a functionally or anatomically defined brain area.

### Step 3: Extract Time Series
For each subject, the average BOLD time series is extracted from each ROI.  
→ Result: a matrix of size *(T timepoints × R regions)*.

### Step 4: Compute Functional Connectivity
- Calculate **Pearson correlation** between every pair of ROI time series.
- This yields a **connectivity matrix** of size *R × R* (e.g., 64 × 64).
- The matrix is symmetric; only the upper triangle is used (excluding diagonal).
- Number of unique connections = \( \frac{R(R-1)}{2} \) → e.g., **2,016 features** for 64 ROIs.

### Step 5: Build Feature Matrix
- Flatten each subject’s connectivity matrix into a 1D vector.
- Stack all subjects’ vectors into a **design matrix** of shape *(N subjects × F features)*.
  - Example: 40 subjects × 2,016 features.

---

## 3. Machine Learning Pipeline

### Data Preparation
- **Normalization**: Apply z-score standardization (mean = 0, std = 1) across features.
- **Train/Test Split**: Typically 60%/40% or similar, ensuring balanced class representation if supervised.
- **No labels required for unsupervised tasks** (e.g., clustering); labels are only needed for classification/regression.

### Model Training (Example: SVM)
- Use a classifier like **Support Vector Machine (SVM)** to learn patterns in connectivity.
- **Hyperparameter tuning** (e.g., via grid search + cross-validation) prevents overfitting.
- Output: predictions (e.g., binary classes) and model weights.

### Interpretation
- In linear models (e.g., linear SVM), **feature weights** indicate which connections drive predictions.
- High-weight connections can be mapped back to brain regions for neuroscientific insight.
- Visualization: heatmap of weights over the connectivity matrix, or network graphs.

