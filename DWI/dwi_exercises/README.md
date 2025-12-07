# **Understanding Diffusion MRI** 


### Feature Extraction through Code Exploration

Imagine you have just started a new position in a neuroscience or medical imaging lab. Your supervisor gives you a dataset containing raw diffusion MRI (dMRI) scans from patients diagnosed with either **Multiple Sclerosis (MS)** or **Alzheimer’s Disease**. Your task is to prepare this data for **machine learning**, but you have never worked with diffusion MRI before.

To begin, you locate **two Jupyter notebooks** on GitHub:

1. **`01_dwi_exploration.ipynb`** – A tutorial demonstrating how to load, visualize, and model dMRI data from a single subject.  
2. **`02_feature_engineering.ipynb`** – A pipeline that preprocesses the data, performs tractography, and extracts quantitative features (e.g., FA profiles, tract lengths).

---

## The Exercise

Your goal is to thoroughly understand both notebooks, line by line. This involves not merely running code but comprehending the theory, intuition, and purpose behind each function, variable, and step.

### What You Will Do

#### 1. **Begin with One Subject**
- Load and examine diffusion MRI data (Stanford HARDI dataset).
- Understand the data structure:
  - Interpret dimensions such as `(81, 106, 76, 160)`.
  - Define **DWI** (Diffusion-Weighted Imaging).
  - Explain **b-values** and **b-vectors**, including why many diffusion directions are required.

#### 2. **Analyze the First Notebook (`01_dwi_exploration.ipynb`)**
- Review the code line by line.
- For each DIPY function (e.g., `median_otsu`, `TensorModel`, `ConstrainedSphericalDeconvModel`), consult the [DIPY documentation](https://dipy.org/).
- Explain in your own words:
  - What each part of the code does.
  - Why it is necessary.
  - The meaning of its inputs and outputs (e.g., FA as a scalar in the range 0–1).
- Connect code components to core neuroimaging concepts:
  - Diffusion tensor modeling vs. CSD.
  - Microstructural metrics (FA, MD, AD, RD, GFA).
  - Fiber orientation and tractography fundamentals.

#### 3. **Analyze the Second Notebook (`02_feature_engineering.ipynb`)**
- Trace how features suitable for machine learning are derived:
  - Preprocessing: denoising (`patch2self`), brain masking.
  - Reconstruction: CSD for fiber orientation estimation.
  - Tractography: probabilistic tracking with FA-based stopping criteria.
  - Bundle extraction: ROI-based identification of the corpus callosum (CC) and corticospinal tract (CST).
  - Feature computation: `afq_profile`, streamline length, centroid, count metrics.
- For each feature, consider:
  - What is being computed.
  - Why it is clinically relevant  
    (e.g., reduced CC FA may indicate demyelination in MS; CST alterations may reflect neurodegeneration).
  - How it would be incorporated into a machine learning pipeline  
    (e.g., as part of a feature matrix for classification).

#### 4. **Document Your Understanding**
- Add detailed comments directly into both notebooks.
- Maintain a log of:
  - Questions  
  - Uncertainties  
  - Key insights  

#### 5. **Prepare for Discussion**
- In the next session, groups will compare interpretations.
- The discussion will address:
  - Biological plausibility and interpretability of extracted features.
  - Assumptions embedded in the pipeline  
    (e.g., tractography as a proxy for anatomy, reliability of CSD in resolving crossings).
  - Potential improvements for real research workflows  
    (e.g., automated bundle extraction, scaling to BIDS datasets, adding motion correction).



<img width="1134" height="547" alt="dwti5" src="https://github.com/user-attachments/assets/d98c2f82-9275-4fe2-a564-eef4336536ae" />



