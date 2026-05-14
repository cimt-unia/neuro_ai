# Feature Engineering in Signal Processing: 
From Raw Data to Machine Learning

<img width="506" height="244" alt="image" src="https://github.com/user-attachments/assets/d22363df-fd37-45a1-bf75-cd08b49d8eab" />


## The Core Problem

You have a continuous neural signal. You want to classify brain states (e.g., "Rest" vs. "Movement"). But you cannot simply feed raw voltage values into a classifier like XGBoost or SVM effectively. Why?

1.  **High Dimensionality:** A few seconds of data at 500Hz contains thousands of data points.
2.  **Noise:** Raw signals are contaminated by drift, line noise, and artifacts.
3.  **Lack of Interpretability:** A single voltage point tells you little about the underlying neural oscillation.

**We need to extract *features*:** compact, meaningful numerical representations of the signal that capture its essential characteristics.

<img width="511" height="455" alt="image" src="https://github.com/user-attachments/assets/84d5195a-a233-4c61-a2a0-0145bf98e1f9" />


## Feature Extraction

When dealing with time-series signals, you generally have three options:

### 1. Raw Time-Domain Features
*   **Method:** Use the raw voltage samples directly as input features.
*   **Pros:** No information loss from transformation.
*   **Cons:** Extremely high dimensionality; sensitive to phase shifts and noise; requires massive amounts of data to train.


### 2. Frequency Band Power (Classical Machine Learning) 
*   **Method:** Decompose the signal into standard frequency bands (Delta, Theta, Alpha, Beta, Gamma) and calculate the power (energy) in each band.
*   **Pros:** Highly interpretable (neuroscientists understand "Beta desynchronization"); low dimensionality; works excellently with classical ML models like XGBoost.
*   **Cons:** Loses fine-grained temporal resolution within the epoch.

### 3. Spectrogram/Image-Based (Deep Learning)
*   **Method:** Convert the signal into a 2D Spectrogram (Time x Frequency) and treat it as an image for a Convolutional Neural Network (CNN).
*   **Pros:** Captures complex time-frequency dynamics.
*   **Cons:** Requires deep learning frameworks; computationally expensive; "black box" interpretation.

---

## Framework Example


### Step 1: Preprocessing (Cleaning the Signal)
Before extracting features, we must clean the signal.
*   **IIR Filtering:** We apply a High-Pass filter to remove slow drifts and a Low-Pass filter to remove high-frequency noise.
*   **Notch Filter:** We remove line noise using a FIR notch filter.
*   **Z-Score Normalization:** We standardize the signal to have a mean of 0 and standard deviation of 1. This ensures that amplitude differences don't bias the model.

### Step 2: Epoching (Cutting the Signal)
We cannot analyze the entire continuous recording at once if we want to track changes over time. We cut the signal into smaller chunks called **Epochs**.
*   **Window Size:** We use **1.5-second windows**.
    *   *Why 1.5s?* To resolve low frequencies like Delta (1-4 Hz), you need at least one full cycle. A 1-second window only gives you 1 Hz resolution. A 1.5s window provides better frequency resolution for lower bands.
*   **Overlap:** We use **75% overlap**.
    *   *Why?* This increases the number of training samples for our ML model and smooths out transient artifacts.

### Step 3: Feature Extraction (The Transformation)
For each 1.5s epoch, we compute the **Power Spectral Density (PSD)** using Welch’s method. Then, we integrate the power under the curve for specific frequency bands:
*   Delta (1-4 Hz)
*   Theta (4-8 Hz)
*   Alpha (8-12 Hz)
*   Beta (13-30 Hz)
*   Low Gamma (30-50 Hz)
*   High Gamma (50-100 Hz)

**Result:** Each 1.5s epoch is now represented by just **6 numbers** (the power in each band) instead of 750 raw voltage samples (1.5s * 500Hz).

### Step 4: Organizing the Data for ML
Machine learning models expect a **Tabular Format** (Rows = Samples, Columns = Features).
1.  We create a DataFrame where each row is one epoch.
2.  We add a `target` column: `'rest'` or `'move'`.
3.  We concatenate all epochs into a single CSV file (`M1_Hand_Knob_ML_Features.csv`).

### Step 5: Classification & Stability Analysis
Finally, we use **XGBoost** to classify the epochs based on their band powers.
*   **Training:** We split the data into training and testing sets.
*   **Evaluation:** We check Accuracy, F1-Score, and AUC.
*   **Stability Check:** A crucial step! We retrain the model 50 times with different random seeds. If a feature (e.g., "Beta Power") is important in *all* 50 runs, it is a **stable** biomarker. If it only appears occasionally, it might be noise.

## Summary

*   **Filtering is non-negotiable:** Never feed raw, unfiltered neural data into an ML model.
*   **Epoch length matters:** Your epoch length dictates your frequency resolution. $Resolution = 1 / Duration$.
*   **Features > Raw Data:** For tabular ML models, engineered features (like Band Power) often outperform raw data because they embed domain knowledge (neuroscience) into the model.
*   **Stability is key:** A high accuracy score means nothing if the model relies on different features every time you shuffle the data. Always check feature stability.
