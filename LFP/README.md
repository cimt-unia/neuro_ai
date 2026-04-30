# Signal Processing Tutorial: M1 Hand Knob Analysis

**Subject:** Right-Hemisphere Primary Motor Cortex (M1) Hand Knob  
**Data Source:** M1 Time Series  
**Original Sampling Rate ($F_s$):** 500 Hz  
**Target Sampling Rate:** 250 Hz  

**Download data:** https://drive.google.com/file/d/1_ndA6MLgOgADUKeEEPhcsA3mD3ox2ymz/view?usp=sharing

## Overview

This tutorial demonstrates a comprehensive pipeline for processing neurophysiological time-series data. It covers essential preprocessing steps, linear and non-linear filtering techniques, spectral analysis, and statistical normalization. The primary objective is to analyze Event-Related Desynchronization (ERD) in the Beta band during motor tasks compared to rest conditions.

---

## Table of Contents

1.  [Imports & Data Loading](#section-0-imports--data-loading)
2.  [Resampling & Decimation](#section-1-resampling-decimation)
3.  [Linear Filtering: IIR Butterworth](#section-2-linear-filtering-iir-butterworth)
4.  [Linear Filtering: FIR Window Design](#section-3-linear-filtering-fir-window-design)
5.  [Phase Delay & Edge Artefacts (`filtfilt` vs `lfilter`)](#section-4-filtfilt-vs-lfilter--gustafsson-edge-method)
6.  [Non-Linear Filtering: Median & Savitzky-Golay](#section-5-non-linear-filtering-median--savitzky-golay)
7.  [Convolution for Time-Series](#section-6-convolution-for-time-series)
8.  [Frequency-Swept Signals (Chirp)](#section-7-frequency-swept-signals-chirp)
9.  [Kalman Filter (1-D Scalar)](#section-8-kalman-filter-1-d-scalar)
10. [Z-Score Normalisation](#section-9-z-score-normalisation)
11. [Power Spectral Density: Welch’s Method](#section-10-power-spectral-density-welchs-method)
12. [Beta-Band Power Ratio (ERD Quantification)](#section-11-beta-band-power-ratio)
13. [Spectrogram Computation](#section-12-spectrogram-computation)

---

## Section 0: Imports & Data Loading

### Objective
Initialize the computational environment and load raw time-series data for two conditions:
1.  **Rest:** Eyes Closed.
2.  **Move:** Left Hand Movement.

### Key Parameters
*   **Library Stack:** `numpy`, `matplotlib`, `scipy.signal`, `scipy.ndimage`.
*   **Data Files:**
    *   `R_M1_hand_knob_voxel_EyesClosed_Rest.npy`
    *   `R_M1_hand_knob_voxel_LeftHand_Move.npy`

### Procedure
1.  Configure matplotlib for high-DPI output and clean aesthetics.
2.  Load `.npy` arrays into memory.
3.  Verify signal dimensions, duration, and amplitude ranges.
4.  Visualize raw signals to inspect baseline noise and gross artifacts.

---

## Section 1: Resampling (Decimation)

### Objective
Reduce the sampling rate from 500 Hz to 250 Hz to decrease computational load while maintaining sufficient resolution for the analysis band (1–45 Hz).

### Theory
*   **Nyquist Criterion:** At 250 Hz, the Nyquist frequency is 125 Hz, which adequately covers the maximum frequency of interest (45 Hz).
*   **Aliasing Prevention:** naive subsampling causes high-frequency content to fold back into the lower spectrum. Anti-aliasing low-pass filtering must precede downsampling.

### Implementation
*   **Function:** `scipy.signal.decimate`
*   **Factor ($q$):** 2
*   **Filter Type:** IIR (Chebyshev Type I, order 8) or FIR.
*   **Phase Handling:** `zero_phase=True` ensures no temporal shift in event markers.

---

## Section 2: Linear Filtering: IIR Butterworth

### Objective
Isolate specific frequency bands using Infinite Impulse Response (IIR) filters.

### Theory
*   **Butterworth Filter:** Characterized by a maximally flat magnitude response in the passband (no ripple).
*   **SOS (Second-Order Sections):** Preferred over transfer function coefficients (`ba`) for numerical stability, particularly at higher orders.
*   **Zero-Phase Filtering:** Achieved via `sosfiltfilt`, which applies the filter forward and backward, doubling the effective order and eliminating phase delay.

### Filter Types Demonstrated
1.  **Low-Pass:** Cutoff at 50 Hz.
2.  **High-Pass:** Cutoff at 1 Hz (removes DC drift).
3.  **Band-Pass:** 1–45 Hz (standard EEG/MEG analysis range).
4.  **Band-Stop (Notch):** 48–52 Hz (removes line noise).

### Visualization
Frequency response plots compare `output='ba'` (using `freqz`) vs `output='sos'` (using `sosfreqz`) to demonstrate numerical precision.

---

## Section 3: Linear Filtering: FIR (Window Design)

### Objective
Design Finite Impulse Response (FIR) filters using the window method to control stop-band attenuation and transition width.

### Theory
*   **FIR Stability:** Always stable as they lack feedback loops.
*   **Window Trade-offs:**
    *   *Rectangular:* Sharpest transition, highest side-lobe leakage.
    *   *Hann/Hanning:* Balanced performance for general purposes.
    *   *Kaiser ($\beta$):* Tunable parameter $\beta$ allows trade-off between main-lobe width and side-lobe attenuation.
    *   *Blackman:* High attenuation, wider transition band.

### Implementation
*   **Function:** `scipy.signal.firwin`
*   **Parameters:** `numtaps=101`, Band-pass [1, 45] Hz.
*   **Analysis:** Compare frequency responses and time-domain outputs across different window types.

---

## Section 4: `filtfilt` vs `lfilter` + Gustafsson Edge Method

### Objective
Demonstrate the impact of phase delay and edge artifacts in filtering.

### Comparison
1.  **`lfilter` (Causal):**
    *   Real-time applicable.
    *   Introduces phase delay (signal features shift right in time).
    *   Unsuitable for precise connectivity or timing analysis.
2.  **`filtfilt` (Zero-Phase):**
    *   Non-causal (requires full signal).
    *   No phase delay.
    *   Effective filter order is $2 \times N$.
3.  **Gustafsson’s Method:**
    *   An advanced initialization technique for `filtfilt`.
    *   Minimizes start-up transients (edge artifacts) by matching initial conditions derived from the signal statistics, rather than simple zero-padding or reflection.

---

## Section 5: Non-Linear Filtering: Median & Savitzky-Golay

### Objective
Remove impulsive noise and smooth signals while preserving morphological features.

### Methods
1.  **Median Filter:**
    *   **Mechanism:** Replaces each point with the median of neighboring points.
    *   **Use Case:** Removal of "salt-and-pepper" spikes.
    *   **Property:** Non-linear; does not preserve superposition. Excellent edge preservation.
2.  **Savitzky-Golay Filter:**
    *   **Mechanism:** Fits a local polynomial (least squares) within a sliding window.
    *   **Use Case:** Smoothing without distorting peak height/width.
    *   **Derivatives:** Can compute instantaneous slope (1st derivative) or acceleration (2nd derivative) directly from the polynomial coefficients.

---

## Section 6: Convolution for Time Series

### Objective
Illustrate filtering as a convolution operation in the time domain.

### Theory
https://jinglescode.github.io/2020/11/01/how-convolutional-layers-work-deep-learning-neural-networks/

### Kernels Demonstrated
1.  **Box (Rectangular):** Equal weights. Simple moving average.
2.  **Hanning:** Smooth bell shape. Reduces spectral leakage compared to Box.
3.  **Gaussian:** Optimal joint time-frequency resolution. Defined by $\sigma$.
4.  **Derivative Kernel:** $[-0.5, 0, 0.5]$. Acts as a high-pass edge detector.

### Performance Note
`scipy.signal.fftconvolve` is used for large kernels, leveraging Fast Fourier Transform for $O(N \log N)$ complexity versus $O(N^2)$ for direct convolution.

---

## Section 7: Frequency-Swept Signals (Chirp)

### Objective
Validate filter performance using signals with time-varying frequency content.

### Signal Types
1.  **Linear Chirp:** Frequency increases linearly from $f_0$ to $f_1$.
2.  **Logarithmic Chirp:** Frequency increases exponentially (mimics human auditory perception).

### Application
*   Pass chirp through a Band-Pass Filter (13–30 Hz).
*   **Result:** Only the segment of the chirp occurring between 13–30 Hz remains.
*   **Visualization:** Spectrograms clearly show the "surviving" frequency track, confirming filter selectivity.

---

## Section 8: Kalman Filter (1-D Scalar)

### Objective
Apply optimal recursive estimation for signal smoothing.

### Model
A scalar state-space model assuming the signal is locally constant with Gaussian noise.
*   **State Transition:** $x_k = x_{k-1}$
*   **Observation:** $z_k = x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)$

### Parameters
*   **$Q$ (Process Noise):** Controls smoothness. Low $Q$ assumes the signal changes slowly (high smoothing).
*   **$R$ (Measurement Noise):** Controls trust in observations. High $R$ relies more on the model prediction.

### Outcome
The Kalman filter adapts its gain dynamically, offering a non-stationary smoothing effect that can track sudden changes better than fixed-coefficient linear filters if tuned correctly.

---

## Section 9: Z-Score Normalisation

### Objective
Standardize signal amplitude for cross-condition comparison.

### Formula
$$ z[n] = \frac{x[n] - \mu}{\sigma} $$

### Implications
*   Centers data at mean 0 with standard deviation 1.
*   Facilitates the identification of outliers ($|z| > 3$).
*   **Caution:** Z-scoring removes absolute power information. It is suitable for relative modulation analysis but not for absolute power comparisons.

---

## Section 10: Power Spectral Density: Welch’s Method

### Objective
Estimate the power distribution across frequencies.

### Method: Welch (1967)
1.  Divide signal into overlapping segments (50% overlap).
2.  Apply a window function (Hann) to each segment to reduce spectral leakage.
3.  Compute the Periodogram (FFT squared) for each segment.
4.  Average the periodograms to reduce variance.

### Parameters
*   **`nperseg`:** 2 seconds (500 samples at 250 Hz).
*   **Frequency Resolution:** $\Delta f = F_s / L = 0.5$ Hz.

### Neurophysiological Expectation
*   **Rest:** High power in Alpha (8–12 Hz) and Beta (13–30 Hz) bands.
*   **Move:** Suppression of Beta power (Event-Related Desynchronization).

---

## Section 11: Beta-Band Power Ratio

### Objective
Quantify Event-Related Desynchronization (ERD).

### Calculation
1.  **Band Power:** Integrate PSD over the frequency band using the trapezoidal rule.
    $$ P_{band} = \int_{f_{low}}^{f_{high}} PSD(f) \, df $$
2.  **ERD Percentage:**
    $$ ERD\% = \frac{P_{move} - P_{rest}}{P_{rest}} \times 100 $$

### Interpretation
*   **Negative ERD%:** Power suppression (Desynchronization) during movement.
*   **Positive ERD%:** Power increase (Synchronization/Rebound).

---

## Section 12: Spectrogram Computation

### Objective
Analyze time-frequency dynamics to observe *when* spectral changes occur.

### Method: Short-Time Fourier Transform (STFT)
*   **Trade-off:**
    *   Wide Window $\rightarrow$ High Frequency Resolution, Low Time Resolution.
    *   Narrow Window $\rightarrow$ High Time Resolution, Low Frequency Resolution.
*   **Settings:** 1-second window with 75% overlap.

### Visualization Enhancements
1.  **Smoothing:** 2D Gaussian filtering applied to the spectrogram matrix to reduce visual noise.
2.  **Log Scale:** Conversion to Decibels (dB) for better dynamic range visualization.
3.  **Differential Spectrogram:**
    $$ Sxx_{diff} = Sxx_{move} - Sxx_{rest} $$
    *   **Blue:** Significant power decrease (ERD).
    *   **Red:** Significant power increase (ERS).

This differential map explicitly highlights the Beta-band suppression (13–30 Hz) coincident with the movement onset.
