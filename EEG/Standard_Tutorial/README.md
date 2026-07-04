### **Summary: Signal Processing in Neuroelectrophysiology**

#### **1. Fundamentals of EEG Acquisition and Physics**
*   **Physiological Origin of Signals:** EEG does not measure the "output" spikes (action potentials) of individual neurons. Instead, it captures the summed postsynaptic potentials (inputs) from large populations of pyramidal neurons. For these signals to be detectable on the scalp, the neurons must be aligned in parallel and active synchronously. Glial activity and blood flow changes are not the primary sources of EEG voltage.
*   **The 10–20 Electrode System:** This standard placement system uses a specific naming convention to indicate location:
    *   **Letters** denote the brain lobe (e.g., F for Frontal, T for Temporal).
    *   **Numbers** indicate lateralization: Odd numbers represent the left hemisphere, even numbers represent the right hemisphere.
    *   **"z"** denotes electrodes placed on the midline (zero distance from the center).
*   **Volume Conduction and Filtering:** The skull acts as a resistive barrier to electrical currents. Due to its thickness and low conductivity, it functions as a **low-pass filter**, significantly attenuating high-frequency signals while allowing slower oscillations to pass through to the scalp sensors with less distortion.
*   **Referencing Strategies:**
    *   **Bipolar Reference:** Calculates the voltage difference between two adjacent electrodes. This method is effective for highlighting localized activity and reducing common-mode noise but may obscure widespread patterns.
    *   **Common Average & Linked Mastoids:** Other common methods that reference all channels to an average or specific ear electrodes, respectively.

#### **2. Linear Filtering and Time Series Analysis**
*   **Fourier Transform Properties:** The Fourier Transform is a reversible mathematical operation. It converts a time-domain signal into the frequency domain without losing information. Consequently, the original time-domain signal can be perfectly reconstructed from its frequency components, ensuring no data destruction occurs during transformation.
*   **Filter Architectures: FIR vs. IIR:**
    *   **Finite Impulse Response (FIR):** These filters use only current and past input values, with no feedback loop. This structure guarantees inherent stability and a linear phase response, meaning all frequencies are delayed by the same amount, preserving the shape of the waveform.
    *   **Infinite Impulse Response (IIR):** These filters utilize feedback from previous outputs. While computationally efficient, they introduce **non-linear phase shifts**. This means different frequencies are delayed by different amounts, which can distort the temporal shape of complex signals even if the amplitude spectrum is preserved.
*   **1D Convolution Mechanics:**
    *   **Kernel Operation:** A kernel (a small array of weights) slides across the input signal. At each position, it performs an element-wise multiplication and sums the results to produce a single output value.
    *   **Dilation:** Dilation introduces gaps between the kernel weights. This allows the filter to cover a wider receptive field (capture broader context) without increasing the number of parameters or computational cost.
    *   **Padding:** Zero-padding is added to the edges of the input signal before convolution. This ensures that the output signal maintains the same spatial dimensions as the input, preventing data loss at the boundaries.

#### **3. Feature Engineering and Spectral Analysis**
*   **Spectral Leakage:** In the Discrete Fourier Transform (DFT), spectral leakage occurs when the analysis window does not contain an integer number of signal cycles. This abrupt cutoff creates a discontinuity, causing signal energy to "smear" or leak into adjacent frequency bins, distorting the true spectral content. Windowing functions (like Hann) are often used to mitigate this.
*   **The Heisenberg-Gabor Limit (Time-Frequency Trade-off):** There is an inverse relationship between time and frequency resolution:
    *   **Long Windows:** Capture more cycles, providing precise **frequency resolution** but poor **time resolution** (blurring *when* events occur).
    *   **Short Windows:** Provide precise **time resolution** (pinpointing *when* events occur) but poor **frequency resolution** (blurring the exact frequency).
*   **Feature Selection for Machine Learning:**
    *   **Band Power:** Calculating power in specific bands (Delta, Theta, Alpha, Beta, Gamma) is highly effective because it reduces high-dimensional raw data into a few biologically meaningful features. It is robust and interpretable.
    *   **Raw Voltage Challenges:** Using raw time-domain samples is problematic due to high dimensionality and extreme sensitivity to phase shifts. A millisecond shift in a wave can completely change the raw voltage values, confusing machine learning models.
    *   **Spectrograms:** Unlike static band power, spectrograms provide a time-frequency representation. They are essential for analyzing non-stationary signals where frequency content changes rapidly over time, allowing models to detect transient events.
    *   **Data Leakage:** Using heavily overlapping windows for training and testing can cause data leakage. The model may learn to recognize shared noise or artifacts between overlapping segments rather than generalizable physiological patterns, leading to inflated performance metrics.

#### **4. Independent Component Analysis (ICA)**
*   **Physical Interpretation of the Mixing Matrix:** In the ICA model $X = AS$, the mixing matrix $A$ represents **volume conduction**. It maps how electrical activity from a single independent source (brain or artifact) spreads instantaneously through the head’s tissues to reach all scalp electrodes.
*   **Preprocessing Requirements:**
    *   **Mean Centering:** Data must be centered (mean subtracted) before ICA. Statistical measures like variance and kurtosis assume a zero-mean distribution. A non-zero mean (DC offset) biases these calculations and prevents proper separation.
    *   **Rank Reduction:** Applying an average reference forces the sum of all channels to zero, creating a linear dependency. For $N$ channels, this reduces the rank to $N-1$. Consequently, ICA can extract at most $N-1$ independent components.
*   **Statistical Foundations:**
    *   **PCA vs. ICA:**
        *   **PCA (Principal Component Analysis):** Finds orthogonal components that maximize **variance**. It is a second-order statistic method.
        *   **ICA (Independent Component Analysis):** Finds statistically **independent** components by maximizing non-Gaussianity. It uses higher-order statistics and does not require orthogonality.
    *   **The Gaussian Problem:** ICA cannot separate sources if they are perfectly Gaussian. The sum of independent Gaussian variables is also Gaussian. Since the mixed signal looks statistically identical to the sources, ICA has no mathematical basis to untangle them. ICA relies on the Central Limit Theorem in reverse: mixing non-Gaussian signals makes them *more* Gaussian, so the algorithm seeks the least Gaussian (most independent) projections.
*   **Kurtosis and Distribution Types:** Kurtosis measures the "tailedness" of a probability distribution. Understanding the three types is critical for identifying artifacts:
    1.  **Mesokurtic (Gaussian):** A normal distribution with a moderate peak and tails. Kurtosis $\approx 3$ (or excess kurtosis $\approx 0$). Most random background noise tends toward this shape.
    2.  **Leptokurtic (Super-Gaussian):** Characterized by a **sharp peak** and **heavy tails**. Kurtosis $> 3$. This distribution indicates that the signal is mostly quiet (near zero) but occasionally has large, rare spikes. This is typical of **artifacts** like eye blinks, heartbeats, or muscle bursts.
    3.  **Platykurtic (Sub-Gaussian):** Characterized by a **flat peak** and **thin tails**. Kurtosis $< 3$. This indicates a more uniform distribution of values, often seen in certain types of continuous brain rhythms or uniform noise.
*   **Algorithmic Advances:** The **Picard** algorithm is a modern ICA solver that combines the speed of FastICA with the robustness of Infomax. It converges faster and more accurately on complex EEG data, making it a preferred choice over older methods.

