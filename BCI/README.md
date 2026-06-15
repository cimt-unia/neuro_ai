# BCI Introduction
<img width="360" height="184" alt="image" src="https://github.com/user-attachments/assets/1f6273a7-d743-45c6-988f-92de3173c17b" />

This repository outlines the current state-of-the-art Python libraries for Brain-Computer Interface (BCI) research, with a specific focus on real-time experimental control and high-density EEG integration.

## Core Libraries and Frameworks

### 1. Lab Streaming Layer (LSL)
*   **[Lab Streaming Layer](https://labstreaminglayer.org/)**: LSL serves as the foundational networking system for time-synchronized multimodal data collection. It ensures precise temporal alignment between EEG acquisition hardware, visual stimuli presentation, and experimental task events [3]. 
    *   *Relevance:* Essential for any online BCI experiment requiring millisecond-level synchronization between brain signals and external applications.

### 2. MNE-LSL (Recommended for Continuous Control)
*   **[MNE-LSL](https://mne.tools/mne-lsl/)** (formerly MNE-Realtime): Integrated within the MNE-Python ecosystem, this framework provides robust, low-latency streaming capabilities via LSL. 
    *   *Application:* It is the preferred tool for constructing custom real-time pipelines for Motor Imagery (MI) and Neurofeedback. Its modular architecture allows researchers to integrate advanced machine learning classifiers (e.g., scikit-learn, PyTorch) for the continuous control of applications or games [2].
    *   *Status:* Actively maintained with support for modern Python versions and recent dependency updates [1].

### 3. BciPy (Recommended for ERP-Based Communication)
*   **[BciPy](https://github.com/CAMBI-tech/BciPy)**: Maintained by the CAMBI consortium, BciPy is a comprehensive library designed for research-grade BCI experiments. Following its major v2.0.0 release in August 2025, it remains the standard for Event-Related Potential (ERP) paradigms, such as P300 RSVP and Matrix Spellers [1].
    *   *Limitation:* Due to its rigid "Inquiry/Series/Trial" architectural structure, BciPy is optimized for discrete communication tasks and is generally unsuitable for continuous control applications like gaming or smooth cursor navigation [1].

### 4. Brainflow
*   **[Brainflow](https://brainflow.org/)**: A unified, high-performance library for biosignal data acquisition that connects directly to EEG hardware via Bluetooth without requiring middleware. It is the standard for real-time BCI applications in 2026 due to its minimal latency and streamlined architecture [1].
    *   *Application:* Ideal for closed-loop systems requiring low-latency feedback (e.g., neurofeedback, Motor Imagery control). It supports direct connection to Muse 2 (`BoardIds.MUSE_2_BOARD`) and other modern devices through a single, consistent API [1].
    *   *Key Features:* Achieves typical latencies of **< 20–50 ms** via direct BLE connection and eliminates setup complexity by removing the need for external bridge applications [1].



## Hardware

| Control Type | Best Band | Frequency | Brain Region | Suitable Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **Directional/Discrete** | **Mu / Beta** | 8–30 Hz | Motor Cortex (C3, C4, Cz) | OpenBCI, Emotiv EPOC X |
| **Continuous/Scalar** | **Beta / Alpha** | 8–25 Hz | Prefrontal (FP1, FP2, AF7, AF8) | Muse 2, NeuroSky, Emotiv Insight |
---

### Research References
- [Opposing cortical forces: Alpha slowing and sensorimotor mu acceleration during motor-related BCI training](https://doi.org/10.1371/journal.pcbi.1014112) 
- [Developing a Brain-Computer Interface Game with Left-Right Motor Imagery](https://www.mdpi.com/2078-2489/14/7/354)
- [Using Muse: Rapid Mobile Assessment of Brain Performance](https://doi.org/10.3389/fnins.2021.634147)  
### Multimedia Resources
- [Inside a Brain-Chip Startup](https://www.youtube.com/watch?v=okxvk08uAwo)

---

### Sources
- [1] [BciPy: Brain-Computer Interface Software in Python](https://github.com/CAMBI-tech/BciPy) - [Details BciPy's focus on ERP spellers and its v2.0.0 release status]
- [2] [MNE-LSL - MNE-Python](https://mne.tools/mne-lsl/) - [Official documentation for the real-time LSL streaming framework used for custom pipelines]
- [3] [Lab Streaming Layer Repository](https://github.com/sccn/labstreaminglayer) - [Explains LSL's function as the synchronization backbone for BCI experiments]
