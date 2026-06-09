# BCI Introduction


## Libraries

*   **[BciPy](https://github.com/bcipy/bcipy)**: An open-source, modular library designed for conducting BCI experiments, particularly focusing on **ERP spelling interfaces** (like RSVP and Matrix Speller) and communication restoration. It supports data acquisition, signal processing, and GUI-based task building on Windows, Linux, and macOS [1].
*   **[PyBCI](https://github.com/alexandrebarachant/pybci)**: A package for creating real-time BCIs that handles data synchronization and pipelining via the **Lab Streaming Layer (LSL)**. It integrates with machine learning libraries like **PyTorch**, **scikit-learn**, and **TensorFlow**, leveraging **AntroPy**, **SciPy**, and **NumPy** for feature extraction [2].
*   **[Wyrm](https://github.com/bbci/wyrm)**: A toolbox suitable for both online BCI experiments and offline EEG data analysis, offering examples for motor imagery classification and P300 Matrix Speller tasks using BCI Competition datasets [3].



## Ideas
### Extended Kalman Filters:
[Opposing cortical forces: Alpha slowing and sensorimotor mu acceleration during motor-related BCI training](https://doi.org/10.1371/journal.pcbi.1014112) - [Study demonstrating the use of EKF for tracking dynamic brain rhythms] 
For research into non-stationary brain rhythms (e.g., *Opposing Cortical Forces*), standard static bandpower features may be insufficient. Researchers can implement **Extended Kalman Filters (EKF)** using **[FilterPy](https://filterpy.readthedocs.io/)** to track instantaneous frequency and magnitude shifts during training sessions [6].

---

### Sources
[1] [BciPy GitHub Repository](https://github.com/bcipy/bcipy) - [Open-source library for ERP-based BCI experiments and data acquisition]
[2] [PyBCI Documentation](https://github.com/alexandrebarachant/pybci) - [Real-time BCI framework integrating LSL and machine learning]
[3] [Wyrm Toolbox](https://github.com/bbci/wyrm) - [Toolbox for online/offline BCI experiments and competition datasets]
[4] [MNE-Python Official Site](https://mne.tools/stable/index.html) - [Standard library for EEG/MEG preprocessing and analysis]
[5] [SciPy Signal Processing Module](https://docs.scipy.org/doc/scipy/reference/signal.html) - [Tools for filtering, spectral estimation, and windowing]
