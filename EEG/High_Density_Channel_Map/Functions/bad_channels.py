import time
from typing import List, Tuple, Optional
import numpy as np
from scipy.stats import median_abs_deviation

def detect_bad_channels(
    raw: 'mne.io.Raw',
    mad_threshold: float = 10.0,
    min_amplitude_uv: float = 0.1
) -> List[str]:
    """Detect flat and noisy EEG channels."""
    import mne
    raw_eeg = raw.copy().pick("eeg")
    data_uv = raw_eeg.get_data() * 1e6
    amplitude = np.ptp(data_uv, axis=1)
    variance = np.var(data_uv, axis=1)

    flat_mask = amplitude < min_amplitude_uv
    flat_chs = [ch for ch, is_flat in zip(raw_eeg.ch_names, flat_mask) if is_flat]

    noisy_mask = np.zeros(len(amplitude), dtype=bool)
    for feat in (variance, amplitude):
        mad = median_abs_deviation(feat, scale="normal", nan_policy="omit")
        if not np.isnan(mad) and mad > 1e-12:
            z = (feat - np.nanmedian(feat)) / mad
            noisy_mask |= z > mad_threshold
    noisy_chs = [ch for ch, is_noisy in zip(raw_eeg.ch_names, noisy_mask) if is_noisy]

    return sorted(set(flat_chs + noisy_chs))