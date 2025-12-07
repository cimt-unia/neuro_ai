# roi_connectivity.py


import numpy as np
import mne
import json
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs
from nilearn import datasets

# ───────────────────────────────────────
# 1. Utility: ROI definitions
# ───────────────────────────────────────

def get_difumo_names():
    try:
        atlas = datasets.fetch_atlas_difumo(dimension=512, resolution_mm=2)
        return atlas.labels['difumo_names'].astype(str).tolist()
    except Exception:
        return [f"Component_{i}" for i in range(512)]

def define_motor_cognitive_regions():
    Motor_M1 = [40, 86, 198, 268, 305, 437, 458, 465]
    Motor_SMA_Premotor = [17, 18, 288, 291, 296, 297, 302, 305, 314, 315, 335, 375, 379, 448]
    Motor_Medial = [101, 102, 388, 409, 498]
    Thalamus = [70, 73, 297, 334, 414, 420] 
    Basal_Ganglia = [30, 53, 224, 260, 405, 422, 109, 110, 315, 331, 467, 479, 55, 71, 307, 223]  
    Cerebellum_Motor = [43, 47, 83, 84, 127, 183, 220, 221, 295, 304, 310, 311, 374, 378, 381, 403, 441, 490, 491]
    Somatosensory = [44, 131, 210, 411, 413, 436]
    Executive_Control = [3, 85, 104, 148, 184, 337, 377, 446, 447, 506, 507]
    Interoception = [2, 387, 358, 389, 165, 469]
    Error_Monitoring = [185, 219, 326, 473, 492]
    return sorted(set(
        Motor_M1 + Motor_SMA_Premotor + Motor_Medial + Thalamus +
        Basal_Ganglia + Cerebellum_Motor + Somatosensory +
        Executive_Control + Interoception + Error_Monitoring
    ))

def get_band_freqs(band_name):
    bands = {
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Low_Beta": (13, 20),
        "High_Beta": (20, 30),
        "Low_Gamma": (30, 60),
        "High_Gamma": (60, 120)
    }
    if band_name not in bands:
        raise ValueError(f"Unknown band: {band_name}. Options: {list(bands.keys())}")
    return bands[band_name]

# ───────────────────────────────────────
# 2. Epoch creation functions
# ───────────────────────────────────────

def create_task_epochs(
    data_file,
    events_file,
    event_id_file,
    condition,
    tmin=0.0,
    tmax=1.5,
    sfreq=500.0
):
    """Create epochs from event markers."""
    data = np.load(data_file)
    if data.shape[0] > data.shape[1]:
        data = data.T

    events = mne.read_events(events_file)
    with open(event_id_file, 'r') as f:
        event_id = json.load(f)

    ch_names = [f'C{i}' for i in range(data.shape[0])]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types='misc')
    raw = mne.io.RawArray(data, info, verbose=False)

    epochs = mne.Epochs(
        raw, events, {condition: event_id[condition]},
        tmin=tmin, tmax=tmax, baseline=None,
        preload=True, verbose=False, event_repeated='drop'
    )
    return epochs

def create_rest_epochs(
    data_file,
    duration=2.5,
    sfreq=500.0
):
    """Create fixed-length epochs from continuous data."""
    data = np.load(data_file)
    if data.shape[0] > data.shape[1]:
        data = data.T

    ch_names = [f'C{i}' for i in range(data.shape[0])]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types='misc')
    raw = mne.io.RawArray(data, info, verbose=False)

    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=duration,
        baseline=None, preload=True, verbose=False
    )
    return epochs

# ───────────────────────────────────────
# 3. Connectivity function
# ───────────────────────────────────────

def compute_roi_connectivity_matrix(
    epochs,
    band_name="Low_Beta",
    method='wpli2_debiased',
    sfreq=500.0
):
    """
    Compute a single ROI × ROI connectivity matrix from MNE Epochs object.
    
    Parameters:
    - epochs: mne.Epochs instance (already loaded and preprocessed)
    - band_name: e.g., "Alpha", "Low_Beta"
    - method: connectivity method (default: 'wpli2_debiased')
    
    Returns:
    - conn_df: pandas DataFrame (n_roi × n_roi) with DiFuMo ROI names as labels
    """
    # Get ROI info
    all_names = get_difumo_names()
    selected_indices = define_motor_cognitive_regions()
    roi_names = [all_names[i] for i in selected_indices]

    # Extract data for selected ROIs
    epoch_data = epochs.get_data()[:, selected_indices, :]  # (n_epochs, n_roi, n_times)

    # Get frequency range
    fmin, fmax = get_band_freqs(band_name)

    # Compute connectivity
    con = spectral_connectivity_epochs(
        data=epoch_data,
        method=method,
        mode='multitaper',
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False
    )
    matrix = con.get_data(output='dense').squeeze()
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    return pd.DataFrame(matrix, index=roi_names, columns=roi_names)


'''
# Task
epochs = create_task_epochs(
    data_file=r"sub-02_task\difumo_time_courses.npy",
    events_file=r"sub-02_task\sub-02_events_mne_binary-eve.fif",
    event_id_file=r"sub-02_task\sub-02_event_id_binary.json",
    condition="InPhase"
)

conn_matrix = compute_roi_connectivity_matrix(epochs, band_name="Alpha")
conn_matrix.to_csv("sub-02_InPhase_Alpha_matrix.csv")


# Rest
epochs = create_rest_epochs(
    data_file=r"sub-02_rest\difumo_time_courses.npy",
    duration=2.5
)

conn_matrix = compute_roi_connectivity_matrix(epochs, band_name="Low_Beta")

'''