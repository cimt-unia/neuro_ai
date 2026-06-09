"""
EKF-based Oscillator Tracking for Alpha/Mu Rhythms
Implementation based on:
Kostoglou & Müller-Putz (2026) - "Opposing cortical forces: Alpha slowing 
and sensorimotor mu acceleration during motor-related BCI training"
PLOS Computational Biology

The EEG signal is modeled as a damped harmonic oscillator:
    y[t] = phi[t] * y[t-1] - psi[t] * y[t-2] + eps[t]

where:
    phi[t] = 2 * exp(-gamma[t] * dt) * cos(2 * pi * f[t] * dt)
    psi[t] = exp(-2 * gamma[t] * dt)

The latent state x = [f, gamma] evolves as a random walk and is tracked
via an Extended Kalman Filter.
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.optimize import differential_evolution
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from typing import Tuple, Optional


# ============================================================================
# EKF Oscillator Tracker - Core Algorithm
# ============================================================================

class EKFOscillatorTracker:
    """
    Extended Kalman Filter for tracking instantaneous frequency (f) and 
    damping factor (gamma) of a narrowband oscillatory signal modeled as 
    a time-varying AR(2) process.
    
    State vector:
        x = [f, gamma]^T
    
    State transition (random walk):
        x[t] = x[t-1] + w[t],  w ~ N(0, Q)
    
    Measurement (AR(2) oscillator output):
        y[t] = phi(f,gamma)*y[t-1] - psi(gamma)*y[t-2] + v[t],  v ~ N(0, R)
    """
    
    def __init__(self, dt: float, f0: float, gamma0: float, 
                 Q: np.ndarray, R: float, P0_scale: float = 1.0):
        """
        Parameters
        ----------
        dt : float
            Sampling interval in seconds (1 / fs)
        f0 : float
            Initial frequency estimate (Hz)
        gamma0 : float
            Initial damping factor estimate
        Q : np.ndarray (2,2)
            Process noise covariance matrix
        R : float
            Measurement noise variance
        P0_scale : float
            Scaling factor for initial state covariance
        """
        self.dt = dt
        self.Q = Q
        self.R = R
        
        # State vector: [frequency, damping]
        self.x = np.array([f0, gamma0], dtype=np.float64)
        
        # State covariance matrix
        self.P = P0_scale * np.eye(2, dtype=np.float64)
        
        # Buffers for AR(2) formulation (need two past samples)
        self.y_prev = 0.0
        self.y_prev2 = 0.0
        
        # Storage for estimated trajectories
        self.freq_history = []
        self.gamma_history = []
        self.P_history = []
        
        # Reconstructed oscillator signal for magnitude computation
        self.reconstructed = []
        
        # Small constant for numerical stability
        self.eps = 1e-8
    
    def _compute_ar_coefficients(self, f: float, gamma: float) -> Tuple[float, float]:
        """Compute phi and psi from instantaneous frequency and damping."""
        exp_term = np.exp(-gamma * self.dt)
        phi = 2.0 * exp_term * np.cos(2.0 * np.pi * f * self.dt)
        psi = exp_term ** 2
        return phi, psi
    
    def _measurement_function(self, x: np.ndarray) -> float:
        """Predicted observation given state x = [f, gamma]."""
        f, gamma = x
        phi, psi = self._compute_ar_coefficients(f, gamma)
        return phi * self.y_prev - psi * self.y_prev2
    
    def _jacobian_h(self, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian of measurement function with respect to state."""
        f, gamma = x
        exp_gdt = np.exp(-gamma * self.dt)
        cos_term = np.cos(2.0 * np.pi * f * self.dt)
        sin_term = np.sin(2.0 * np.pi * f * self.dt)
        
        # dh/df
        dh_df = -2.0 * exp_gdt * sin_term * 2.0 * np.pi * self.dt * self.y_prev
        
        # dh/dgamma
        dphi_dgamma = -2.0 * self.dt * exp_gdt * cos_term
        dpsi_dgamma = -2.0 * self.dt * exp_gdt**2
        dh_dgamma = dphi_dgamma * self.y_prev - dpsi_dgamma * self.y_prev2
        
        return np.array([dh_df, dh_dgamma], dtype=np.float64)
    
    def step(self, y_new: float) -> Tuple[float, float]:
        """
        Perform one EKF update step.
        
        Parameters
        ----------
        y_new : float
            New EEG sample
            
        Returns
        -------
        f_est : float
            Estimated instantaneous frequency (Hz)
        gamma_est : float
            Estimated instantaneous damping factor
        """
        # --- Predict Step ---
        x_pred = self.x.copy()
        P_pred = self.P + self.Q
        
        # --- Update Step ---
        y_pred = self._measurement_function(x_pred)
        innovation = y_new - y_pred
        
        # Jacobian at predicted state
        H = self._jacobian_h(x_pred).reshape(1, 2)  # Shape (1, 2)
        
        # Innovation covariance
        S = H @ P_pred @ H.T + self.R  # Shape (1, 1)
        
        # Kalman gain: P_pred (2,2) @ H.T (2,1) = (2,1)
        K = P_pred @ H.T / (S[0, 0] + self.eps)  # Shape (2, 1)
        
        # Update state estimate
        self.x = x_pred + (K * innovation).flatten()
        
        # Joseph form covariance update (numerically stable)
        # I_KH must be (2,2), K is (2,1), H is (1,2), so K @ H is (2,2)
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ K.T * self.R
        
        # Enforce constraints
        self.x[0] = max(self.x[0], 0.5)   # Minimum frequency: 0.5 Hz
        self.x[1] = max(self.x[1], 0.01)  # Minimum damping
        
        # Reconstruct the oscillator output for magnitude tracking
        f_curr, gamma_curr = self.x
        phi_curr, psi_curr = self._compute_ar_coefficients(f_curr, gamma_curr)
        y_reconstructed = phi_curr * self.y_prev - psi_curr * self.y_prev2
        
        # Shift buffers
        self.y_prev2 = self.y_prev
        self.y_prev = y_new
        
        # Store history
        self.freq_history.append(self.x[0])
        self.gamma_history.append(self.x[1])
        self.P_history.append(self.P.copy())
        self.reconstructed.append(y_reconstructed)
        
        return self.x[0], self.x[1]
    
    def track_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track frequency, damping, and magnitude over an entire signal.
        
        Magnitude is computed from the Hilbert envelope of the EKF-reconstructed
        oscillator output, not the raw filtered signal. This better matches the
        paper's methodology by deriving amplitude from the model state.
        
        Parameters
        ----------
        signal : np.ndarray (n_samples,)
            Bandpass-filtered EEG signal
            
        Returns
        -------
        freq_trajectory : np.ndarray (n_samples,)
            Instantaneous frequency estimates (Hz)
        gamma_trajectory : np.ndarray (n_samples,)
            Instantaneous damping estimates
        magnitude_trajectory : np.ndarray (n_samples,)
            Envelope magnitude from reconstructed oscillator
        """
        n = len(signal)
        freq_est = np.zeros(n)
        gamma_est = np.zeros(n)
        
        for i in range(n):
            f, g = self.step(signal[i])
            freq_est[i] = f
            gamma_est[i] = g
        
        # Magnitude from Hilbert envelope of the EKF-reconstructed signal
        reconstructed = np.array(self.reconstructed)
        analytic_signal = hilbert(reconstructed)
        magnitude = np.abs(analytic_signal)
        
        return freq_est, gamma_est, magnitude


# ============================================================================
# Hyperparameter Optimization via Genetic Algorithm
# ============================================================================

def ekf_prediction_error(params: np.ndarray, signal: np.ndarray, dt: float) -> float:
    """
    Cost function for EKF hyperparameter optimization.
    Minimizes the mean squared prediction error.
    """
    Q_f, Q_gamma, Q_cross, R, f0, gamma0, P0_scale = params
    
    Q_f = abs(Q_f) + 1e-10
    Q_gamma = abs(Q_gamma) + 1e-10
    R = abs(R) + 1e-10
    
    Q = np.array([[Q_f, Q_cross], [Q_cross, Q_gamma]])
    
    try:
        np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        return 1e10
    
    ekf = EKFOscillatorTracker(
        dt=dt, f0=f0, gamma0=gamma0, Q=Q, R=R, P0_scale=abs(P0_scale)
    )
    
    errors = []
    for y in signal:
        f_pred, _ = ekf.step(y)
        y_pred = ekf._measurement_function(ekf.x)
        errors.append((y - y_pred) ** 2)
    
    return np.mean(errors)


def optimize_ekf_parameters(signal: np.ndarray, fs: float,
                            param_bounds: Optional[dict] = None) -> dict:
    """Optimize EKF hyperparameters using differential evolution."""
    if param_bounds is None:
        param_bounds = {
            'Q_f': (1e-6, 10.0),
            'Q_gamma': (1e-6, 10.0),
            'Q_cross': (-5.0, 5.0),
            'R': (1e-4, 100.0),
            'f0': (7.0, 13.0),
            'gamma0': (0.5, 20.0),
            'P0_scale': (0.01, 100.0)
        }
    
    bounds = [param_bounds[k] for k in ['Q_f', 'Q_gamma', 'Q_cross', 'R', 
                                          'f0', 'gamma0', 'P0_scale']]
    dt = 1.0 / fs
    
    result = differential_evolution(
        ekf_prediction_error,
        bounds,
        args=(signal, dt),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-8,
        seed=42,
        disp=False
    )
    
    Q_f, Q_gamma, Q_cross, R, f0, gamma0, P0_scale = result.x
    
    return {
        'Q': np.array([[abs(Q_f), Q_cross], [Q_cross, abs(Q_gamma)]]),
        'R': abs(R),
        'f0': f0,
        'gamma0': abs(gamma0),
        'P0_scale': abs(P0_scale)
    }


# ============================================================================
# Signal Processing Utilities
# ============================================================================

def bandpass_filter(signal: np.ndarray, fs: float, lowcut: float, 
                    highcut: float, order: int = 5) -> np.ndarray:
    """Butterworth bandpass filter (zero-phase)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ============================================================================
# Full Analysis Pipeline
# ============================================================================

def track_alpha_mu_band(eeg_data: np.ndarray, fs: float, 
                         band: Tuple[float, float],
                         optimization_signal_length: float = 60.0,
                         filter_order: int = 5,
                         optimize_per_channel: bool = False) -> dict:
    """
    Full pipeline for tracking alpha/mu oscillations across channels.
    
    Parameters
    ----------
    eeg_data : np.ndarray (n_channels, n_samples)
        Multi-channel EEG data
    fs : float
        Sampling frequency
    band : tuple
        (lowcut, highcut) for bandpass filter
    optimization_signal_length : float
        Length in seconds for hyperparameter optimization
    filter_order : int
        Butterworth filter order
    optimize_per_channel : bool
        If True, optimize EKF per channel. If False, use Cz or first channel.
        
    Returns
    -------
    results : dict
        Frequency, magnitude, damping, and covariance norm trajectories
    """
    n_channels, n_samples = eeg_data.shape
    opt_samples = int(optimization_signal_length * fs)
    
    results = {
        'frequency': np.zeros((n_channels, n_samples)),
        'magnitude': np.zeros((n_channels, n_samples)),
        'damping': np.zeros((n_channels, n_samples)),
        'state_covariance_norm': np.zeros((n_channels, n_samples))
    }
    
    # Optimize parameters (once, unless per_channel is True)
    if not optimize_per_channel:
        ref_channel = min(0, n_channels - 1)  # Use first channel as reference
        ref_filtered = bandpass_filter(eeg_data[ref_channel], fs, 
                                       band[0], band[1], filter_order)
        opt_signal = ref_filtered[:opt_samples]
        opt_params = optimize_ekf_parameters(opt_signal, fs)
    
    for ch in range(n_channels):
        # Bandpass filter
        filtered = bandpass_filter(eeg_data[ch], fs, band[0], band[1], filter_order)
        
        # Optimize per channel if requested
        if optimize_per_channel:
            opt_signal = filtered[:opt_samples]
            opt_params = optimize_ekf_parameters(opt_signal, fs)
        
        # Initialize EKF
        ekf = EKFOscillatorTracker(
            dt=1.0/fs,
            f0=opt_params['f0'],
            gamma0=opt_params['gamma0'],
            Q=opt_params['Q'],
            R=opt_params['R'],
            P0_scale=opt_params['P0_scale']
        )
        
        # Track full signal
        freq_est, gamma_est, mag_est = ekf.track_signal(filtered)
        results['frequency'][ch] = freq_est
        results['damping'][ch] = gamma_est
        results['magnitude'][ch] = mag_est
        
        # State covariance norm from stored history
        P_history = np.array([np.linalg.norm(P, 'fro') for P in ekf.P_history])
        results['state_covariance_norm'][ch] = P_history
    
    return results


def compute_session_long_slopes(trajectories: np.ndarray, 
                                 fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute robust linear regression slopes for session-long trajectories.
    
    The paper z-scores trajectories before regression to standardize across
    channels, then converts slopes back to original units.
    
    Parameters
    ----------
    trajectories : np.ndarray (n_channels, n_samples)
        EKF-tracked trajectories (frequency or magnitude)
    fs : float
        Sampling frequency
        
    Returns
    -------
    slopes : np.ndarray (n_channels,)
        Slope in original units per hour
    p_values : np.ndarray (n_channels,)
        P-values from regression
    """
    n_channels, n_samples = trajectories.shape
    time_hours = np.arange(n_samples) / (fs * 3600.0)
    X = sm.add_constant(time_hours)  # Design matrix reused for all channels
    
    slopes = np.zeros(n_channels)
    p_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        y = trajectories[ch]
        y_z = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        rlm_model = sm.RLM(y_z, X, M=sm.robust.norms.TukeyBiweight())
        rlm_results = rlm_model.fit()
        
        # Slope in z-scored units per hour, convert back to original
        slope_z = rlm_results.params[1]
        slope_original = slope_z * np.std(y)
        slopes[ch] = slope_original
        p_values[ch] = rlm_results.pvalues[1]
    
    return slopes, p_values


def select_optimal_frequency_band(eeg_data: np.ndarray, fs: float,
                                   task_labels: np.ndarray,
                                   candidate_bands: list) -> Tuple[float, float]:
    """Select optimal alpha/mu band based on task correlation."""
    n_channels = eeg_data.shape[0]
    band_correlations = {}
    
    for band in candidate_bands:
        results = track_alpha_mu_band(eeg_data, fs, band)
        
        ch_corrs = []
        for ch in range(n_channels):
            corr = np.abs(np.corrcoef(results['magnitude'][ch], task_labels)[0, 1])
            ch_corrs.append(corr)
        
        band_correlations[band] = np.mean(ch_corrs)
    
    return max(band_correlations, key=band_correlations.get)


def slope_change_likelihood(trajectories: np.ndarray, fs: float,
                             gaussian_sigma_minutes: float = 3.0) -> np.ndarray:
    """Compute slope change likelihood curve."""
    n_channels, n_samples = trajectories.shape
    sigma_samples = gaussian_sigma_minutes * 60 * fs
    
    change_accumulator = np.zeros(n_samples)
    
    for ch in range(n_channels):
        y = trajectories[ch]
        d2 = np.diff(np.diff(y))
        d2_padded = np.concatenate([[0, 0], np.abs(d2)])
        
        threshold = 2.0 * np.std(d2_padded)
        change_points = d2_padded > threshold
        change_accumulator += change_points.astype(float)
    
    likelihood = gaussian_filter1d(change_accumulator, sigma_samples)
    likelihood /= (likelihood.max() + 1e-10)
    
    return likelihood


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate synthetic EEG-like signal with known frequency shift
    fs = 120.0
    dt = 1.0 / fs
    duration = 300.0
    t = np.arange(0, duration, dt)
    n_samples = len(t)
    
    f_t = 9.5 + 1.0 * t / duration
    phase = 2.0 * np.pi * np.cumsum(f_t) * dt
    
    signal = np.cos(phase) * np.exp(-0.5 * t / 60.0)
    signal += 0.3 * np.random.randn(n_samples)
    
    filtered = bandpass_filter(signal, fs, 8.0, 12.0, order=5)
    
    opt_params = optimize_ekf_parameters(filtered[:int(60*fs)], fs)
    print("Optimized parameters:", opt_params)
    
    ekf = EKFOscillatorTracker(
        dt=dt, f0=opt_params['f0'], gamma0=opt_params['gamma0'],
        Q=opt_params['Q'], R=opt_params['R'], P0_scale=opt_params['P0_scale']
    )
    
    freq_traj, gamma_traj, mag_traj = ekf.track_signal(filtered)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t, freq_traj, 'b', label='EKF Estimate')
    axes[0].plot(t, f_t, 'r--', label='Ground Truth')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].legend()
    axes[0].set_title('EKF Oscillator Tracking')
    
    axes[1].plot(t, mag_traj, 'g')
    axes[1].set_ylabel('Magnitude (reconstructed)')
    
    axes[2].plot(t, gamma_traj, 'm')
    axes[2].set_ylabel('Damping')
    axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()
    
    trajectories = freq_traj.reshape(1, -1)
    slopes, pvals = compute_session_long_slopes(trajectories, fs)
    print(f"Frequency slope: {slopes[0]:.3f} Hz/hour, p = {pvals[0]:.4f}")
    
    likelihood = slope_change_likelihood(trajectories, fs)
    print(f"Max slope change likelihood at t = {t[np.argmax(likelihood)]:.1f} s")