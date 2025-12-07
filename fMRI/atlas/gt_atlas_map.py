# gt_atlas_map.py
"""
Glasser + Tian (GT) Atlas Parcellation Pipeline
- Handles variable TR across datasets
- Resamples to standardized duration/TR
- Standardizes AFTER resampling (correct order)
- No aggressive high-pass filtering
- Memory-efficient & parallelized
- Reusable across ABIDE, UKB, and other fMRI datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)

# Default standardization targets (can be overridden per call)
DEFAULT_TARGET_TR = 2.0
DEFAULT_TARGET_DURATION = 300.0  # 5 minutes → 150 timepoints


def resample_timeseries(ts, original_tr, target_tr=DEFAULT_TARGET_TR,
                        target_duration=DEFAULT_TARGET_DURATION, kind='cubic'):
    """
    Resample time series to target TR and duration.
    
    Args:
        ts: (T, n_rois) array
        original_tr: float, original TR in seconds
        target_tr: float, target TR in seconds
        target_duration: float, target duration in seconds
        kind: str, interpolation method (e.g., 'linear', 'cubic')
    
    Returns:
        resampled: (target_n_timepoints, n_rois) or None if insufficient data
    """
    if ts.ndim != 2:
        raise ValueError(f"Input must be 2D (T, n_rois), got shape {ts.shape}")
    
    original_time = np.arange(ts.shape[0]) * original_tr
    if original_time[-1] < target_duration:
        return None

    target_n = int(target_duration / target_tr)
    target_time = np.arange(target_n) * target_tr
    resampled = np.zeros((target_n, ts.shape[1]), dtype=np.float32)

    for roi_idx in range(ts.shape[1]):
        interp_func = interp1d(
            original_time, ts[:, roi_idx],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True
        )
        resampled[:, roi_idx] = interp_func(target_time)

    return resampled


class GlasserTianParcellator:
    """Efficient parcellation using Glasser (360) + Tian (54) atlases → 414 ROIs."""

    def __init__(self, atlas_dir: str):
        self.atlas_dir = Path(atlas_dir)
        self.glasser_nii = self.atlas_dir / "glasser_360_MNI152NLin6Asym.nii.gz"
        self.tian_nii = self.atlas_dir / "tian_subcortex_54_MNI152NLin6Asym.nii"
        self.roi_labels_csv = self.atlas_dir / "roi_labels.csv"
        self._validate_files()

    def _validate_files(self):
        missing = [f for f in [self.glasser_nii, self.tian_nii, self.roi_labels_csv] if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing atlas files: {missing}")
        roi_df = pd.read_csv(self.roi_labels_csv)
        if len(roi_df) != 414 or 'roi_name' not in roi_df.columns:
            raise ValueError("ROI labels must have exactly 414 rows with 'roi_name' column")

    def parcellate_subject(self, fmri_path: str, tr: float,
                          resample_atlases: bool = True,
                          target_tr: float = DEFAULT_TARGET_TR,
                          target_duration: float = DEFAULT_TARGET_DURATION):
        """
        Parcellate a single subject with correct TR handling and standardization order.
        
        Returns:
            (target_n_timepoints, 414) array or None if scan too short
        """
        fmri_img = image.load_img(fmri_path)

        # Resample atlases to fMRI space if needed
        if resample_atlases:
            resample_kwargs = {
                'target_img': fmri_img,
                'interpolation': 'nearest',
                'force_resample': True,
                'copy_header': True
            }
            glasser_res = image.resample_to_img(self.glasser_nii, **resample_kwargs)
            tian_res = image.resample_to_img(self.tian_nii, **resample_kwargs)
        else:
            glasser_res = self.glasser_nii
            tian_res = self.tian_nii

        # Masker: use real TR, no high-pass, standardize=False (do it later)
        masker_kwargs = {
            'standardize': False,
            't_r': tr,
            'detrend': True,
            'low_pass': 0.1,
            'memory': "nilearn_cache",
            'verbose': 0
        }

        g_ts = NiftiLabelsMasker(glasser_res, **masker_kwargs).fit_transform(fmri_img)
        t_ts = NiftiLabelsMasker(tian_res, **masker_kwargs).fit_transform(fmri_img)
        ts = np.concatenate([g_ts, t_ts], axis=1)  # (T, 414)

        # Resample to target TR/duration
        resampled = resample_timeseries(ts, tr, target_tr=target_tr, target_duration=target_duration)
        if resampled is None:
            logger.debug(f"Skipping {Path(fmri_path).name}: insufficient duration (<{target_duration}s)")
            return None

        # Standardize AFTER resampling (critical!)
        resampled = (resampled - resampled.mean(axis=0)) / (resampled.std(axis=0) + 1e-8)
        return resampled

    def process_dataset(self, fmri_paths, tr_values, n_jobs=-1,
                        target_tr=DEFAULT_TARGET_TR,
                        target_duration=DEFAULT_TARGET_DURATION):
        """
        Process entire dataset in parallel.
        
        Args:
            fmri_paths: list of fMRI file paths
            tr_values: array of TRs (one per subject)
            n_jobs: number of parallel jobs
            target_tr, target_duration: standardization targets
        
        Returns:
            processed_data: list of (T, 414) arrays
            valid_indices: indices of successfully processed subjects
        """
        logger.info(f"Processing {len(fmri_paths)} subjects with n_jobs={n_jobs}")

        # Check if atlases need resampling (once)
        test_img = image.load_img(fmri_paths[0])
        glasser_img = image.load_img(self.glasser_nii)
        tian_img = image.load_img(self.tian_nii)
        resample = not (
            np.allclose(test_img.affine, glasser_img.affine, atol=1e-3) and
            np.allclose(test_img.affine, tian_img.affine, atol=1e-3)
        )

        # >>> ADD THESE TWO LINES <<< 
        if resample:
            logger.warning("Atlases misaligned with fMRI data — will resample atlases")
        else:
            logger.info("Atlases already aligned with fMRI data — skipping resampling")
            
        def _process_one(idx):
            return idx, self.parcellate_subject(
                fmri_paths[idx], tr_values[idx], resample,
                target_tr=target_tr, target_duration=target_duration
            )

        if n_jobs == 1:
            results = [_process_one(i) for i in range(len(fmri_paths))]
        else:
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_process_one)(i) for i in range(len(fmri_paths))
            )

        processed_data, valid_indices = [], []
        for idx, data in results:
            if data is not None:
                processed_data.append(data)
                valid_indices.append(idx)

        logger.info(f"✅ Successfully processed {len(processed_data)}/{len(fmri_paths)} subjects")
        return processed_data, valid_indices

    def get_roi_labels(self):
        """Return ROI labels DataFrame."""
        return pd.read_csv(self.roi_labels_csv)


def create_analysis_phenotype(
    phenotype_df: pd.DataFrame,
    eid_col: str = 'SUB_ID',
    age_col: str = 'AGE_AT_SCAN',
    sex_col: str = 'SEX',
    target_col: str = 'DX_GROUP',
    sex_male_value=1,
    target_positive_value=1
):
    """
    Create standardized phenotype for modeling.
    
    Output columns: ['eid', 'Age', 'Sex', 'Target']
    - Sex: 1 = male, 0 = female
    - Target: 1 = case, 0 = control
    """
    return pd.DataFrame({
        'eid': phenotype_df[eid_col],
        'Age': phenotype_df[age_col],
        'Sex': (phenotype_df[sex_col] == sex_male_value).astype(int),
        'Target': (phenotype_df[target_col] == target_positive_value).astype(int)
    })