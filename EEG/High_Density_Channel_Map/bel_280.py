# ────────────────────────────────────────────────
# STANDARDIZE MONTAGE FOR BEL 280-CHANNEL SYSTEM (optional but recommended)
# ────────────────────────────────────────────────

'''
# Example: EGI exports channels as '1', '2', ..., '257' + reference
rename_map = {str(i): f'E{i}' for i in range(1, 281)}
rename_map['REF CZ'] = 'Cz'  # Map reference to Cz

# Apply renaming
raw.rename_channels(rename_map)

# Apply montage from BEL .gpsc file
channels = bel_280.parse_gpsc("ghw280_from_egig.gpsc")
montage = bel_280.create_montage_from_gpsc(channels)
raw.set_montage(montage, on_missing="warn")
'''



from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import mne

def parse_gpsc(filepath: Path) -> List[Tuple[str, float, float, float]]:
    """Parse BEL .gpsc file into list of (name, x, y, z)."""
    channels = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    name = parts[0]
                    x, y, z = map(float, parts[1:4])
                    channels.append((name, x, y, z))
                except ValueError:
                    continue
    return channels

def create_montage_from_gpsc(
    channels: List[Tuple[str, float, float, float]],
    coord_frame: str = 'head'
) -> mne.channels.DigMontage:
    """Create MNE montage from parsed GPSC channels."""
    if not channels:
        raise ValueError("No valid channels provided.")
    
    gpsc_array = np.array([ch[1:4] for ch in channels])
    mean_pos = gpsc_array.mean(axis=0)
    
    ch_pos = {
        ch[0]: np.array([
            ch[1] - mean_pos[0],
            ch[2] - mean_pos[1],
            ch[3] - mean_pos[2]
        ]) / 1000.0  # Convert mm → meters
        for ch in channels
    }
    
    return mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=ch_pos.get('FidNz'),
        lpa=ch_pos.get('FidT9'),
        rpa=ch_pos.get('FidT10'),
        coord_frame=coord_frame
    )

class BELStandardizer:
    """Convenience class to standardize BEL EEG data."""
    def __init__(self, gpsc_file: Path, rename_map: Optional[Dict[str, str]] = None):
        self.gpsc_file = Path(gpsc_file)
        if not self.gpsc_file.exists():
            raise FileNotFoundError(f"GPSC file not found: {self.gpsc_file}")
        self.rename_map = rename_map or {}

    def standardize(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw = raw.copy()
        # 1. Rename channels
        if self.rename_map:
            existing_map = {old: new for old, new in self.rename_map.items() if old in raw.ch_names}
            if existing_map:
                raw.rename_channels(existing_map)
        # 2. Apply montage
        channels = parse_gpsc(self.gpsc_file)
        montage = create_montage_from_gpsc(channels)
        raw.set_montage(montage, on_missing='warn')
        return raw
    
