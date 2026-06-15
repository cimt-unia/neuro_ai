# EEG data **4D object**

<br>

<img width="366" height="324" alt="image" src="https://github.com/user-attachments/assets/639e0d49-8aaa-45d1-9f1e-e3393d8fb0f7" />

```
Three dimensions of space and one dimension of time.
```


<br>


### 1. The Two Pillars of MNE
MNE separates your data into two distinct parts:
1.  **The Data Array (`raw._data`):** A simple grid of numbers (Volts). It has no idea where the sensors are on the head.
2.  **The Info/Montage (`raw.info` & `raw.get_montage()`):** The "map" that tells MNE which number belongs to which physical location $(x, y, z)$.



<br>

### 2. The Data Shape: `(n_channels, n_times)`
When you load your BEL 280-channel system, the raw data is stored as a NumPy array.

```python
data = raw.get_data()
print(data.shape) 
# Output: (280, 67626)
```

*   **Axis 0 (Rows):** Represents the **Channels**. Row 0 is channel `E1`, Row 1 is `E2`, etc.
*   **Axis 1 (Columns):** Represents **Time**. Each column is a single sample point in time (at 500 Hz, each column is 0.002 seconds apart).

**The Problem:** If you look at `data[0, :]`, you see a wave. But MNE doesn't know if that wave is coming from the front of the head or the back. It’s just "Row 0."



<br>

### 3. The Montage: Giving Data a "Body"
This is where your `BELStandardizer` and the `.gpsc` file come in. A montage is essentially a dictionary that links a **Channel Name** to a **3D Coordinate**.

In MNE, these coordinates are stored in meters relative to the center of the head:
*   **X-axis:** Left (-) to Right (+)
*   **Y-axis:** Back (-) to Front (+) *(Note: This can vary by convention, but MNE usually follows Nasion=+Y)*
*   **Z-axis:** Down (-) to Up (+)

When you run `raw.set_montage(montage)`, MNE updates its internal `info` structure. It now knows that "Row 0" (E1) is located at `[0.02, 0.05, 0.08]` meters.



<br>

### 4. Accessing Coordinates: The "First Principles" Way
You don't need complex functions to get these; you can pull them directly from the object.

```python
# 1. Get the montage object
montage = raw.get_montage()

# 2. Get the dictionary of all positions
positions = montage.get_positions()['ch_pos']

# 3. Look at a specific channel
print(positions['E1']) 
# Output: array([ 0.021,  0.054,  0.089]) -> [x, y, z] in meters
```



<br>

### 5. How the "Lobe Logic" Works Mathematically
When we want to find the **Occipital** lobe, we are performing a **geometric filter**.

Imagine plotting all 280 channels on a 2D graph where the Y-axis is "Front-to-Back."

1.  **Extraction:** We pull the Y-coordinate for every channel.
    ```python
    y_values = [pos[1] for pos in positions.values()]
    ```
2.  **Sorting:** We sort these values. In a standard head model:
    *   High positive Y = Forehead (Frontal)
    *   Zero Y = Top/Center (Central)
    *   High negative Y = Back of head (Occipital)
3.  **Selection:** By taking the channels with the **most negative Y values**, we are mathematically isolating the sensors physically closest to the Inion (the back of the skull).

<br>

### 6. Why Dimensions Matter for Analysis
When you perform operations like **Average Reference** or **ICA**, MNE uses these dimensions differently:

*   **Spatial Operations (e.g., Average Reference):** MNE looks at **Axis 0** (all channels at one moment in time) and calculates the mean. It treats every channel equally unless you tell it otherwise.
*   **Temporal Operations (e.g., Filtering):** MNE looks at **Axis 1** (one channel across all time) to remove specific frequencies.
*   **Topographic Mapping:** When you plot a "topomap," MNE takes the value from **Axis 0** and places it on a circle using the $(x, y)$ coordinates from the **Montage**. Without the montage, MNE cannot draw a brain map; it can only draw line graphs.

<br>

### Summary Table

| Component | What it is | Shape/Type | Purpose |
| :--- | :--- | :--- | :--- |
| **Raw Data** | The electrical signal | `(280, 67626)` | Holds the voltage changes over time. |
| **Channel Names** | The labels | `List[str]` | Links "Row 0" to "E1". |
| **Montage** | The 3D Map | `Dict {name: [x,y,z]}` | Tells MNE where "E1" is on the head. |
| **Info Object** | The Container | `mne.Info` | Holds sampling rate, names, and montage together. |
