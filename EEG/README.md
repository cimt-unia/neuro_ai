# **EEG Signal Theory** 


<img width="419" height="386" alt="image" src="https://github.com/user-attachments/assets/b815fe70-ab9c-4c5f-b3b4-1a24ea341e95" />



<br>

<br>


### What Is a Signal?
A **signal** is just a **measurement that changes over time**.

- Your heartbeat? A signal (beats per minute over time).
- Temperature in your room? A signal.
- The volume of music from your speaker? A signal.

In EEG, the signal is **voltage** (electrical pressure) measured at your scalp, changing **thousands of times per second**.




-----------------------------------------



<br>

<br>

### **Part 1: How Does the Brain Make Electricity?**



<img width="660" height="319" alt="image" src="https://github.com/user-attachments/assets/828492d2-9bfe-4076-9e41-0db406f7a932" />



Your brain is made of **neurons** (nerve cells). When a neuron "fires," it lets charged particles (ions like Na⁺, K⁺) flow in and out. This creates a tiny **electric current**.

- One neuron = too weak to measure.
- But **millions firing together** in sync? That creates a small electric field that reaches your scalp.


This voltage is **microscopic**: about **10 to 100 microvolts (µV)**.  
For comparison: a AA battery is **1,500,000 µV**. So brain signals are **1/15,000th** of a battery!

That’s why EEG is so sensitive—and so easily polluted by noise.

**VIDEO:** [**Neuronal electrophysiology**](https://www.youtube.com/watch?v=oa6rvUJlg7o)


<br>


<br>



### **Part 2: What We Record?**

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/2502727e-44b0-4b39-a958-2adaf72c24ce" />



#### 1. **Why Skull Matters**
- The **skull is a strong low-pass filter**: it smears and weakens fast, high-frequency signals.
- Scalp EEG sees only the “tip of the iceberg”—large, synchronized cortical patches.
- iEEG bypasses the skull → sees **local, fast, and deep** activity.

#### 2. **What Generates the Signal?**
- Only **pyramidal neurons** (aligned perpendicular to the scalp) produce dipoles strong enough to be recorded.
- Their **postsynaptic potentials** (not action potentials!) summate over ~50,000+ neurons to create measurable fields.
- This is why EEG reflects **input processing**, not output firing.

#### 3. **When Do We Use iEEG?**
- When scalp EEG **can’t localize** the seizure focus (e.g., deep temporal lobe).
- When surgery is planned, and we need to **map eloquent cortex** (speech, motor areas) to avoid damage.
- SEEG (stereotactic EEG) uses **depth electrodes** to probe deep structures like the hippocampus—something scalp EEG cannot do reliably.


#### Three Main Approaches Compared

| Feature | **Scalp EEG** | **Intracranial EEG (iEEG)** | 
|--------|----------------|-------------------------------|
| **Invasiveness** | Non-invasive (electrodes on skin) | Invasive (surgery required) | 
| **Electrode Placement** | On scalp using 10–20/10–10 system | Directly on cortex (ECoG grids) or deep in brain (SEEG depth electrodes) | 
| **What It Records** | Volume-conducted, spatially blurred signals from **superficial cortical pyramidal neurons** (synchronized postsynaptic potentials) | **Local field potentials** from small neuron populations; captures high-frequency activity (e.g., gamma, HFOs) | 
| **Spatial Resolution** | Low (~1–3 cm); blurred by skull/scalp | Very high (<1 cm); can resolve single gyri or nuclei | 
| **Temporal Resolution** | Excellent (sub-millisecond) | Excellent (sub-millisecond) | Excellent |
| **Frequency Range** | Best for <70 Hz (alpha, beta, etc.); attenuates high frequencies | Captures up to 200+ Hz (including high-frequency oscillations, HFOs) |
| **Primary Use** | Epilepsy screening, sleep studies, cognitive research, coma assessment | Pre-surgical mapping for drug-resistant epilepsy; research on memory, language 
| **Duration** | Minutes to days (ambulatory) | Days to weeks (inpatient monitoring) | 

---
**VIDEO:** [**Introduction to EEG**](https://www.youtube.com/watch?v=T7MKlPYiL48&t=385s)



<br>

<br>


### **Part 3: What Is Sampling Rate?**

<img width="1000" height="349" alt="image" src="https://github.com/user-attachments/assets/aa901358-6af8-4314-8837-5560e32fa42a" />


Voltage is **continuous**, but computers only understand **numbers**—discrete snapshots.

So we **sample**: take a voltage reading many times per second.

- **Sampling rate = how many snapshots per second**.
- Your  **600 samples per second** → every **1.0 milliseconds**, it takes a reading.

#### Why Not Just Sample Once Per Second?
Because brain waves wiggle **fast**!

- Alpha waves (8–13 Hz) wiggle **8 to 13 times per second**.
- To capture one full wiggle, you need **at least 2 samples** (one at the peak, one at the trough).
- But in practice, you need **much more** to see the shape clearly.

> Imagine drawing a circle using only 4 dots—you’d get a diamond. Use 100 dots—you get a smooth circle.

### The Nyquist Theorem:
> **You cannot accurately record a wave that wiggles faster than half your sampling rate.**

- If you sample at **600 Hz**, the fastest wave you can record is **300 Hz**.
- Any real wave faster than 300 Hz will **trick** your system into looking like a slower wave. This is called **aliasing**.


So: **Nyquist frequency = sampling rate ÷ 2**.  
In your case: **600 ÷ 2 ≈ 300 Hz**.


<br>

<br>


### **Part 4: What Is a Fourier Transform?** 

<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/dc232320-1046-4243-87fb-cf091e48868f" />

Your EEG signal is a **messy mix** of many rhythms happening at once:
- Slow delta (sleep)
- Medium alpha (relaxation)
- Fast gamma (thinking)

The **Fourier Transform** answers:  
> “What pure sine waves (frequencies) are hiding inside this messy signal?”

- We use **power** (energy) at each frequency.


<br>

<br>


### **Part 5: Filtering — Cleaning the Signal**

<img width="624" height="344" alt="image" src="https://github.com/user-attachments/assets/2da9dce4-e246-4cba-9430-65a811fd4e0f" />


#### Why Filter?

We use **filters** to keep only the frequencies we care about.

##### 1. Bandpass Filter (1–40 Hz)
- **Low cutoff (1 Hz)**: Removes super-slow drifts (e.g., from sweat changing electrode contact).
- **High cutoff (40 Hz)**: Removes muscle noise (which is very fast, >50 Hz).



##### 2. Notch Filter (60 Hz)
- A **very narrow** filter that removes **only 60 Hz** (and a tiny bit around it).
- Leaves everything else untouched.


<br>

<br>


### **Part 6: Re-referencing — Choosing Your “Zero”**

<img width="300" height="308" alt="image" src="https://github.com/user-attachments/assets/4f227d43-cdb8-458a-b83b-99057b327c74" />

Remember: EEG measures **differences**, not absolute voltage.

But what should we subtract from each electrode?

#### Common Choices:
1. **One mastoid**: Simple, but mastoid is located behind the ear and might have its own noise.
2. **Average of all electrodes**: Assumes total brain activity averages to zero (reasonable).
   - This is the **average reference**.

#### How It Works:
- Compute the average voltage across all 59 EEG channels at each moment.
- Subtract that average from every channel.


This often **cancels out noise** that affects all electrodes equally (like distant electrical interference).



<br>

<br>



### **Part 7: Bad Channels — The Broken Microphones**

<img width="600" height="304" alt="image" src="https://github.com/user-attachments/assets/9f60037b-5f98-4b2a-a77c-5aaee280f4c6" />


Sometimes an electrode isn’t touching well (hair, dry gel). It might:
- Show a **flat line** (no signal)
- Show **wild spikes** (intermittent contact)

We detect these by checking **variance** (how much the signal wiggles).

- Healthy channel: medium wiggle → medium variance.
- Dead channel: flat → near-zero variance.
- Noisy channel: huge spikes → very high variance.

We flag extremes as “bad.” Later, we can **interpolate** them (guess their signal from neighbors).



<br>

<br>



### **Part 8: ICA — Untangling Mixed Signals**

<img width="500" height="281" alt="image" src="https://github.com/user-attachments/assets/9fcc782f-2328-4dfc-8add-b259ca467cdf" />

#### The Core Problem:
Your EEG isn’t just brain. It’s a **mix** of:
- Brain waves
- Eye blinks 
- Heartbeats
- Neck muscles
<img width="200" height="100" alt="image" src="https://github.com/user-attachments/assets/ed7bc19e-0d93-49cb-b0ae-973c744c123f" />

All mixed together at every electrode.

#### What ICA Does:



ICA says:  
> “Suppose this combination was made by mixing a few pure ingredients. Can I reverse-engineer what those ingredients were?”

It finds **independent components**—patterns that:
- Have consistent spatial shape (topography)
- Vary independently over time

```
# After ICA, interpolate 
cleaned = ica.apply(raw)
if bads:
    cleaned.info['bads'] = bads
    cleaned.interpolate_bads(reset_bads=True)
    print(f"Interpolated {len(bads)} bad channels")
```


<br>





<br>


### **Part 9: 10-20 EEG system**

<img width="1000" height="403" alt="image" src="https://github.com/user-attachments/assets/93ecbd72-57a2-4c32-a4e4-1e9106851365" />

EEG measures the brain’s electrical activity generated by synchronized firing of millions of neurons. This signal is very weak (microvolts) and must pass through skull, tissue, and hair before being recorded.

**Why Standardization Matters**:  
To ensure reproducibility and allow comparison across individuals and studies, standardized electrode placement systems were developed.

**The 10-20 System (Established 1947)**:
- Uses **4 anatomical landmarks**: nasion (front), inion (back), and left/right pre-auricular points (ears).
- Electrodes are placed at **10% or 20% intervals** along key head circumferences (sagittal and coronal).
- Includes **21 standard electrodes**.
- Labels reflect **brain regions**:
  - **Fp** = fronto-polar, **F** = frontal, **C** = central, **T** = temporal, **P** = parietal, **O** = occipital
- **Laterality**: 
  - Odd numbers (1,3,5,7) → **left hemisphere**
  - Even numbers (2,4,6,8) → **right hemisphere**
  - **"z"** (e.g., Cz, Fz) → **midline**

**Higher-Density Extensions**:
- **10-10 System**: Adds electrodes at **10% intervals**, resulting in **~81 channels** for finer spatial resolution.
- **10-5 System**: Further subdivides to **5% intervals**, supporting up to **~320 electrodes**—used in advanced research and source imaging.

**EEG Montages (How Signals Are Referenced)**:
1. **Bipolar**: Measures voltage difference between **adjacent electrodes**; good for detecting focal abnormalities.
2. **Referential**: All electrodes referenced to a **common point** (e.g., mastoid or average); shows global activity.
3. **Laplacian (Local Average Reference)**: Estimates local activity by subtracting a weighted average of **nearby electrodes**; enhances spatial specificity and reduces volume conduction effects.

**Key Benefit**:  
These standardized systems link scalp electrode positions to underlying brain functions, enabling consistent interpretation across clinical and research settings.





<img width="800" height="593" alt="image" src="https://github.com/user-attachments/assets/f039672f-48b7-4a0e-a09b-c37554e2fb3c" />

<br>

### **Download demo data:** 
https://drive.google.com/file/d/1ZguSllqH66k9pCSLTsqXQCz3GfUYnrnr/view?usp=sharing

