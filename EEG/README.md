# **EEG Signal Theory** 





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




Your brain is made of **neurons** (nerve cells). When a neuron "fires," it lets charged particles (ions like Na⁺, K⁺) flow in and out. This creates a tiny **electric current**.

- One neuron = too weak to measure.
- But **millions firing together** in sync? That creates a small electric field that reaches your scalp.


This voltage is **microscopic**: about **10 to 100 microvolts (µV)**.  
For comparison: a AA battery is **1,500,000 µV**. So brain signals are **1/15,000th** of a battery!

That’s why EEG is so sensitive—and so easily polluted by noise.



<br>


<br>



### **Part 2: How Do We Record It? (The Electrode)**

An **EEG electrode** is a small metal disc stuck to your scalp with conductive gel.

- It doesn’t *send* electricity—it only *listens*.
- It measures the **difference in voltage** between itself and another point (the “reference”).


But here’s the catch: **you always need two points** to measure voltage. So every EEG channel is really:  
**Electrode A – Reference Electrode**

> Imagine two water tanks connected by a pipe. You don’t care how full each is—you care about the **difference in water level**. That difference makes water flow. Similarly, EEG measures **voltage differences**.



<br>

<br>


### **Part 3: What Is Sampling Rate?**


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

#### The Big Idea:
Your EEG signal is a **messy mix** of many rhythms happening at once:
- Slow delta (sleep)
- Medium alpha (relaxation)
- Fast gamma (thinking)

The **Fourier Transform** answers:  
> “What pure sine waves (frequencies) are hiding inside this messy signal?”

- We use **power** (energy) at each frequency.

### Power Spectral Density (PSD)
This is just a **graph** showing:
- **X-axis**: Frequency (Hz) → how fast the wave wiggles
- **Y-axis**: Power (µV²/Hz) → how strong that rhythm is

The **60 Hz spike**? That’s not your brain—it’s your **wall outlet**! Electrical systems in North America vibrate at 60 Hz, and your electrodes pick it up like an antenna.


<br>

<br>


### **Part 5: Filtering — Cleaning the Signal**

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

#### The Core Problem:
Your EEG isn’t just brain. It’s a **mix** of:
- Brain waves
- Eye blinks 
- Heartbeats
- Neck muscles

All mixed together at every electrode.

#### What ICA Does:
ICA says:  
> “Suppose this combination was made by mixing a few pure ingredients. Can I reverse-engineer what those ingredients were?”

It finds **independent components**—patterns that:
- Have consistent spatial shape (topography)
- Vary independently over time

#### How It Finds Eye Blinks:
- Eye blinks create a strong positive voltage at forehead electrodes, negative at back.
- This makes a **front-heavy topographic map**.
- The time course shows **sharp upward spikes** every few seconds.
- It also **correlates perfectly** with the EOG channel (placed near the eye).





