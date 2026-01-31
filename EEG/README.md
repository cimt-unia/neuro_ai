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



<img width="324" height="399" alt="image" src="https://github.com/user-attachments/assets/26a0a9da-e387-45b3-b486-a5bc028377bd" />



Your brain is made of **neurons** (nerve cells). When a neuron "fires," it lets charged particles (ions like Na⁺, K⁺) flow in and out. This creates a tiny **electric current**.

- One neuron = too weak to measure.
- But **millions firing together** in sync? That creates a small electric field that reaches your scalp.




This voltage is **microscopic**: about **10 to 100 microvolts (µV)**.  
For comparison: a AA battery is **1,500,000 µV**. So brain signals are **1/15,000th** of a battery!

That’s why EEG is so sensitive—and so easily polluted by noise.



<br>


<br>



### **Part 2: How Do We Record It? (The Electrode)**

<img width="660" height="319" alt="image" src="https://github.com/user-attachments/assets/828492d2-9bfe-4076-9e41-0db406f7a932" />

An **EEG electrode** is a small metal disc stuck to your scalp with conductive gel.

- It doesn’t *send* electricity—it only *listens*.
- It measures the **difference in voltage** between itself and another point (the “reference”).


But here’s the catch: **you always need two points** to measure voltage. So every EEG channel is really:  
**Electrode A – Reference Electrode**

> Imagine two water tanks connected by a pipe. You don’t care how full each is—you care about the **difference in water level**. That difference makes water flow. Similarly, EEG measures **voltage differences**.



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

#### The Core Problem:
Your EEG isn’t just brain. It’s a **mix** of:
- Brain waves
- Eye blinks 
- Heartbeats
- Neck muscles

All mixed together at every electrode.

#### What ICA Does:
<img width="500" height="281" alt="image" src="https://github.com/user-attachments/assets/9fcc782f-2328-4dfc-8add-b259ca467cdf" />

ICA says:  
> “Suppose this combination was made by mixing a few pure ingredients. Can I reverse-engineer what those ingredients were?”

It finds **independent components**—patterns that:
- Have consistent spatial shape (topography)
- Vary independently over time

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/ed7bc19e-0d93-49cb-b0ae-973c744c123f" />

<img width="590" height="390" alt="image" src="https://github.com/user-attachments/assets/a4abb460-f54e-4a05-b60a-9dbe2fd16473" />






