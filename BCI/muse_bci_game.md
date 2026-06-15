## **Mind the Move: Developing a Brain-Computer Interface Game with Left-Right Motor Imagery** 
*by Prapas et al., published in *Information* (2023)* 
Link: https://www.mdpi.com/2078-2489/14/7/354
### 1. Executive Summary
This study presents the development and evaluation of a 3D Brain-Computer Interface (BCI) game designed to assess user adaptation and improvement in controlling a system via mental commands. The system utilizes non-invasive Electroencephalography (EEG) via the **Muse 2 headband**, processes signals using the **OpenViBE** platform, and renders the game environment in **Unity**.

The primary objective was not merely to create a game, but to scientifically evaluate whether users can improve their BCI control proficiency through training. The study involved **33 participants** and achieved a classification accuracy of **96.94%** using a Multi-Layer Perceptron (MLP) algorithm.

### 2. Technical Architecture & Methodology

#### 2.1 Hardware and Software Stack
*   **EEG Device:** Muse 2 Headband.
    *   *Rationale:* Chosen for portability, affordability, and ease of use.
    *   *Limitation:* It has only four dry electrodes (TP9, AF7, AF8, TP10), which do not cover the motor cortex (C3/C4 locations) optimally. To compensate, the authors used "eye polarization" (gazing left/right) combined with motor imagery.
*   **Data Streaming:**
    *   **BlueMuse:** Open-source software that streams EEG data from the Muse headset via Bluetooth.
    *   **Lab Streaming Layer (LSL):** A framework used to synchronize and transmit data streams between BlueMuse, OpenViBE, and the Unity game engine in real-time.
*   **Signal Processing:** **OpenViBE** platform.
*   **Game Engine:** **Unity** (for the 3D environment).

#### 2.2 Signal Processing Pipeline
The processing occurs in two phases: Offline (Training) and Online (Real-time Gameplay).

1.  **Acquisition:** Raw EEG data is captured at a sampling frequency of **256 Hz**.
2.  **Filtering:** A Chebyshev bandpass filter (8–40 Hz) is applied to remove noise and artifacts.
3.  **Epoching:** Signals are divided into non-overlapping windows of **3 seconds**. This means the game updates the avatar's action every 3 seconds based on the dominant mental state during that window.
4.  **Feature Extraction:** The signal is segmented into five frequency bands:
    *   Alpha (8–12 Hz)
    *   Beta 1 (12–20 Hz)
    *   Beta 2 (20–30 Hz)
    *   Gamma 1 (30–35 Hz)
    *   Gamma 2 (35–40 Hz)
    *   The **energy** of each band is calculated to form the feature vector.
5.  **Classification:** A **Multi-Layer Perceptron (MLP)** neural network is used.
    *   *Structure:* Two-layer MLP with hyperbolic tangent activation in the hidden layer and softmax in the output layer.
    *   *Classes:* Left Motor Imagery, Right Motor Imagery, and Eye Blink.

### 3. Experimental Design

#### 3.1 Participants
*   **Sample Size:** 33 subjects (18 males, 15 females), aged 21–45.
*   **Health Status:** All healthy with normal/corrected vision.
*   **Prior Experience:** 26 were novices; 7 had prior BCI experience.

#### 3.2 Data Collection Protocol
Each participant underwent three 5-minute recording sessions:
1.  **Blink:** Blinking hard every 1 second.
2.  **Left MI:** Looking left and imagining left-hand movement.
3.  **Right MI:** Looking right and imagining right-hand movement.
*   *Note:* The recordings were asynchronous (continuous performance without cues), resulting in 97 feature vectors per class per subject.

#### 3.3 The Game ("Mind the Move")
*   **Objective:** Control an avatar moving forward on a 3-lane platform to collect coins.
*   **Controls:**
    *   **Left MI:** Slide avatar left.
    *   **Right MI:** Slide avatar right.
    *   **Blink:** Jump (to collect airborne coins).
*   **Structure:** 50 coins arranged in 17 clusters.
*   **Procedure:**
    1.  10 practice trials.
    2.  10 evaluation trials.
    3.  Pre- and post-game testing of MI accuracy (15 trials each for left/right) to measure improvement.

### 4. Key Results

#### 4.1 Classification Accuracy
*   **Overall Accuracy:** **96.94%** across all subjects.
*   **True Positive Rate (TPR):**
    *   Left MI: 95.6%
    *   Right MI: 95.4%
    *   Blink: High precision (specific values vary by subject, but overall system performance was robust).

#### 4.2 Game Performance Metrics
*   **Average Game Score:** 27.6 coins collected (55.3% of total).
*   **Average Coin Clusters:** 10.04 clusters accessed (59%).
*   **User Grouping:**
    *   *Group 1 (Struggling):* 10 users (40–49.9% score). Difficulty synchronizing mental commands.
    *   *Group 2 (Competent):* 14 users (50–59.9% score).
    *   *Group 3 (Expert):* 9 users (60–78.8% score).

#### 4.3 User Improvement (Learning Effect)
The study statistically validated that users improved their BCI control after playing the game.
*   **Left MI Improvement:** Increased from 73.73% to 80.80% accuracy (**+7.1%**).
*   **Right MI Improvement:** Increased from 71.66% to 79.73% accuracy (**+8.07%**).
*   **Statistical Significance:** Paired samples t-tests confirmed these improvements were statistically significant ($p < 0.001$).

### 5. Discussion and Critical Analysis

#### Strengths
1.  **Large Sample Size:** With 33 participants, this study exceeds the average sample size (approx. 17.8) found in similar BCI gaming literature.
2.  **Robust Evaluation:** The use of multiple metrics (Accuracy, Game Score, Clusters, and Pre/Post Improvement) provides a holistic view of system performance.
3.  **Proof of Learning:** The statistical evidence that users *learn* to control the BCI better after exposure is a crucial finding for rehabilitation applications.

#### Limitations
1.  **Electrode Placement:** The Muse 2 does not cover the motor cortex (C3/C4). The authors mitigated this by combining gaze direction with motor imagery, but this may limit the purity of the MI signal.
2.  **Temporal Resolution:** The 3-second epoch window creates a delay. Users must wait 3 seconds for the command to register, which can disrupt the flow of gameplay.
3.  **Generalizability:** The classifier is trained specifically for this game and user group; it is not a plug-and-play solution for other games without retraining.

### 6. Conclusion and Future Work

The authors conclude that low-cost, commercial EEG headsets like the Muse 2 can be effectively used to create reliable BCI games. The high classification accuracy and demonstrated user improvement suggest potential for:
*   **Neurorehabilitation:** Using gamified BCI training for patients with motor disabilities.
*   **Robotics Control:** Adapting the training protocol for controlling wheelchairs or robotic arms.
*   **VR/AR Integration:** Extending the system to immersive Extended Reality environments.

