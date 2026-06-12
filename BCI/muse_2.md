### **Tutorial: Building a Mind-Controlled Pong Game with Muse 2**

This tutorial guides you through streaming EEG data from the Muse 2, processing brainwaves to detect mental states, and mapping those states to control a character in a Pygame environment.

---

### **1. Prerequisites & Setup**

Before starting, ensure you have the necessary hardware and software installed.

*   **Hardware:** Muse 2 Headset (MU-03 2018) .
*   **Python Version:** 3.9 or higher .
*   **Required Libraries:**
    *   `muselsl`: For connecting to the Muse 2 and streaming data via Lab Streaming Layer (LSL) .
    *   `pylsl`: The Python interface for LSL, used to receive the data stream .
    *   `pygame`: For rendering the game and handling input events .
    *   `numpy`: For numerical processing of EEG signals .

Install the libraries using pip:
```bash
pip install muselsl pylsl pygame numpy
```

**Note on Bluetooth:** 
*   **Windows:** You may need **BlueMuse** to create the LSL outlet if direct connection fails .
*   **macOS/Linux:** `muselsl` can often connect directly via BLE .

---

### **2. Step 1: Streaming Data from Muse 2**

The Muse 2 streams data using the Lab Streaming Layer (LSL) protocol, which ensures precise time-synchronization across devices . We will use `muselsl` to start the stream.

Create a file named `bci_game.py`. First, we define a function to find and connect to the Muse stream.

```python
import pylsl
import numpy as np
import pygame
import time

def find_muse_stream():
    """Finds the Muse EEG stream on the local network."""
    print("Looking for an EEG stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    if not streams:
        raise Exception("No EEG stream found. Is Muse connected and streaming?")
    inlet = pylsl.StreamInlet(streams[0])
    print("Connected to Muse stream.")
    return inlet
```

---

### **3. Step 2: Calibration Phase**

To distinguish between "Relaxed" and "Concentrated" states, we need to establish a baseline for the user. 
*   **Alpha Waves (8–12 Hz):** Associated with relaxation .
*   **Beta Waves (13–30 Hz):** Associated with active concentration and focus .

We will record 10 seconds of data for each state to calculate average power levels.

```python
def calibrate_user(inlet):
    """Calibrates the user by recording Relax and Concentrate baselines."""
    sample_rate = 256  # Muse 2 standard sampling rate
    duration = 10      # Seconds per state
    
    def get_average_power(seconds, state_name):
        print(f"\n--- {state_name.upper()} PHASE ---")
        print(f"Please {state_name} for {seconds} seconds...")
        start_time = time.time()
        powers = []
        
        while time.time() - start_time < seconds:
            sample, timestamp = inlet.pull_sample()
            if sample:
                # Use channels AF7 and AF8 (Forehead) for simplicity
                # Indices 0 and 1 usually correspond to AF7 and AF8 in Muse LSL
                eeg_data = np.array(sample[:4]) 
                # Simple power estimation: mean of absolute values (can be improved with FFT)
                power = np.mean(np.abs(eeg_data))
                powers.append(power)
        
        return np.mean(powers)

    relax_baseline = get_average_power(duration, "relax")
    time.sleep(2) # Short break
    concentrate_baseline = get_average_power(duration, "concentrate")
    
    # Set thresholds: If current power is closer to concentrate baseline, move up
    threshold = (relax_baseline + concentrate_baseline) / 2
    print(f"\nCalibration Complete.")
    print(f"Relax Baseline: {relax_baseline:.4f}")
    print(f"Concentrate Baseline: {concentrate_baseline:.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    return threshold
```

---

### **4. Step 3: The Game Engine (Pygame)**

We will create a simple "Paddle" game where the paddle moves **UP** when you concentrate and **DOWN** when you relax.

```python
class PaddleGame:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Muse 2 BCI Pong")
        self.clock = pygame.time.Clock()
        
        # Paddle properties
        self.paddle_width = 15
        self.paddle_height = 100
        self.paddle_x = 50
        self.paddle_y = self.height // 2
        self.speed = 5
        
        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.blue = (0, 0, 255)

    def draw(self, state):
        self.screen.fill(self.black)
        
        # Draw Paddle
        pygame.draw.rect(self.screen, self.white, 
                         (self.paddle_x, self.paddle_y, self.paddle_width, self.paddle_height))
        
        # Display Current State
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"State: {state}", True, self.blue)
        self.screen.blit(text, (300, 50))
        
        pygame.display.flip()

    def update_paddle(self, direction):
        if direction == "up":
            self.paddle_y -= self.speed
        elif direction == "down":
            self.paddle_y += self.speed
            
        # Keep paddle within screen bounds
        if self.paddle_y < 0:
            self.paddle_y = 0
        if self.paddle_y > self.height - self.paddle_height:
            self.paddle_y = self.height - self.paddle_height
```

---

### **5. Step 4: Main Loop & Integration**

This loop combines the LSL data reception with the game logic. It continuously pulls samples from the Muse and compares them against the calibrated threshold.

```python
def main():
    try:
        # 1. Connect to Muse
        inlet = find_muse_stream()
        
        # 2. Calibrate
        threshold = calibrate_user(inlet)
        
        # 3. Initialize Game
        game = PaddleGame()
        running = True
        current_state = "Idle"
        
        print("\nStarting Game! Close the window to quit.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get latest EEG sample
            sample, timestamp = inlet.pull_sample(timeout=0.0)
            
            if sample:
                # Calculate current power (same method as calibration)
                eeg_data = np.array(sample[:4])
                current_power = np.mean(np.abs(eeg_data))
                
                # Determine State
                if current_power > threshold:
                    current_state = "Concentrating"
                    game.update_paddle("up")
                else:
                    current_state = "Relaxing"
                    game.update_paddle("down")
            
            # Render Game
            game.draw(current_state)
            game.clock.tick(60) # 60 FPS
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
```

---

### **6. How to Run**

1.  **Turn on your Muse 2** and pair it with your computer via Bluetooth.
2.  **Start the Stream:** Open a terminal and run:
    ```bash
    muselsl stream
    ```
    *Keep this terminal open.* It will broadcast the EEG data over LSL .
3.  **Run the Game:** In a new terminal, run:
    ```bash
    python bci_game.py
    ```
4.  **Follow the Calibration:** 
    *   Sit still and **relax** for 10 seconds.
    *   Then, **concentrate** hard (e.g., do mental math) for 10 seconds.
5.  **Play:** Once calibrated, the paddle will move based on your mental state.

### **Troubleshooting & Tips**

*   **Signal Quality:** Ensure the Muse sensors are wetted (with water or conductive gel) and sitting firmly on your forehead (AF7/AF8) and behind the ears (TP9/TP10) . Poor contact leads to noisy data.
*   **Artifacts:** Blinking creates large spikes in the data. The simple `np.mean(np.abs(...))` method used here is robust, but for better performance, consider using a bandpass filter (e.g., 8–30 Hz) to isolate Alpha/Beta waves .
*   **Latency:** LSL introduces minimal latency, but Bluetooth interference can cause drops. Stay close to your computer's Bluetooth receiver .

### **Sources**
*   [Muselsl Documentation](https://pypi.org/project/muselsl/) - Library for streaming Muse data via LSL.
*   [Lab Streaming Layer (LSL)](https://labstreaminglayer.readthedocs.io/) - Protocol for real-time time-series data exchange.
*   [Muse 101 Development Guide](https://anushmutyala.medium.com/muse-101-how-to-start-developing-with-the-muse-2-right-now-a1b87119be5c) - Basics of connecting to Muse 2 with Python.
*   [Pygame Pong Tutorial](https://ryanstutorials.net/pygame-tutorial/pygame-pong-simple.php) - Base logic for the game engine.
