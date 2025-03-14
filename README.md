# Kalman Filter Signal Processing

This project demonstrates the use of a Kalman Filter to process and filter noisy signals. The project includes a synthetic signal generator, noise addition, and a Kalman Filter implementation. Additionally, it performs a grid search to find the optimal parameters for the Kalman Filter.

## Project Structure

- `KalmanFilter.py`: Contains the implementation of the Kalman Filter.
- `SignalGenerator.py`: Contains the implementation of the Signal Generator.
- `Kalman_Filter.ipynb`: Jupyter notebook demonstrating the usage of the Kalman Filter and the grid search for optimal parameters.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- tqdm (optional, for progress bar)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/BolivarTech/KalmanFilterSignalProcessing.git
    cd KalmanFilterSignalProcessing
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib tqdm
    ```

## Usage

1. Run the Jupyter notebook `Kalman_Filter.ipynb` to see the Kalman Filter in action and perform a grid search for optimal parameters.

2. The notebook includes the following steps:
    - Generate a synthetic signal.
    - Add noise to the signal.
    - Apply the Kalman Filter to the noisy signal.
    - Perform a grid search to find the optimal Q and R parameters for the Kalman Filter.
    - Plot the results and visualize the filtered signal.

## Example

The following example shows how to use the Kalman Filter to filter a noisy signal:

```python
from KalmanFilter import KalmanFilter
from SignalGenerator import SignalGenerator
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic signal
SigGen = SignalGenerator(frequency=480, sampling_rate=20000, duration=0.1)
true_signal = SigGen.generate_signal('square')
noisy_signal = SigGen.add_noise(true_signal, noise_std=1)

# Initialize Kalman filter
F = np.array([[1]])
H = np.array([[1]])
Q = np.array([[18.679136]])
R = np.array([[100]])
P = np.array([[1]])
x = np.array([[0]])

kf = KalmanFilter(F, H, Q, R, P, x)
filtered_signal = kf.filter(noisy_signal)

# Plot results
plt.plot(true_signal, label='True Signal')
plt.plot(noisy_signal, label='Noisy Signal')
plt.plot(filtered_signal, label='Filtered Signal')
plt.legend()
plt.show()