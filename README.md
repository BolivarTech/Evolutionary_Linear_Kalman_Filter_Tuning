# Kalman Filter Signal Processing

This project demonstrates the use of a Kalman Filter to process and filter noisy signals. The project includes a synthetic signal generator, noise addition, and a Kalman Filter implementation. Additionally, it performs a grid search to find the optimal parameters for the Kalman Filter.

A evolutionary algorithm option is available to find the optimal parameters for the Kalman Filter. The algorithm is based on a genetic algorithm that evolves a population of Kalman Filter parameters to minimize the error between the filtered signal and the true signal.

## References

- The genetic algorithm implementation is based on the DEAP library: https://deap.readthedocs.io/en/master/

## Project Structure

- `KalmanFilter.py`: Contains the implementation of the Kalman Filter.
- `SignalGenerator.py`: Contains the implementation of the Signal Generator.
- `Kalman_Filter.ipynb`: Jupyter notebook demonstrating the usage of the Kalman Filter and the grid search for optimal parameters.
- `Evolutive_Optimizator.ipynb`: Jupyter notebook demonstrating the usage of the genetic algorithm to find the optimal parameters for the Kalman Filter.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- DEAP (for genetic algorithm)
- tqdm (optional, for progress bar)
- Jupyter Notebook (for running the notebook)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/BolivarTech/KalmanFilterSignalProcessing.git
    cd KalmanFilterSignalProcessing
    ```

2. Install the required packages:
    ```sh
    pip install -r `requirements.txt`
    ```

## Usage

1. Run the Jupyter notebook `Kalman_Filter.ipynb` to see the Kalman Filter in action and perform a grid search for optimal parameters.

2. The notebook includes the following steps:
    - Generate a synthetic signal.
    - Add noise to the signal.
    - Apply the Kalman Filter to the noisy signal.
    - Perform a grid search to find the optimal Q and R parameters for the Kalman Filter.
    - Plot the results and visualize the filtered signal.

3. To use the evolutionary algorithm to find the optimal parameters for the Kalman Filter, run the Jupyter notebook `Evolutive_Optimizator.ipynb`.
4. The notebook includes the following steps:
    - Generate a synthetic signal.
    - Add noise to the signal.
    - Apply the Kalman Filter to the noisy signal.
    - Use the genetic algorithm to find the optimal Q and R parameters for the Kalman Filter.
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
# Get time vector
time= SigGen.get_time()

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
kf.plot_results(noisy_signal, true_signal, filtered_signal, time)
```
