# Evolutionary Linear Kalman Filter Tuning for Signal Processing

This project demonstrates the use of a Kalman Filter to process and filter noisy signals.   
The [project](Kalman_Filter.ipynb) includes a synthetic signal generator, noise addition, and a Kalman Filter implementation. Additionally, it performs a grid search to find the optimal parameters for the Kalman Filter.

An [evolutionary algorithm](Evolutionary_Optimizator.ipynb) implementation is available to find the optimal parameters for the Kalman Filter.  
The algorithm is based on a genetic algorithm that evolves a population of Kalman Filter parameters to minimize the error between the filtered signal and the true signal.

## Kalman Filter

The Kalman Filter is an algorithm that uses a series of measurements observed over time, containing statistical noise and other inaccuracies, and produces estimates of unknown variables that tend to be more accurate than those based on a single measurement alone. It is widely used in time series analysis, control systems, and signal processing.

### Key Components of the Kalman Filter:
- **State Vector (x)**: Represents the variables of interest.
- **State Transition Matrix (F)**: Describes how the state evolves from one time step to the next.
- **Observation Matrix (H)**: Maps the true state space into the observed space.
- **Process Noise Covariance (Q)**: Represents the uncertainty in the process model.
- **Measurement Noise Covariance (R)**: Represents the uncertainty in the measurements.
- **Error Covariance Matrix (P)**: Represents the estimated accuracy of the state estimate.

### Kalman Filter Equations:
1. **Prediction Step**:
    - Predicted State Estimate: $$\hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1}+B_k u_k$$
    - Predicted Error Covariance: $$P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$

2. **Update Step**:
    - Innovation or Measurement pre-fit Residual: $$y_k = z_k - H_k \hat{x}_{k|k-1}$$
    - Innovation Covariance: $$S_k = H_k P_{k|k-1} H_k^T + R$$
    - Optimal Kalman Gain: $$K_k = P_{k|k-1} H_k^T S_k^{-1}$$
    - Updated State Estimate: $$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k y_k$$
    - Updated Error Covariance: $$P_{k|k} = (I - K_k H_k) P_{k|k-1}$$
    - Measurement post-fit residual: $$y_{k|k} = z_k - H_k \hat{x}_{k|k}$$

The Kalman Filter iteratively performs these prediction and update steps to refine the state estimates and reduce uncertainty.


## Evolutionary Algorithm

The evolutionary algorithm is a subset of evolutionary computation, a generic population-based optimization algorithm inspired by biological evolution. The algorithm evolves a population of candidate solutions to an optimization problem using operations such as mutation, crossover, and selection. 

In this project, the evolutionary algorithm is based on the Differential Evolution algorithm, which is a simple and efficient evolutionary algorithm that optimizes a problem by iteratively improving candidate solutions with regard to a given measure of quality. The algorithm is particularly useful for optimizing complex, multi-modal functions.

The Differential Evolution algorithm works as follows:
1. **Initialization**: A population of candidate solutions is randomly generated.
2. **Mutation**: For each candidate solution, a mutant vector is created by adding the weighted difference between two population vectors to a third vector.
3. **Crossover**: The mutant vector is mixed with the target vector to create a trial vector.
4. **Selection**: The trial vector is compared to the target vector, and the one with the better fitness is selected for the next generation.

This process is repeated for a number of generations or until a stopping criterion is met. The result is a set of optimized parameters that minimize the error between the filtered signal and the true signal.

## References

- The Kalman Filter implementation is based on the following resources:
    - [Kalman Filter - Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
    - [Kalman Filter for Beginners with MATLAB Examples - Phil Kim](https://www.mathworks.com/matlabcentral/fileexchange/45465-kalman-filter-for-beginners)
    - [Kalman Filter - Python for Engineers](https://pythonforengineers.com/kalman-filter)
- The evolutionary algorithm implementation is based on the DEAP library: https://deap.readthedocs.io/en/master/


## Project Structure

- `KalmanFilter.py`: Contains the implementation of the Kalman Filter.
- `SignalGenerator.py`: Contains the implementation of the Signal Generator.
- [`Kalman_Filter.ipynb`](Kalman_Filter.ipynb): Jupyter notebook demonstrating the usage of the Kalman Filter and the grid search for optimal parameters.
- [`Evolutionary_Optimizator.ipynb`](Evolutionary_Optimizator.ipynb): Jupyter notebook demonstrating the usage of the genetic algorithm to find the optimal parameters for the Kalman Filter.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- DEAP (for genetic algorithm)
- tqdm (for interface progress bar)
- tabulate (for interface table formatting)
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

1. Run the Jupyter notebook [`Kalman_Filter.ipynb`](Kalman_Filter.ipynb) to see the Kalman Filter in action and perform a grid search for optimal parameters.

2. The notebook includes the following steps:
    - Generate a synthetic signal.
    - Add noise to the signal.
    - Apply the Kalman Filter to the noisy signal.
    - Perform a grid search to find the optimal Q and R parameters for the Kalman Filter.
    - Plot the results and visualize the filtered signal.

3. To use the evolutionary algorithm to find the optimal parameters for the Kalman Filter, run the Jupyter notebook [`Evolutionary_Optimizator.ipynb`](Evolutionary_Optimizator.ipynb).
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
