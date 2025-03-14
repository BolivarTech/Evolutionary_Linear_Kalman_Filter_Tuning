import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, F, H, Q, R, initial_P, initial_x):
        """
        Initialize the Kalman Filter.

        Parameters:
        - F: State transition matrix
        - H: Observation matrix
        - Q: Process noise covariance
        - R: Observation noise covariance
        - initial_P: Initial state covariance
        - initial_x: Initial state
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = initial_P
        self.x = initial_x

    def predict(self):
        """Prediction step of the Kalman filter."""
        self.x_pred = np.dot(self.F, self.x)
        self.P_pred = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x_pred

    def update(self, z):
        """Update step of the Kalman filter."""
        y = z - np.dot(self.H, self.x_pred)
        S = np.dot(self.H, np.dot(self.P_pred, self.H.T)) + self.R
        K = np.dot(self.P_pred, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x_pred + np.dot(K, y)
        self.P = self.P_pred - np.dot(K, np.dot(self.H, self.P_pred))
        return self.x

    def filter_step(self, z):
        """Apply one step of the Kalman filter to a single measurement."""
        self.predict()
        self.update(z)
        return self.x[0][0]

    def filter(self, measurements):
        """Apply Kalman filter to a sequence of measurements."""
        filtered_values = []
        for z in measurements:
            filtered_values.append(self.filter_step(z))
        return filtered_values

    def calculate_error(self, true_signal, filtered_signal):
        """Calculate error metrics between true and filtered signals."""
        mse = np.mean((true_signal - filtered_signal) ** 2)
        rmse = np.sqrt(mse)
        return mse, rmse

    def plot_results(self, input_signal, true_signal, filtered_signal):
        """Plot the original, noisy, and filtered signals."""
        plt.plot(input_signal, label='Señal con ruido')
        plt.plot(true_signal, label='Señal original')
        plt.plot(filtered_signal, label='Señal filtrada')
        plt.legend()
        plt.show()