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
        return np.array([self.filter_step(z) for z in measurements])

    def calculate_error(self, true_signal, filtered_signal):
        """Calculate error metrics between true and filtered signals."""
        mse = np.mean((true_signal - filtered_signal) ** 2)  #Mean Squared Error
        rmse = np.sqrt(mse)  #Root Mean Square Error
        return mse, rmse

    def calculate_rms_value(self, signal):
        """Calculate the RMS value of a signal."""
        return np.sqrt(np.mean(signal ** 2))

    def calculate_rms_value_error(self, true_signal, filtered_signal):
        """
        Calculate the RMS value error between the true signal and the filtered signal.

        Parameters:
        - true_signal: The true signal without noise
        - filtered_signal: The signal after applying the Kalman filter

        Returns:
        - rms_value_error: The relative RMS value error between the true and filtered signals
        """
        org_rms = np.sqrt(np.mean(true_signal ** 2))
        filt_rms = np.sqrt(np.mean(filtered_signal ** 2))
        rms_value_error = np.abs(org_rms - filt_rms)
        return rms_value_error

    def plot_results(self, noisy_signal, true_signal, filtered_signal, time, zoom_begin=0, zoom_end=1, title=''):
        """
        Plot the results of the Kalman filter.

        Parameters:
        - noisy_signal: The noisy input signal
        - true_signal: The true signal without noise
        - filtered_signal: The signal after applying the Kalman filter
        - time: Array of time values corresponding to the signals
        - zoom_begin: Fraction of the signal to start the zoomed plot (default is 0 for begginning)
        - zoom_end: Fraction of the signal to end the zoomed plot (default is 1 for end)
        - title: Title of the plot (default is an empty string)
        """
        if not (0 <= zoom_begin < zoom_end <= 1):
            if zoom_begin < 0:
                zoom_begin = 0
            if zoom_end > 1:
                zoom_end = 1
            if zoom_begin >= zoom_end:
                zoom_begin, zoom_end = zoom_end, zoom_begin
        zoom_samples_begin = int(zoom_begin * len(time))
        zoom_samples_end = int(zoom_end * len(time))
        plt.figure(figsize=(12, 6))
        plt.plot(time[zoom_samples_begin:zoom_samples_end], noisy_signal[zoom_samples_begin:zoom_samples_end], 'gray', alpha=0.5, label='Noisy Signal')
        plt.plot(time[zoom_samples_begin:zoom_samples_end], true_signal[zoom_samples_begin:zoom_samples_end], 'b', linewidth=2.5, label='Original Signal')
        plt.plot(time[zoom_samples_begin:zoom_samples_end], filtered_signal[zoom_samples_begin:zoom_samples_end], 'r',linewidth=2.5, label='Filtered Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        if title:
            plt.title(title)
        else:
            if zoom_begin == 0 and zoom_end == 1:
                plt.title(f'Signal Filtered with Kalman Filter')
            else:
                plt.title(f'Signal Filtered with Kalman Filter (Zoomed)')
        plt.legend()
        plt.grid(True)
        plt.show()
