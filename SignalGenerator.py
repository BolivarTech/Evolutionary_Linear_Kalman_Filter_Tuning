import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    """
    A class for generating various types of signals.
    Supports sine, square, triangular, and ramp waveforms.
    """

    def __init__(self, frequency, sampling_rate, duration=1.0, amplitude=1.0):
        """
        Initialize the signal generator.

        Parameters:
        - frequency: Signal frequency in Hz
        - sampling_rate: Sampling rate in Hz
        - duration: Signal duration in seconds (default: 1.0)
        - amplitude: Signal amplitude (default: 1.0)
        """
        self.frequency = frequency
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.amplitude = amplitude

        # Create time vector
        self.t = np.arange(0, self.duration, 1/self.sampling_rate)
        self.num_samples = len(self.t)

    def sine_wave(self):
        """Generate a sine wave signal."""
        return self.amplitude * np.sin(2 * np.pi * self.frequency * self.t)

    def square_wave(self, duty_cycle=0.5):
        """Generate a square wave signal with adjustable duty cycle."""
        return self.amplitude * np.sign(np.sin(2 * np.pi * self.frequency * self.t - np.pi * (1 - duty_cycle)))

    def triangular_wave(self):
        """Generate a triangular wave signal."""
        return self.amplitude * (2/np.pi) * np.arcsin(np.sin(2 * np.pi * self.frequency * self.t))

    def ramp_wave(self):
        """Generate a ramp (sawtooth) wave signal."""
        t_mod = self.frequency * self.t
        return self.amplitude * 2 * (t_mod - np.floor(t_mod + 0.5))

    def add_noise(self, signal, noise_std):
        """Add Gaussian noise to a signal."""
        noise = np.random.normal(0, noise_std, self.num_samples)
        return signal + noise

    def plot_signal(self, signal, title="Signal"):
        """Plot the generated signal."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'{title} - {self.frequency} Hz, {self.sampling_rate} Hz sampling')
        plt.grid(True)
        plt.show()

    def generate_signal(self, signal_type, noise_std=0.0, duty_cycle=0.5):
        """
        Generate signal of specified type with optional noise.

        Parameters:
        - signal_type: 'sine', 'square', 'triangular', or 'ramp'
        - noise_std: Noise standard deviation (default: 0.0)
        - duty_cycle: Duty cycle for square wave (default: 0.5)

        Returns:
        - Generated signal array
        """
        if signal_type.lower() == 'sine':
            signal = self.sine_wave()
        elif signal_type.lower() == 'square':
            signal = self.square_wave(duty_cycle)
        elif signal_type.lower() == 'triangular':
            signal = self.triangular_wave()
        elif signal_type.lower() == 'ramp':
            signal = self.ramp_wave()
        else:
            raise ValueError("Signal type must be 'sine', 'square', 'triangular', or 'ramp'")

        if noise_std > 0:
            signal = self.add_noise(signal, noise_std)

        return signal


# Example usage
if __name__ == "__main__":
    # Create a 480 Hz signal sampled at 20 kHz
    sg = SignalGenerator(frequency=480, sampling_rate=20000, duration=0.05)

    # Generate different signals
    sine = sg.generate_signal('sine')
    square = sg.generate_signal('square')
    triangle = sg.generate_signal('triangular')
    ramp = sg.generate_signal('ramp')

    # Add noise to a sine wave
    noisy_sine = sg.generate_signal('sine', noise_std=0.3)

    # Plot each signal
    sg.plot_signal(sine, "Sine Wave")
    sg.plot_signal(square, "Square Wave")
    sg.plot_signal(triangle, "Triangular Wave")
    sg.plot_signal(ramp, "Ramp Wave")
    sg.plot_signal(noisy_sine, "Noisy Sine Wave")