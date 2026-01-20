import numpy as np
import scipy.signal

class ActionFilter:
    """
    Applies a Gaussian filter to smooth the control signals, ensuring they are differentiable 
    for the leakage cost calculation.
    """
    def __init__(self, buffer_size=10, sigma=2.0):
        self.buffer_size = buffer_size
        self.sigma = sigma
        # Create Gaussian kernel
        x = np.arange(-buffer_size, buffer_size + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        self.kernel = kernel / kernel.sum()
        
    def smooth(self, trajectory):
        """
        Smooths the full trajectory of controls.
        Args:
            trajectory: Shape (T, n_actions)
        Returns:
            smoothed: Shape (T, n_actions)
        """
        smoothed = np.zeros_like(trajectory)
        for i in range(trajectory.shape[1]):
            smoothed[:, i] = np.convolve(trajectory[:, i], self.kernel, mode='same')
        return smoothed

def compute_derivatives(values, dt):
    """
    Computes 1st and 2nd derivatives using finite differences (batch).
    """
    first = np.gradient(values, dt, axis=0) # d/dt
    second = np.gradient(first, dt, axis=0) # d^2/dt^2
    return first, second

def simple_filter_step(current_val, prev_val, alpha=0.5):
    """Exponential moving average step."""
    if prev_val is None:
        return current_val
    return alpha * current_val + (1 - alpha) * prev_val

def compute_derivatives_step(history, dt):
    """
    Compute derivatives from history list.
    Returns (1st, 2nd) derivatives (vectors).
    """
    if len(history) < 3:
        if len(history) > 0:
            return np.zeros_like(history[0]), np.zeros_like(history[0])
        else:
            return np.array([]), np.array([])
    
    # Use last 3 points
    y = np.array(history[-3:])
    # y[0] = t-2, y[1] = t-1, y[2] = t
    
    # First deriv at t: (y[t] - y[t-1])/dt
    d1 = (y[2] - y[1]) / dt
    
    # Second deriv: (d1[t] - d1[t-1])/dt = (y[2] - 2y[1] + y[0]) / dt^2
    d2 = (y[2] - 2*y[1] + y[0]) / (dt**2)
    
    return d1, d2
