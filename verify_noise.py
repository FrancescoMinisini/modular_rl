import numpy as np
import torch
from quantum_system import QuantumSystem
from quantum_env import QuantumEnv

def test_noise_fidelity():
    sys = QuantumSystem()
    target = sys.target_gate_cz()
    
    # Levels of noise (std dev of action in [-1, 1])
    # Control scale typically 500 MHz for deltas.
    noise_levels = [0.0, 0.01, 0.135, 0.5, 1.0] # 0, -4.6, -2.0, -0.7, 0 log_std
    
    control_scale = np.array([500.0, 500.0, 100.0, np.pi, 100.0, np.pi, 50.0])
    
    print(f"{'Action Std':<12} | {'Max Control (MHz)':<18} | {'Fidelity':<10}")
    print("-" * 50)
    
    for std in noise_levels:
        # Simulate 500 steps (full episode)
        U = sys.get_initial_unitary()
        
        # Constant random noise for the whole trajectory? 
        # Or random per step? RL agent outputs random per step.
        
        # Let's average over 5 seeds
        fids = []
        for _ in range(5):
            U = sys.get_initial_unitary()
            for _ in range(500): # 500 steps
                # Sample action
                action = np.random.normal(0, std, size=7)
                controls = action * control_scale
                U = sys.evolve_step(U, controls)
            
            f = sys.fidelity(U, target)
            fids.append(f)
            
        avg_fid = np.mean(fids)
        max_ctrl = std * 500.0 # Approximately
        print(f"{std:<12.4f} | {max_ctrl:<18.1f} | {avg_fid:<10.4f}")

if __name__ == "__main__":
    test_noise_fidelity()
