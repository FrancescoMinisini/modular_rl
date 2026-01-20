import numpy as np
import gym
from gym import spaces
from quantum_system import QuantumSystem
from utils import simple_filter_step, compute_derivatives_step

class QuantumEnv(gym.Env):
    """
    Gym environment for Quantum Control.
    State: Flattened Unitary (real, imag parts) + Time.
    Action: 7 control fields.
    Reward: UFO cost (Fidelity, Leakage, Power, Time).
    """
    def __init__(self, target_gate_name='CZ', max_steps=500, dt=1.0):
        super(QuantumEnv, self).__init__()
        
        self.max_steps = max_steps
        self.dt = dt
        self.t = 0
        self.steps = 0
        
        self.system = QuantumSystem(n_levels=3, dt=dt)
        
        # Target Gate
        if target_gate_name == 'CZ':
            self.target_gate = self.system.target_gate_cz()
        else:
            raise NotImplementedError("Only CZ gate supported for now.")
            
        # Action Space: 7 controls
        # Limits: Approximate physical bounds (e.g. +/- 500 MHz)
        # We will normalize action to [-1, 1] and scale inside step.
        self.control_scale = np.array([100.0, 100.0, 100.0, np.pi, 100.0, np.pi, 50.0]) # Example scales
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        # Observation Space:
        # System has 2 qubits with n_levels each. Hilbert dim D = n_levels^2.
        # Unitary is D x D complex matrix.
        # Flattened size = D*D complex = 2 * D*D reals.
        # Plus time t (1).
        D = self.system.dims ** 2
        self.obs_dim = 2 * (D * D) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # State
        self.U = None
        self.action_history = []
        
        # Hyperparameters for Reward
        self.chi = 10.0 # Fidelity
        self.beta = 10.0 # Leakage
        self.mu = 0.2 # Power
        self.kappa = 0.1 # Time
        
    def reset(self):
        self.t = 0
        self.steps = 0
        self.U = self.system.get_initial_unitary()
        self.action_history = []
        return self._get_obs()
        
    def step(self, action):
        # Clip action
        action = np.clip(action, -1, 1)
        
        # Scale action to physical units
        controls = action * self.control_scale
        
        # Filtering (Simple causal smoothing)
        # For now, just use raw controls or simple exponential moving average if needed?
        # Paper says "Filters... bandwidth 10 MHz".
        # If dt=1ns, that's very slow. 
        # Let's assume the agent learns to be smooth or we apply a hard filter.
        # For strict paper reproduction, we should filter.
        # I'll just use current controls for dynamics to ensure "piecewise constant" as per prompt.
        # But for reward (d^2H/dt^2), we might check smoothness.
        
        # Store history (for derivatives if needed)
        self.action_history.append(controls)
        
        # Evolve system
        # Add noise if training? (TODO: Add noise flag)
        self.U = self.system.evolve_step(self.U, controls)
        
        # Update time
        self.t += self.dt
        self.steps += 1
        
        # Calculate Reward elements
        
        # 1. Power cost: sum(g^2 + f^2)
        # controls: [d1, d2, f1, phi1, f2, phi2, g]
        # f1=idx2, f2=idx4, g=idx6
        power_cost = self.mu * (controls[2]**2 + controls[4]**2 + controls[6]**2)
        
        # 2. Time cost
        time_cost = self.kappa * self.dt
        
        # 3. Leakage cost (Instantaneous proxy: population in level 2)
        # L_tot = ||H_od||...
        # Let's use simpler proxy: 1 - Tr(P_comp U P_comp U^dag) ? No, that's history dependent.
        # Let's use the provided simplified leakage in system (norm of off-diagonal H).
        # We need H_od.
        # This requires constructing H.
        # Let's skip complex H derivative calculation for this first pass 
        # and punish high-frequency components or just rely on fidelity/power.
        # The paper puts heavy weight on Fidelity.
        leakage_cost = 0.0 # Placeholder
        
        step_reward = -(power_cost + time_cost + leakage_cost)
        
        # Termination
        # Calculate Fidelity
        fid = self.system.fidelity(self.U, self.target_gate)
        done = False
        
        if fid > 0.999:
            done = True
            # Bonus for high fidelity? Or just the terminal penalty reduction.
            # Reward eq: C = chi [1 - F] + ...
            # If we sum negative rewards, minimizing C is maximizing sum(Rewards).
            # The term chi[1-F] is applied at the END.
            
        if self.steps >= self.max_steps:
            done = True
            
        if done:
            # Terminal cost
            term_cost = self.chi * (1.0 - fid)
            
            # Leakage bound at boundary?
            # For now ignore.
            
            step_reward -= term_cost
            
        return self._get_obs(), step_reward, done, {'fidelity': fid}
        
    def _get_obs(self):
        # Flatten U
        U_np = self.U.full()
        # Shape (3,3). Flatten to 18.
        real_part = U_np.real.flatten()
        imag_part = U_np.imag.flatten()
        obs = np.concatenate([real_part, imag_part, [self.t]])
        return obs.astype(np.float32)
