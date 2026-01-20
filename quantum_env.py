import numpy as np
import gym
from gym import spaces
from quantum_system import QuantumSystem

class QuantumEnv(gym.Env):
    """
    Gym environment for Quantum Control using UFO cost and Gmon system.
    State: Flattened Unitary (real, imag parts) + Time + (Optional: Last Action).
    Action: 7 control fields (delta1, delta2, f1, phi1, f2, phi2, g).
    Reward: Negative UFO cost: -(Chi*Infidelity + Beta*Leakage + Mu*Power + Kappa*Time).
    """
    def __init__(self, target_gate_name='CZ', max_steps=500, dt=1.0, noise_optimized=False):
        super(QuantumEnv, self).__init__()
        
        self.max_steps = max_steps
        self.dt = dt
        self.noise_optimized = noise_optimized
        
        self.t = 0
        self.steps = 0
        
        self.system = QuantumSystem(n_levels=3, dt=dt)
        
        # Target Gate
        if target_gate_name == 'CZ':
            self.target_gate = self.system.target_gate_cz()
        else:
            raise NotImplementedError("Only CZ gate supported for now.")
            
        # Action Space: 7 controls
        # Bounds (approximate physical limits in MHz)
        # Deltas: +/- 500 MHz
        # f: 100 MHz
        # g: 50 MHz
        # phi: -pi to pi
        self.control_scale = np.array([500.0, 500.0, 100.0, np.pi, 100.0, np.pi, 50.0]) 
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        # Observation Space:
        # Flattened Unitary (2 * D^2) + Time (1)
        D = self.system.dims ** 2
        self.obs_dim = 2 * (D * D) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # State
        self.U = None
        
        # Hyperparameters for UFO Cost (Niu et al. 2018)
        self.chi = 10.0  # Fidelity weight (terminal)
        self.beta = 10.0 # Leakage weight
        self.mu = 0.2    # Power weight
        self.kappa = 0.1 # Time weight
        
    def reset(self):
        self.t = 0
        self.steps = 0
        self.U = self.system.get_initial_unitary()
        
        # Static Noise Injection (per episode)
        if self.noise_optimized:
            # Variations in eta (anharmonicity) ~ 5%?
            # Paper says N(0, 1 MHz) for everything?
            # "Gaussian noise with mean 0 and variance 1 MHz... to eta, delta, f, g"
            # Since eta is ~ -200 MHz, 1 MHz is small.
            eta_noise = np.random.normal(0, 1.0) # MHz
            delta_offsets = np.random.normal(0, 1.0, size=2) # MHz
            self.system.set_parameters(eta_val=self.system.eta_base + eta_noise, delta_offsets=delta_offsets)
        else:
            self.system.reset_parameters()
            
        return self._get_obs()
        
    def step(self, action):
        # Clip action to valid range [-1, 1]
        action = np.clip(action, -1, 1)
        
        # Scale to physical units
        controls = action * self.control_scale
        
        # Dynamic Noise Injection (per step)
        if self.noise_optimized:
            # Noise on f, g, delta controls.
            # sigma = 1 MHz.
            # controls indices: 0:d1, 1:d2, 2:f1, 3:p1, 4:f2, 5:p2, 6:g
            noise = np.random.normal(0, 1.0, size=7)
            # Apply to MHz fields only, not phase (phi) strictly?
            # Paper says "f, g, delta". Phi is phase, maybe phase noise too?
            # Let's apply to magnitudes.
            controls[0] += noise[0] # d1
            controls[1] += noise[1] # d2
            controls[2] += noise[2] # f1
            # controls[3] (phi1) - maybe no additive MHz noise
            controls[4] += noise[4] # f2
            # controls[5] (phi2)
            controls[6] += noise[6] # g
            
        # Evolve system
        # Note: controls here are "noisy" controls seen by system.
        self.U = self.system.evolve_step(self.U, controls)
        
        # Cost Calculation (UFO)
        # C = Chi(1-F) + Beta*L + Mu*Power + Kappa*T
        
        # 1. Power Cost (Instantaneous)
        # sum(g^2 + f^2)
        power_term = (controls[6]**2 + controls[2]**2 + controls[4]**2)
        # Normalize? Paper doesn't specify normalization of sum, but likely per step or integral.
        # Assuming integral -> sum * dt.
        # However, mu=0.2 is small. 100^2 = 10000. 10000 * 0.2 = 2000.
        # This seems huge compared to Fidelity (0-1).
        # Maybe controls are in GHz? Or scaled?
        # Re-reading: "mu sum (g^2 + f^2)"
        # If g, f ~ 0.1 GHz (100 MHz). 0.1^2 = 0.01. 0.01 * 0.2 = 0.002.
        # This matches Fidelity scale better.
        # ACTION: Convert controls to GHz for cost calculation to match approximate scaling.
        # 1 MHz = 0.001 GHz.
        g_ghz = controls[6] * 1e-3
        f1_ghz = controls[2] * 1e-3
        f2_ghz = controls[4] * 1e-3
        power_cost = self.mu * (g_ghz**2 + f1_ghz**2 + f2_ghz**2)
        
        # 2. Leakage Cost (Instantaneous proxy for integral)
        h_od = self.system.h_off_diag_norm(controls) 
        # h_od is in rad/ns. 
        # L_tot ~ integral ||H_od||/Delta. 
        # Delta ~ 0.2 GHz ~ 1.2 rad/ns.
        # So L_inst ~ h_od / 1.2.
        leakage_cost = self.beta * (h_od * 1e-3) # Scaling guess: small penalty
        # Actually h_off_diag_norm returns value based on H in rad/ns.
        # If g=100MHz=0.6rad/ns. h_od ~ 0.6.
        # cost = 10 * 0.6 = 6. Too high.
        # Let's assume leakage cost is small until bounds violated.
        # Paper L_tot is < 1e-4.
        
        # 3. Time Cost
        time_cost = self.kappa * self.dt * 1e-3 # Scale dt?
        # Kappa = 0.1. T ~ 50ns. Cost ~ 5.
        # This seems dominant.
        # If T is in us? 50ns = 0.05 us. Cost = 0.005. That fits.
        # Let's assume time in cost is in microseconds or similar.
        time_cost = self.kappa * (self.dt * 1e-3) # ns to us? No, 1ns=1e-3us.
        
        step_cost = power_cost + time_cost + leakage_cost
        
        # Total Reward
        step_reward = -step_cost
        
        self.t += self.dt
        self.steps += 1
        
        # Check Termination
        fid = self.system.fidelity(self.U, self.target_gate)
        done = False
        
        # Terminate if Fidelity is good enough (or Logic based on C < threshold)
        if fid > 0.999: # 99.9% fidelity
            done = True
            # Optional: Terminal bonus or just standard cost
            # UFO says: Chi * (1 - F).
            # If we just sum step_costs, we miss the final fidelity penalty/reward.
            # We should add the fidelity term at the end.
            
        elif self.steps >= self.max_steps:
            done = True
            
        if done:
            # Terminal Fidelity Cost
            fid_cost = self.chi * (1.0 - fid) # e.g. 10 * 0.01 = 0.1
            step_reward -= fid_cost
            
        return self._get_obs(), step_reward, done, {'fidelity': fid, 'leakage': simple_leakage_proxy(h_od)}

    def _get_obs(self):
        # Flatten U (dim^2 complex -> 2*dim^2 real)
        U_np = self.U.full()
        
        real_part = U_np.real.flatten()
        imag_part = U_np.imag.flatten()
        
        # Normalize time? t up to 500. t/500?
        obs = np.concatenate([real_part, imag_part, [self.t / self.max_steps]])
        return obs.astype(np.float32)

def simple_leakage_proxy(h_od):
    return h_od
