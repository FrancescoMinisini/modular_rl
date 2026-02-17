import numpy as np
import gym
from gym import spaces
from quantum_system import QuantumSystem

class ExactExponentialFilter:
    """
    Exact two-pole normalized double exponential smoothing filter (Appendix C).
    F_sample = 1/dt. Bw = 10 MHz.
    """
    def __init__(self, dim, dt, bw_mhz=10.0):
        self.dim = dim
        self.dt = dt # ns
        # F_sample in GHz = 1/dt(ns)
        # Bw in GHz = 10 MHz = 0.01 GHz
        # alpha = exp(-pi * Bw / F_sample)
        
        bw_ghz = bw_mhz * 1e-3
        f_sample_ghz = 1.0 / dt
        
        alpha = np.exp(-np.pi * bw_ghz / f_sample_ghz)
        
        self.a1 = (1 - alpha)**2
        self.b1 = -2 * alpha
        self.b2 = alpha**2
        
        self.reset()
        
    def reset(self):
        self.c_prev1 = np.zeros(self.dim)
        self.c_prev2 = np.zeros(self.dim)
        
    def filter(self, u_raw):
        # u_raw is c_RL[n]
        # c[n] = a1 * c_RL[n] - b1 * c[n-1] - b2 * c[n-2]
        c_new = self.a1 * u_raw - self.b1 * self.c_prev1 - self.b2 * self.c_prev2
        
        # Shift history
        self.c_prev2 = self.c_prev1.copy()
        self.c_prev1 = c_new.copy()
        
        return c_new

class QuantumEnv(gym.Env):
    """
    Refined Gym Env for Niu et al. (2018).
    Strict adherence to cost function, boolean boundary conditions, and noise models.
    """
    def __init__(self, target_alpha=np.pi, max_steps=500, dt=1.0, noise_optimized=False):
        super(QuantumEnv, self).__init__()
        
        self.max_steps = max_steps
        self.dt = dt
        self.noise_optimized = noise_optimized
        self.target_alpha = target_alpha
        
        self.system = QuantumSystem(n_levels=3, dt=dt)
        self.update_target(target_alpha)
            
        # Action Space: 7 controls [d1, d2, f1, p1, f2, p2, g]
        # Ranges: 
        # d, f, g: [-20, 20] MHz
        # phi: [0, 2pi]
        self.control_ranges = np.array([
            20.0, 20.0, 20.0, np.pi, 20.0, np.pi, 20.0 # Scales for [-1, 1] input
        ])
        
        # For phi, network outputs [-1, 1]. Map to [0, 2pi].
        # We'll treat network output as raw [-1, 1] then transform.
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        # Observation Space
        # A1: obs_dim = 2 * (dims^4) + 1
        D_squared = self.system.dims ** 4
        self.obs_dim = 2 * D_squared + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # Cost Weights (Eq 4)
        self.chi = 10.0   # Infidelity
        self.beta = 10.0  # Leakage L_tot
        self.mu = 0.2     # Power (Boundary only)
        self.kappa = 0.1  # Time penalty (Total runtime)
        # Note: Kappa * T -> Per step, penalty = Kappa * dt * unit_scale? 
        # User said "Kappa*T is proportional to total runtime". 
        # Let's accumulate K * dt/500? Or just K * 1 if T is "normalized"?
        # Paper usually assumes T in units of something.
        # If max_steps=500 and cost should be competitive, 0.1 * 500 = 50. High.
        # We will apply kappa per step as 0.1 / max_possible_steps? 
        # Or Just 0.1 * dt * 1e-3?
        # User: "ensure the sum equals K*T".
        # If T is in ns (500). K*T = 50. 
        # If T is in us (0.5). K*T = 0.05.
        # Given Chi=10, 0.05 is tiny. 50 is huge.
        # Let's assume T is normalized to some characteristic time or just use step count.
        # Re-reading: "kappa * T".
        # Let's try constant small penalty.
        self.time_penalty_per_step = self.kappa * (self.dt / self.max_steps) * 10.0 # Tuning

        self.filter = ExactExponentialFilter(dim=7, dt=dt, bw_mhz=10.0)
        
        # History for Derivatives (TSWT)
        self.h_od_block_history = [] # Stores off-diagonal blocks for derivative calc
        self.controls_history = [] # For boundary cost check

        
    def update_target(self, alpha, gamma=np.pi/2):
        self.target_alpha = alpha
        self.gamma_val = gamma
        self.target_gate = self.system.target_gate_niu(alpha, gamma=gamma)

    def reset(self):
        self.t = 0
        self.steps = 0
        self.U = self.system.get_initial_unitary()
        
        # Reset Logic
        # Reset Logic
        self.filter.reset()
        self.h_od_block_history = []
        self.controls_history = []
        
        # Parameter Noise
        self.system.reset_parameters()
        if self.noise_optimized:
             # Noise on Eta (Anharmonicity)
             eta_noise = np.random.normal(0, 1.0)
             self.system.set_parameters(self.system.eta_base + eta_noise)
            
        return self._get_obs()
        
    def step(self, action):
        # 1. Action to Controls
        action = np.clip(action, -1, 1)
        
        # Map [-1, 1] to physical units
        # d, f, g: scale by 20 -> [-20, 20]
        # phi: scale by PI then shift by PI -> [0, 2pi]
        
        controls = np.zeros(7)
        # d1, d2, f1, f2, g
        indices_linear = [0, 1, 2, 4, 6]
        controls[indices_linear] = action[indices_linear] * 20.0
        
        # phi1, phi2
        indices_phi = [3, 5]
        # action in [-1, 1] -> +1 -> [0, 2] -> * pi -> [0, 2pi]
        controls[indices_phi] = (action[indices_phi] + 1.0) * np.pi
        
        # 2. Filter
        controls_filtered = self.filter.filter(controls)
        self.controls_history.append(controls_filtered)
        
        # 3. Add Control Noise (Stochastic Environment)
        # E. Noise Model: Add Gaussian noise to amplitudes at EACH step.
        # std = 1 MHz = 1e-3 GHz (if units are GHz) or just 1.0 if units MHz.
        # System uses MHz for params.
        # Noise on: eta (anharm), g (coupling), d1, d2, f1, f2
        
        controls_noisy = controls_filtered.copy()
        
        # Noise amounts (std=1MHz)
        noise_std = 1.0
        
        if self.noise_optimized:
            # Add N(0, 1MHz) to Delta1(0), Delta2(1), F1(2), F2(4), G(6)
            # And Eta (handled in system setter, but here we need to perturb it per step?)
            # Paper: "Fluctuations ... to control amplitudes ... at each time step"
            # It lists eta as one of them.
            
            # Perturb Eta
            eta_noise = np.random.normal(0, noise_std)
            self.system.set_parameters(self.system.eta_base + eta_noise)
            
            # Perturb Controls
            # Indices: 0, 1, 2, 4, 6
            noise = np.random.normal(0, noise_std, size=5)
            controls_noisy[0] += noise[0]
            controls_noisy[1] += noise[1]
            controls_noisy[2] += noise[2]
            controls_noisy[4] += noise[3]
            controls_noisy[6] += noise[4]
            # Phases (3, 5) not noised
        else:
             # Ensure Eta is base
             self.system.set_parameters(self.system.eta_base)
            
        # 4. Evolve
        self.U = self.system.evolve_step(self.U, controls_noisy)
        
        # 5. Track H_OD for TSWT Leakage Cost
        # Uses Noisy controls because that's the physical Hamiltonian
        h_od_block = self.system.get_h_off_diag(controls_noisy)
        self.h_od_block_history.append(h_od_block)
        
        self.t += self.dt
        self.steps += 1
        
        # 6. Check Done / Cost Calculation
        fid = self.system.fidelity(self.U, self.target_gate)
        
        # Cost Calculation (UFO)
        # We need to compute C(T) to see if we terminate.
        # C = Chi*(1-F) + Beta*L_tot + Mu*Bounds + Kappa*T
        
        # L_tot Calculation (Eq 3)
        # L_tot = |Hod(0)|/Delta + |Hod(T)|/Delta + Integral(1/Delta^2 * |d^2 Hod/dt^2|)
        # Delta = 200 MHz
        Delta = 200.0
        
        # Helper to get norm of block (Frobenius norm of block accounts for half of full H_od norm sq)
        # ||H_od|| = sqrt(2 * ||block||^2) = sqrt(2) * ||block||
        def get_full_norm(block):
            return np.sqrt(2.0) * np.linalg.norm(block)

        # Current Leakage Term (Integral approximation)
        # We need derivatives of H_od.
        # Finite difference on history.
        # d^2/dt^2 H[n] ~ (H[n] - 2H[n-1] + H[n-2]) / dt^2
        
        leakage_integral = 0.0
        # We can calculate this incrementally or sum up history.
        # Summing history is safer for correctness.
        
        if len(self.h_od_block_history) >= 3:
            # We can compute 2nd derivative for indices 1 to N-1?
            # Discrete sum: sum_n ||H[n] - 2H[n-1] + H[n-2]|| / dt^2 * dt / Delta^2 (?)
            # Paper metric is integral.
            # Int |H''| dt ~ Sum |H''| * dt
            # H'' ~ (H_n - 2H_n-1 + H_n-2)/dt^2
            # So Sum |(H_n ...)|/dt^2 * dt = Sum |...| / dt
            
            # Vectorized calc for speed
            stack = np.array(self.h_od_block_history) # shape (Steps, Rows, Cols)
            # diff2 = stack[2:] - 2*stack[1:-1] + stack[:-2]
            # This corresponds to centered 2nd diff ??
            # Or just diff(diff(stack)).
            # diff(stack) -> size N-1
            # diff(diff(stack)) -> size N-2.
            
            d2_stack = np.diff(stack, n=2, axis=0) / (self.dt**2)
            
            # Norms of each 2nd deriv matrix
            # einsum to get frobenius norms efficiently?
            # norm = sqrt(sum abs(x)^2)
            d2_norms_sq = np.sum(np.abs(d2_stack)**2, axis=(1,2)) 
            # This is ||block||^2. Full norm is sqrt(2)*sqrt(block_sq)
            d2_norms_full = np.sqrt(2.0 * d2_norms_sq)
            
            leakage_integral = np.sum(d2_norms_full) * self.dt / (Delta**2)
            
        # Boundary terms
        l_bound = 0.0
        if len(self.h_od_block_history) > 0:
            h0 = self.h_od_block_history[0]
            ht = self.h_od_block_history[-1]
            l_bound = (get_full_norm(h0) + get_full_norm(ht)) / Delta

        L_tot = l_bound + leakage_integral
        
        # Boundary Power Cost
        # Mu * sum_(0, T) (g^2 + f^2) -- Actually Paper Eq 4 says sum_{t in {0,T}} (Boundary only)
        # "mu sum_{t in {0,T}} ..." implies t=0 and t=T only.
        c0 = self.controls_history[0]
        ct = self.controls_history[-1]
        pow0 = c0[6]**2 + c0[2]**2 + c0[4]**2
        powT = ct[6]**2 + ct[2]**2 + ct[4]**2
        C_bound = self.mu * (pow0 + powT)
        
        # Time Cost
        # Kappa * T. T is time in unknown units? 
        # Usually T in same units as 1/Kappa?
        # If Kappa=0.1 and T=500ns, C=50.
        C_time = self.kappa * self.t # self.t in ns
        # If T=20ns, C=2.
        
        # UFO Cost
        C_ufo = self.chi * (1.0 - fid) + self.beta * L_tot + C_bound + C_time
        
        # Termination Logic (B2)
        # "Satisfiable value of UFO cost"
        # We need a threshold.
        # E.g. Fid > 0.999 => Err < 0.001 => Cost ~ 10*0.001 = 0.01
        # L_tot small...
        # Let's say Threshold = 0.1? Or user configured.
        # Let's enforce max_steps as hardcap.
        # But if C_ufo is "good enough"?
        # User doesn't give number.
        # Let's stick to Max Steps for now, but allow Done if Fid > 0.9999?
        # User says "Termination logic is wrong... min(runtime upper bound, time to meet termination condition)".
        # Condition = "satisfiable value".
        # Let's define C_threshold = 0.05 (Arbitrary but tight).
        
        C_threshold = 0.05
        
        done = False
        if self.steps >= self.max_steps:
            done = True
        elif C_ufo < C_threshold:
            # Terminate early with success
            done = True
            
        # Reward
        # R = -C.
        # We return 0 intermediate and -C final?
        # Or -Delta C?
        # User: "Reward returned to TRPO should be negative cost: R = -C"
        # "Step rewards can be 0 except terminal... or distribute... sum must equal -C exactly".
        
        # Distributed Reward:
        # R_t = -(C_t - C_{t-1})
        # Sum(R) = - (C_T - C_0) = -C_T + C_0.
        # C_0 is initial cost (Fid~0, Time=0, etc).
        # We want Sum(R) = -C_T.
        # So R_t = -(C_t - C_{t-1}). And ensure C_{-1} = 0.
        
        # Need to reconstruct C_prev.
        # We can just store 'current_cost' in self.
        if not hasattr(self, 'prev_cost'):
            self.prev_cost = 0.0 # Effectively assuming cost starts at 0?
            # Actually C(0) is not 0. Fid is 1.0 (bad).
            # If we want Sum = -C_final,
            # We can set R_final = -C_final, R_step = 0.
            # But dense shaping helps learning.
            # Difference strategy:
            # R_step = - (Cost_current - Cost_prev)
            # Sum R = - (C_final - C_initial).
            # This maximizes -(C_final - C_init) = C_init - C_final.
            # Since C_init is constant for an episode, this is equivalent to minimizing C_final.
            
            # Let's compute C_current.
            pass
            
        current_cost = C_ufo
        reward = -(current_cost - self.prev_cost)
        self.prev_cost = current_cost
        
        # Check boundary condition for first step?
        # If steps==1, prev_cost was 0 (init).
        # So we paid C_1 - 0.
        # At end, we paid Sum (C_t - C_t-1) = C_T - C_0.
        # But we want total return = -C_T.
        # So we need to subtract C_0 at some point? Or just accept offset?
        # TRPO baseline handles offset.
        # User says "R = -C" explicitly.
        # If we return -C at end, and 0 otherwise:
        # R = -C_ufo if done else 0.
        # This is sparse. Might assume harder to learn.
        # "Step rewards can be 0 except terminal step, OR you can distribute terms... sum must equal -C exactly"
        # I will use the Distributed approach (Difference of Potentials) but ensuring the sum is exactly -C_T.
        # Sum_t (C_{t-1} - C_t) = C_0 - C_T.
        # To get -C_T, we need to subtract C_0.
        # So first reward should be: -(C_1 - C_0) - C_0 = -C_1.
        # Basically R_t = -C_t if t=1? No.
        # Simplest:
        # R = -C_ufo if done else 0.
        # But "fidelity does not improve" -> sparse reward might be issue.
        # User mentions "your dense shaping... breaks optimization target".
        # So sticking to R=-C (terminal) or exact diff is required.
        # Let's use **Difference of Potentials** where Phi(s) = -Cost(s).
        # R(s,a,s') = Phi(s') - Phi(s) - step_penalty?
        # No, just R = -C_final.
        # Let's try R = -C_final at end, 0 otherwise. Safe and compliant.
        
        reward = 0.0
        if done:
            reward = -C_ufo
            
        # Logging
        info = {
            'fidelity': fid,
            'cost': C_ufo,
            'leakage': L_tot,
            'b_cost': C_bound,
            't_cost': C_time
        }
            
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        U_np = self.U.full()
        obs = np.concatenate([
            U_np.real.flatten(), 
            U_np.imag.flatten(), 
            [self.steps / self.max_steps]
        ])
        return obs.astype(np.float32)
