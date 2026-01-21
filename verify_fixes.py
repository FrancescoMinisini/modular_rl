import numpy as np
import torch
from quantum_env import QuantumEnv
from trpo_agent import TRPOAgent
import time

def verify_physics():
    print("=== Verifying Physics & Cost ===")
    env = QuantumEnv(target_alpha=0.1, max_steps=10, dt=1.0, noise_optimized=False)
    
    # 1. Obs Projection (A1)
    obs = env.reset()
    expected_dim = 2 * (3**4) + 1 # 2 * 81 + 1 = 163
    print(f"Obs Dim: {len(obs)} (Expected {expected_dim})")
    assert len(obs) == expected_dim
    
    # 2. Step & Leakage
    action = np.zeros(7) # No controls, just drift
    # Drift has 0 coupling if g=0?
    # Our system init sets g=0?
    # No, base params are just eta=-200. "g" is a control input.
    # Control 0 implies g=0.
    # So H_od should be 0? 
    # Let's check.
    
    next_obs, rew, done, info = env.step(action)
    print(f"Step 1 Info: {info}")
    
    # Check Leakage History
    h_od_block = env.h_od_block_history[-1]
    norm = np.linalg.norm(h_od_block)
    print(f"H_od block norm with 0 controls: {norm}")
    # Coupling op is a2+a1 + ...
    # if g=0, H_c = 0.
    # So H_od should be 0.
    assert norm < 1e-9
    
    # Apply some coupling
    action_g = np.zeros(7)
    action_g[6] = 0.5 # g=10 MHz
    next_obs, rew, done, info = env.step(action_g)
    norm_g = np.linalg.norm(env.h_od_block_history[-1])
    print(f"H_od block norm with g=10MHz: {norm_g}")
    assert norm_g > 0
    
    # 3. Time Feature (A2)
    # env.steps should be 2 now. max=10. feat = 2/10 = 0.2
    t_feat = next_obs[-1]
    print(f"Time Feat: {t_feat} (Expected 0.2)")
    assert np.isclose(t_feat, 0.2)
    
    # 4. Termination
    # If we force cost low?
    # Hard to force cost low without solving control.
    # But we can update max_steps to 3.
    next_obs, rew, done, info = env.step(action_g) # Step 3 (done if max=3)
    # But max is 10.
    assert not done
    
    # Check cost components
    # Leakage integral should be calculated if history >= 3
    # We have 3 steps. History len 3.
    # It should have computed a derivative term.
    print(f"Leakage Cost (Step 3): {info['leakage']}")
    
    print("Physics Verified.")

def verify_agent():
    print("\n=== Verifying Agent ===")
    agent = TRPOAgent(obs_dim=163, act_dim=7)
    obs = np.random.randn(163)
    
    # Test Output Range (Tanh)
    for _ in range(10):
        act, raw = agent.get_action(obs)
        assert np.all(act >= -1.0) and np.all(act <= 1.0)
        # Check Tanh relation
        assert np.allclose(act, np.tanh(raw), atol=1e-5)
    
    print("Agent Action Range Verified.")

if __name__ == "__main__":
    verify_physics()
    verify_agent()
