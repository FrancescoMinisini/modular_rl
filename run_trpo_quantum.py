import numpy as np
import torch
import argparse
import time
import os
import json
from quantum_env import QuantumEnv
from trpo_agent import TRPOAgent

def train(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Logging
    log_dir = f"logs/{args.target}_seed{args.seed}_noise{args.noise_optimized}"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training_log.jsonl")
    
    # Curriculum: Fix Gamma, Vary Alpha (0 to Pi)
    # Target Family: N(alpha, alpha, gamma)
    gamma_val = np.pi / 2 # Fixed gamma
    alpha_step_size = 0.1
    # Start alpha
    current_alpha = 0.1 
    
    # Initialize Env
    # Note: timesteps_per_batch in arg. Paper says 20,000 episodes.
    # Single core sim is slow. 20k episodes * 500 steps = 10M steps. Too slow for this demo.
    # We will use args.timesteps_per_batch as total steps per update (approx 40 eps).
    # User can increase this arg.
    
    env = QuantumEnv(target_alpha=current_alpha, max_steps=args.max_steps, dt=args.dt, noise_optimized=args.noise_optimized)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    device = torch.device("cpu")
    agent = TRPOAgent(obs_dim, act_dim, device=device)
    
    start_time = time.time()
    total_iters = 0
    
    # Loop indefinitely (until target reached and converged)
    while current_alpha <= np.pi + 1e-3:
        print(f"\n=== Curriculum Phase: Alpha = {current_alpha:.3f} rad, Gamma = {gamma_val:.3f} ===")
        env.update_target(current_alpha)
        
        phase_converged = False
        
        # Train for this alpha
        for i_iter in range(args.n_iter): # Max iters per phase
            iter_start = time.time()
            
            # Collection
            rollouts = []
            batch_steps = 0
            batch_rew = []
            batch_fid = []
            batch_cost = []
            batch_leak = []
            
            # Run episodes until batch size met
            while batch_steps < args.timesteps_per_batch:
                obs = env.reset()
                done = False
                traj = {'obs': [], 'act': [], 'rew': [], 'mask': []}
                ep_rew = 0
                
                while not done:
                    action = agent.get_action(obs)
                    next_obs, reward, done, info = env.step(action)
                    
                    traj['obs'].append(obs)
                    traj['act'].append(action)
                    traj['rew'].append(reward)
                    traj['mask'].append(0 if done else 1)
                    
                    obs = next_obs
                    ep_rew += reward
                    batch_steps += 1
                    
                    if done:
                        batch_rew.append(ep_rew)
                        batch_fid.append(info.get('fidelity', 0.0))
                        batch_cost.append(info.get('cost', 0.0))
                        batch_leak.append(info.get('leakage', 0.0))
            
                rollouts.extend([{'obs': o, 'act': a, 'rew': r, 'mask': m} for o,a,r,m in zip(traj['obs'], traj['act'], traj['rew'], traj['mask'])])

            # Update
            success, loss = agent.update(rollouts)
            
            # Stats
            avg_rew = np.mean(batch_rew)
            avg_fid = np.mean(batch_fid)
            avg_cost = np.mean(batch_cost)
            avg_leak = np.mean(batch_leak)
            runtime = time.time() - iter_start
            total_iters += 1
            
            print(f"Alpha {current_alpha:.2f} | Iter {i_iter} (G {total_iters}) | "
                  f"Rew {avg_rew:.1f} | Cost {avg_cost:.2f} | Fid {avg_fid:.4f} | "
                  f"Leak {avg_leak:.2f} | Time {runtime:.1f}s")
            
            # Log
            log_entry = {
                'iter': total_iters,
                'alpha': current_alpha,
                'avg_reward': float(avg_rew),
                'avg_cost': float(avg_cost),
                'avg_fidelity': float(avg_fid),
                'avg_leakage': float(avg_leak),
                'loss': float(loss),
                'timestamp': time.time()
            }
            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Check Convergence (Low Cost OR High Fidelity)
            # Paper: "UFO cost satisfiable"
            # If Cost < threshold? 
            # Or Fidelity > 0.99 for intermediate steps?
            if avg_fid > 0.99:
                print(f"--> Converged for Alpha {current_alpha:.3f}! Moving forward.")
                phase_converged = True
                break
                
        if not phase_converged:
            print(f"Warning: Failed to converge for Alpha {current_alpha:.3f} in {args.n_iter} iters. Advancing anyway (forced march).")
            
        current_alpha += alpha_step_size
        if current_alpha > np.pi and current_alpha < np.pi + alpha_step_size:
            current_alpha = np.pi # Ensure we hit exactly Pi

    print("Training Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="CZ")
    parser.add_argument("--n_iter", type=int, default=100) # Iters per Alpha
    parser.add_argument("--timesteps_per_batch", type=int, default=20000) 
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=1.0)
    # Optimized noise is now standard per paper, but flag kept
    parser.add_argument("--noise_optimized", action='store_true') 
    
    args = parser.parse_args()
    train(args)
