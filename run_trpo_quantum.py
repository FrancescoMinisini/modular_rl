import numpy as np
import torch
import argparse
import time
import os
import json
import csv
from quantum_env import QuantumEnv
from trpo_agent import TRPOAgent

def save_checkpoint(path, agent, iter_idx, args, best_fid):
    torch.save({
        'iteration': iter_idx,
        'agent_state': agent.state_dict(),
        'args': vars(args),
        'best_fid': best_fid
    }, path)

def load_checkpoint(path, agent):
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found.")
        return 0, 0.0
    
    ckpt = torch.load(path)
    agent.load_state_dict(ckpt['agent_state'])
    start_iter = ckpt['iteration'] + 1
    best_fid = ckpt.get('best_fid', 0.0)
    print(f"Resumed from iteration {start_iter}")
    return start_iter, best_fid

def train(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Logging Directories
    log_dir = f"logs/{args.target}_seed{args.seed}_noise{args.noise_optimized}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Log File (JSONL)
    log_file_path = os.path.join(log_dir, "training_log.jsonl")
    
    # Environment
    env = QuantumEnv(
        target_gate_name=args.target, 
        max_steps=args.max_steps, 
        dt=args.dt,
        noise_optimized=args.noise_optimized
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Agent
    device = torch.device("cpu") # Use CPU for small nets/env
    agent = TRPOAgent(obs_dim, act_dim, device=device)
    
    # Resume capability
    start_iter = 0
    best_fid = 0.0
    
    latest_ckpt_path = os.path.join(log_dir, "checkpoint_latest.pth")
    best_ckpt_path = os.path.join(log_dir, "checkpoint_best.pth")
    
    if args.resume:
        # If resume flag is set, try to load latest
        start_iter, best_fid = load_checkpoint(latest_ckpt_path, agent)
    
    # Training Loop
    start_time = time.time()
    
    for i_iter in range(start_iter, args.n_iter):
        iter_start = time.time()
        
        # Collect trajectories
        rollouts = []
        total_steps = 0
        batch_rew = []
        batch_fid = []
        batch_leakage = [] # If available
        
        while total_steps < args.timesteps_per_batch:
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
                # Mask is 0 if done ONLY if it's terminal state, not timeout.
                # But here we default to 0 if done for simplicity in GAE.
                traj['mask'].append(0 if done else 1)
                
                obs = next_obs
                ep_rew += reward
                total_steps += 1
                
                if done:
                    batch_rew.append(ep_rew)
                    batch_fid.append(info.get('fidelity', 0))
                    batch_leakage.append(info.get('leakage', 0)) # Placeholder if environment returns it
            
            # Process trajectory
            rollouts.extend([{'obs': o, 'act': a, 'rew': r, 'mask': m} for o,a,r,m in zip(traj['obs'], traj['act'], traj['rew'], traj['mask'])])

        # Update
        success, loss = agent.update(rollouts)
        
        # Calculate Metrics
        avg_rew = float(np.mean(batch_rew))
        avg_fid = float(np.mean(batch_fid))
        max_fid = float(np.max(batch_fid)) if batch_fid else 0.0
        avg_leak = float(np.mean(batch_leakage)) if batch_leakage else 0.0
        runtime = time.time() - iter_start
        
        # Log to Console
        print(f"Iter {i_iter} | Steps {total_steps} | AvgRew {avg_rew:.4f} | AvgFid {avg_fid:.4f} | MaxFid {max_fid:.4f} | Loss {loss:.4f} | Time {runtime:.1f}s")
        
        # Log to File
        log_entry = {
            'iter': i_iter,
            'avg_reward': avg_rew,
            'avg_fidelity': avg_fid,
            'max_fidelity': max_fid,
            'avg_leakage': avg_leak,
            'loss': float(loss),
            'success': success,
            'runtime': runtime,
            'timestamp': time.time()
        }
        
        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Checkpointing
        # Save Latest
        save_checkpoint(latest_ckpt_path, agent, i_iter, args, best_fid)
        
        # Save Best
        if avg_fid > best_fid:
            best_fid = avg_fid
            print(f"New Best Fidelity: {best_fid:.4f}")
            save_checkpoint(best_ckpt_path, agent, i_iter, args, best_fid)
        
        # Converged?
        if avg_fid > 0.999:
            print("Converged!")
            break
            
    print(f"Training finished.")

def evaluate_baseline_sgd(args):
    """
    Simple SGD baseline for control optimization.
    Optimizes a fixed sequence of controls directly.
    """
    # Placeholder as before
    print("SGD Baseline: Differentiable simulation not available. Skipping exact SGD.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="CZ")
    parser.add_argument("--n_iter", type=int, default=100) 
    parser.add_argument("--timesteps_per_batch", type=int, default=20000) 
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--noise_optimized", action='store_true', help="Enable noise during training")
    parser.add_argument("--baseline", action='store_true', help="Run SGD baseline instead of RL")
    parser.add_argument("--resume", action='store_true', help="Resume from latest checkpoint if available")
    
    args = parser.parse_args()
    
    if args.baseline:
        evaluate_baseline_sgd(args)
    else:
        train(args)
