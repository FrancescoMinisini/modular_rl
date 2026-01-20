import numpy as np
import torch
import argparse
import time
import os
from quantum_env import QuantumEnv
from trpo_agent import TRPOAgent

def train(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment
    env = QuantumEnv(target_gate_name=args.target, max_steps=args.max_steps, dt=args.dt)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Agent
    device = torch.device("cpu") # Use CPU for small nets/env
    agent = TRPOAgent(obs_dim, act_dim, device=device)
    
    # Training Loop
    start_time = time.time()
    
    for i_iter in range(args.n_iter):
        # Collect trajectories
        rollouts = []
        total_steps = 0
        batch_rew = []
        batch_fid = []
        
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
                # But here we default to 0 if done.
                traj['mask'].append(0 if done else 1)
                
                obs = next_obs
                ep_rew += reward
                total_steps += 1
                
                if done:
                    batch_rew.append(ep_rew)
                    batch_fid.append(info.get('fidelity', 0))
            
            rollouts.extend([{'obs': o, 'act': a, 'rew': r, 'mask': m} for o,a,r,m in zip(traj['obs'], traj['act'], traj['rew'], traj['mask'])])

        # Update
        success, loss = agent.update(rollouts)
        
        # Log
        avg_rew = np.mean(batch_rew)
        avg_fid = np.mean(batch_fid)
        print(f"Iter {i_iter} | Steps {total_steps} | AvgRew {avg_rew:.4f} | AvgFid {avg_fid:.4f} | Loss {loss:.4f} | Success {success}")
        
        # Check success condition (Fidelity > 0.999 consistently?)
        if avg_fid > 0.999:
            print("Converged!")
            # Save model
            torch.save(agent.policy.state_dict(), "quantum_policy_converged.pth")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    torch.save(agent.policy.state_dict(), "quantum_policy_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="CZ")
    parser.add_argument("--n_iter", type=int, default=100) # Short default for testing
    parser.add_argument("--timesteps_per_batch", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200) # ns?
    parser.add_argument("--dt", type=float, default=1.0)
    args = parser.parse_args()
    
    train(args)
