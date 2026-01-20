import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Utils for PyTorch TRPO
def flat_grad(grads, params):
    grad_flat = []
    for g, p in zip(grads, params):
        if g is None:
            grad_flat.append(torch.zeros_like(p).view(-1))
        else:
            grad_flat.append(g.view(-1))
    return torch.cat(grad_flat)

def flat_params(params):
    return torch.cat([p.data.view(-1) for p in params])

def set_params(params, flat_params):
    prev_ind = 0
    for p in params:
        flat_size = int(np.prod(list(p.size())))
        p.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(p.size()))
        prev_ind += flat_size

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)
    for _ in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.mean_layer = nn.Linear(32, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.val_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.val_layer(x)

class TRPOAgent:
    def __init__(self, obs_dim, act_dim, device="cpu"):
        self.policy = PolicyNetwork(obs_dim, act_dim).to(device)
        self.value_net = ValueNetwork(obs_dim).to(device)
        self.optimizer_val = optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.device = device
        self.max_kl = 0.01
        self.damping = 0.1
        self.gamma = 0.99
        self.lam = 0.95

    def get_action(self, obs):
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(obs_t)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.cpu().numpy()[0]

    def compute_returns_advantages(self, rewards, values, masks):
        # Generalized Advantage Estimation
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        running_return = 0
        prev_value = 0
        running_adv = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * masks[t]
            returns[t] = running_return
            
            # GAE
            if t == len(rewards) - 1:
                next_val = 0 # Assume terminal value 0 if done
            else:
                next_val = values[t+1]
                
            delta = rewards[t] + self.gamma * next_val * masks[t] - values[t]
            running_adv = delta + self.gamma * self.lam * masks[t] * running_adv
            advantages[t] = running_adv
            
        return returns, advantages

    def update(self, rollouts):
        # Check if rollouts empty
        if not rollouts:
            return

        # Prepare batch
        obs_batch = torch.FloatTensor(np.array([r['obs'] for r in rollouts])).to(self.device)
        act_batch = torch.FloatTensor(np.array([r['act'] for r in rollouts])).to(self.device)
        rew_batch = torch.FloatTensor(np.array([r['rew'] for r in rollouts])).to(self.device)
        mask_batch = torch.FloatTensor(np.array([r['mask'] for r in rollouts])).to(self.device)
        
        # Current Value Estimates
        with torch.no_grad():
            values = self.value_net(obs_batch).squeeze()
            
        # Compute Advantages
        returns, advantages = self.compute_returns_advantages(rew_batch, values, mask_batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy Update (TRPO)
        
        # Old Policy distribution
        with torch.no_grad():
            old_mean, old_std = self.policy(obs_batch)
            old_dist = torch.distributions.Normal(old_mean, old_std)
            old_log_probs = old_dist.log_prob(act_batch).sum(dim=1)

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    mean, std = self.policy(obs_batch)
            else:
                mean, std = self.policy(obs_batch)
            
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(act_batch).sum(dim=1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr = (ratio * advantages).mean()
            return -surr # Minimize negative expected reward

        def get_kl():
            mean, std = self.policy(obs_batch)
            dist = torch.distributions.Normal(mean, std)
            # Analytical KL for diagonal Gaussians with fixed std vs computed
            # KL(old || new)
            # kl = log(std_new/std_old) + (std_old^2 + (mean_old-mean_new)^2)/(2*std_new^2) - 0.5
            # Here old_std is fixed during this calc.
            
            # Using torch.distributions.kl_divergence would be easier if we rebuild old_dist
            # old_dist is detach().
            new_dist = torch.distributions.Normal(mean, std)
            old_dist_d = torch.distributions.Normal(old_mean.detach(), old_std.detach())
            kl = torch.distributions.kl.kl_divergence(old_dist_d, new_dist).sum(dim=1).mean()
            return kl

        # Compute Policy Gradient
        loss = get_loss()
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = flat_grad(grads, self.policy.parameters())

        # Fisher Vector Product
        def fisher_vector_product(v):
            kl = get_kl()
            # First derivative of KL wrt params
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = flat_grad(grads, self.policy.parameters())
            
            # Inner product with v
            kl_v = (flat_grad_kl * v).sum()
            
            # Second derivative (Hessian * v)
            grads_2nd = torch.autograd.grad(kl_v, self.policy.parameters())
            flat_grad_2nd = flat_grad(grads_2nd, self.policy.parameters())
            
            return flat_grad_2nd + self.damping * v

        # Conjugate Gradient Step
        step_dir = conjugate_gradient(fisher_vector_product, -loss_grad)

        # Line Search
        shs = 0.5 * (step_dir * fisher_vector_product(step_dir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        full_step = step_dir / lm[0]
        
        trpo_success = False
        old_params = flat_params(self.policy.parameters())
        old_loss = loss.item()
        
        for backtrack_idx in range(10):
            step = 0.5**backtrack_idx
            new_params = old_params + step * full_step
            set_params(self.policy.parameters(), new_params)
            
            new_loss = get_loss(volatile=True).item()
            kl = get_kl().item()
            
            if kl < self.max_kl * 1.5 and new_loss < old_loss: # Minimize loss (negative reward)
                trpo_success = True
                break
        
        if not trpo_success:
            set_params(self.policy.parameters(), old_params)

        # Value Function Update
        # Simple MSE Regression
        for _ in range(10):
            values_pred = self.value_net(obs_batch).squeeze()
            val_loss = nn.MSELoss()(values_pred, returns)
            self.optimizer_val.zero_grad()
            val_loss.backward()
            self.optimizer_val.step()
            
        return trpo_success, old_loss
