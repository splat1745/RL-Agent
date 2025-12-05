import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .muzero_network import MuZeroNetwork

class MuZeroAgent:
    def __init__(self, action_dim=26, lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.action_dim = action_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Network
        self.network = MuZeroNetwork(action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4)
        
        # Loss Config
        self.c1 = 0.5   # Value Loss weight
        self.c2 = 0.05  # Dynamics Loss weight
        self.c3 = 0.01  # Entropy weight
        
        self.hidden_state = None
        
    def select_action(self, frame_stack):
        """
        frame_stack: np.array [12, 128, 128] (Channel-first, float 0-1)
        """
        with torch.no_grad():
            x = torch.FloatTensor(frame_stack).unsqueeze(0).to(self.device) # [1, C, H, W]
            
            # Add Sequence dim implicitly handled in forward
            policy, value, self.hidden_state, latent = self.network(x, self.hidden_state)
            
            dist = torch.distributions.Categorical(policy)
        return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, memory):
        # ... (Existing update logic tailored for batch) ...
        # If memory is a ReplayBuffer sample, it is already a list of tuples
        
        # Unpack indices
        states = torch.stack([torch.FloatTensor(m[0]) for m in memory]).to(self.device) # [B, C, H, W]
        actions = torch.tensor([m[1] for m in memory]).to(self.device)
        rewards = torch.tensor([m[2] for m in memory], dtype=torch.float32).to(self.device)
        # next_states = torch.stack([torch.FloatTensor(m[3]) for m in memory]).to(self.device) 
        old_logprobs = torch.tensor([m[5] for m in memory]).to(self.device)
        
        # Calculate Advantages (GAE or just Discounted)
        # Warning: Random sampling breaks temporal advantage calculation if not careful.
        # But Phase C implies "Replay Buffer".
        # If sampling random transitions, GAE is hard. We rely on Value Target bootstrapping.
        # R = r + gamma * V(s')
        # We need V(s') for that. We can run network on next_states.
        
        with torch.no_grad():
             next_states_seq = torch.stack([torch.FloatTensor(m[3]) for m in memory]).to(self.device).unsqueeze(1)
             _, next_values, _, _ = self.network(next_states_seq)
             next_values = next_values.squeeze()
             
        # Target Value
        # done flag?
        dones = torch.tensor([m[4] for m in memory], dtype=torch.float32).to(self.device)
        returns = rewards + self.gamma * next_values * (1 - dones)
        
        # Normalize returns?
        # returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Add Sequence Dimension [B, 1, C, H, W]
        states_seq = states.unsqueeze(1)
        
        # Forward Pass
        policy, values, _, latents = self.network(states_seq)
        
        policy = policy.squeeze(1)
        values = values.squeeze(1)
        latents = latents.squeeze(1)
        
        # --- Dynamics Loss ---
        # With random replay, we can't easily predict "next" in sequence unless we stored it.
        # MuZero Dynamics predicts s[t+1] latent.
        # We need the target latent for s[t+1].
        # We can run encoder on s[t+1] to get target.
        with torch.no_grad():
             next_latents_target = self.network.encoder(torch.stack([torch.FloatTensor(m[3]) for m in memory]).to(self.device))
             
        actions_one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
        pred_next_latents = self.network.dynamics(latents, actions_one_hot)
        dynamics_loss = F.mse_loss(pred_next_latents, next_latents_target)
        
        # --- Policy & Value Loss ---
        dist = torch.distributions.Categorical(policy)
        new_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()
        
        ratios = torch.exp(new_logprobs - old_logprobs)
        advantages = returns - values.detach().squeeze()
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        loss = policy_loss + (self.c1 * value_loss) + (self.c2 * dynamics_loss) - (self.c3 * dist_entropy)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item(), dynamics_loss.item()

    def update_with_buffer(self, buffer, batch_size=64):
        if len(buffer) < batch_size:
            return 0,0,0,0
            
        memory = buffer.sample(batch_size)
        return self.update(memory)

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        
    def push(self, frame, action, reward, next_frame, done, logprob):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (frame, action, reward, next_frame, done, logprob)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

