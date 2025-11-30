import torch
import torch.optim as optim
import numpy as np
from .network import TwoStreamNetwork

class PPOAgent:
    def __init__(self, action_dim, lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize TwoStreamNetwork
        self.policy = TwoStreamNetwork(action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = TwoStreamNetwork(action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = torch.nn.MSELoss()
        
        # Hidden state management for LSTM
        self.hidden_state = None
        
    def reset_hidden(self):
        self.hidden_state = None

    def select_action(self, obs_dict):
        """
        obs_dict: {
            'full': np.array [12, 160, 160],
            'crop': np.array [12, 128, 128],
            'flow': np.array [2, 160, 160]
        }
        """
        with torch.no_grad():
            # Convert to tensors and add batch dim
            full = torch.FloatTensor(obs_dict['full']).unsqueeze(0).to(self.device)
            crop = torch.FloatTensor(obs_dict['crop']).unsqueeze(0).to(self.device)
            flow = torch.FloatTensor(obs_dict['flow']).unsqueeze(0).to(self.device)
            
            # Forward pass
            probs, _, self.hidden_state = self.policy_old(full, crop, flow, self.hidden_state)
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item()

    def update(self, memory):
        # Convert memory to tensors
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Process Observations from Memory
        # memory.states is a list of dicts
        full_batch = torch.stack([torch.FloatTensor(s['full']) for s in memory.states]).to(self.device)
        crop_batch = torch.stack([torch.FloatTensor(s['crop']) for s in memory.states]).to(self.device)
        flow_batch = torch.stack([torch.FloatTensor(s['flow']) for s in memory.states]).to(self.device)
        
        old_actions = torch.stack([torch.tensor(a).to(self.device) for a in memory.actions]).detach()
        old_logprobs = torch.stack([torch.tensor(l).to(self.device) for l in memory.logprobs]).detach()
        
        # For update, we process the entire memory as a single sequence (Batch=1, Seq=N)
        # This allows the LSTM to learn temporal dependencies over the full trajectory.
        seq_len = len(memory.states)
        
        for _ in range(self.k_epochs):
            # Forward pass with fresh hidden state (or we could try to preserve it, but fresh is safer for stability)
            # We pass seq_len to tell the network to reshape inputs
            probs, state_values, _ = self.policy(full_batch, crop_batch, flow_batch, seq_len=seq_len)
            
            dist = torch.distributions.Categorical(probs)
            
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(action_logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.1*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
        
    def load(self, filename):
        saved_state = torch.load(filename)
        current_state = self.policy.state_dict()
        
        filtered_state = {k: v for k, v in saved_state.items() if k in current_state and v.shape == current_state[k].shape}
        
        if len(filtered_state) != len(current_state):
            print("Warning: Loaded model has mismatched layers. Initializing new layers randomly.")
        
        current_state.update(filtered_state)
        self.policy.load_state_dict(current_state)
        self.policy_old.load_state_dict(self.policy.state_dict())

