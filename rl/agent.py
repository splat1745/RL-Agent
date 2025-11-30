import torch
import torch.optim as optim
import numpy as np
from .network import TwoStreamNetwork

class PPOAgent:
    def __init__(self, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
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
        self.train_hidden = None # Persist hidden state across updates for temporal memory

    def reset_hidden(self):
        self.hidden_state = None
        self.train_hidden = None

    def select_action(self, pixel_obs, vector_obs):
        """
        pixel_obs: {
            'full': np.array [16, 160, 160],
            'crop': np.array [16, 128, 128],
            'flow': np.array [2, 160, 160]
        }
        vector_obs: np.array [36]
        """
        with torch.no_grad():
            # Convert to tensors and add batch dim
            full = torch.FloatTensor(pixel_obs['full']).unsqueeze(0).to(self.device)
            crop = torch.FloatTensor(pixel_obs['crop']).unsqueeze(0).to(self.device)
            flow = torch.FloatTensor(pixel_obs['flow']).unsqueeze(0).to(self.device)
            vector = torch.FloatTensor(vector_obs).unsqueeze(0).to(self.device)
            
            # Forward pass
            probs, _, self.hidden_state, intention = self.policy_old(full, crop, flow, vector, self.hidden_state)
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item(), intention.cpu().numpy()

    def update(self, memory, hit_history=None):
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
        # memory.states is a list of tuples (pixel_obs, vector_obs)
        full_batch = torch.stack([torch.FloatTensor(s[0]['full']) for s in memory.states]).to(self.device)
        crop_batch = torch.stack([torch.FloatTensor(s[0]['crop']) for s in memory.states]).to(self.device)
        flow_batch = torch.stack([torch.FloatTensor(s[0]['flow']) for s in memory.states]).to(self.device)
        vector_batch = torch.stack([torch.FloatTensor(s[1]) for s in memory.states]).to(self.device)
        
        old_actions = torch.stack([torch.tensor(a).to(self.device) for a in memory.actions]).detach()
        old_logprobs = torch.stack([torch.tensor(l).to(self.device) for l in memory.logprobs]).detach()
        
        # For update, we process the entire memory as a single sequence (Batch=1, Seq=N)
        # This allows the LSTM to learn temporal dependencies over the full trajectory.
        seq_len = len(memory.states)
        
        # Handle Temporal Memory (161 frames context)
        # We use the persisted train_hidden state to maintain context across short updates
        if self.train_hidden is None:
             # Initialize with zeros if first update
             # LSTM expects (num_layers, batch, hidden_dim)
             h0 = torch.zeros(2, 1, 768).to(self.device)
             c0 = torch.zeros(2, 1, 768).to(self.device)
             self.train_hidden = (h0, c0)
        
        # Detach hidden state to prevent backprop through the entire history (Truncated BPTT)
        # But keep the values to provide context
        initial_hidden = (self.train_hidden[0].detach(), self.train_hidden[1].detach())
        
        current_hidden = initial_hidden

        for _ in range(self.k_epochs):
            # Forward pass with persisted hidden state
            # We pass seq_len to tell the network to reshape inputs
            probs, state_values, current_hidden, _ = self.policy(full_batch, crop_batch, flow_batch, vector_batch, hidden_state=initial_hidden, seq_len=seq_len)
            
            dist = torch.distributions.Categorical(probs)
            
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(action_logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Reflective Learning Loss
            # Analyze hit_history to penalize actions that led to hits
            reflective_loss = 0
            if hit_history is not None and len(hit_history) == seq_len:
                # Identify indices where a hit occurred
                hit_indices = [i for i, hit in enumerate(hit_history) if hit]
                if hit_indices:
                    # For these indices, we want to minimize the probability of the taken action
                    # We can do this by maximizing the negative log prob (minimizing log prob)
                    # Or simply adding the probability itself to the loss (minimizing likelihood)
                    
                    # Get probs of the actions that were taken at hit indices
                    hit_probs = probs[hit_indices, old_actions[hit_indices]]
                    
                    # We want to minimize these probabilities
                    reflective_loss = hit_probs.mean() * 0.5 # Weight for reflective loss

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.1*dist_entropy + reflective_loss
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            
            self.optimizer.step()
            
        # Update the train_hidden state for the next batch
        # We take the hidden state from the LAST epoch's forward pass (or just one of them)
        # The hidden state output by LSTM corresponds to the end of the sequence
        self.train_hidden = (current_hidden[0].detach(), current_hidden[1].detach())
            
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

