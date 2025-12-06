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
        
        # Internal Replay Buffer for online training
        self.buffer = ReplayBuffer(capacity=10000)
        
    def select_action(self, frame_stack):
        """
        frame_stack: np.array [12, 128, 128] (Channel-first, float 0-1)
        Returns: action_idx (int)
        """
        with torch.no_grad():
            # Normalize if needed (assume 0-255 uint8 -> 0-1 float)
            if frame_stack.max() > 1.0:
                frame_stack = frame_stack.astype(np.float32) / 255.0
            
            x = torch.FloatTensor(frame_stack).unsqueeze(0).to(self.device)  # [1, C, H, W]
            
            # Forward pass
            policy, value, self.hidden_state, latent = self.network(x, self.hidden_state)
            
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done, logprob):
        """
        Store a transition in the internal replay buffer.
        """
        self.buffer.push(state, action, reward, next_state, done, logprob)

    def update_trajectory(self, memory, chunk_size=32):
        """
        Updates the agent on a full trajectory using Truncated BPTT.
        Chunks the trajectory to avoid OOM.
        memory: list of (frames, action, reward, next_frames, done, logprob)
        """
        # Prepare Data
        # We process as Batch=1, Seq=chunk_size
        
        # Unpack all data first (Mutes OOM if we load 1024 tensors at once? 
        # 1024x12x128x128 floats is ~768MB. Fit in RAM/VRAM fine if just tensors.
        # The gradients are the issue.)
        
        frames_all = [m[0] for m in memory]
        actions_all = [m[1] for m in memory]
        rewards_all = [m[2] for m in memory]
        dones_all = [m[4] for m in memory]
        old_probs_all = [m[5] for m in memory]
        
        # Calculate Returns (GAE or Discounted) for WHOLE trajectory first
        # This gives better Value targets than chunked returns
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards_all), reversed(dones_all)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Convert actions/probs to tensors
        actions_all = torch.tensor(actions_all).to(self.device)
        old_probs_all = torch.tensor(old_probs_all).to(self.device)
        
        total_loss = 0
        p_loss_avg = 0
        v_loss_avg = 0
        d_loss_avg = 0
        chunks = 0
        
        # Reset Hidden State for Trajectory
        hidden_state = None
        
        for i in range(0, len(memory), chunk_size):
            # Define Chunk
            end_i = min(i + chunk_size, len(memory))
            current_len = end_i - i
            
            # Prepare Inputs [1, Seq, C, H, W]
            chunk_frames = torch.stack([torch.FloatTensor(f) for f in frames_all[i:end_i]]).to(self.device)
            chunk_frames = chunk_frames.unsqueeze(0) # Batch=1
            
            # Prepare Targets
            chunk_actions = actions_all[i:end_i] # [Seq]
            chunk_old_probs = old_probs_all[i:end_i]
            chunk_returns = returns[i:end_i]
            
            # Forward Pass (with hidden state carryover)
            # network returns: policy [1, Seq, A], value [1, Seq, 1], hidden, latents [1, Seq, Latent]
            policy, values, hidden_state, latents = self.network(chunk_frames, hidden_state)
            
            # Detach hidden state for next truncated BPTT step
            # LSTM hidden is tuple (h, c)
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            
            # Squeeze Batch Dim [Seq, ...]
            policy = policy.squeeze(0)
            values = values.squeeze(0)
            latents = latents.squeeze(0)
            
            # --- Dynamics Loss ---
            # Predict latent[t+1]
            # Requires latent[t+1] target. 
            # If we are at end of chunk, we need the NEXT frame's latent from the Next chunk?
            # Or we just lose 1 step of dynamics training per chunk boundary.
            # Simpler: Slice 0..-1 for Input, 1..End for Target within the chunk.
            
            if current_len > 1:
                curr_latents = latents[:-1]
                target_latents = latents[1:].detach() # Self-supervised from encoder
                curr_actions = chunk_actions[:-1]
                
                actions_oh = F.one_hot(curr_actions, num_classes=self.action_dim).float()
                pred_latents = self.network.dynamics(curr_latents, actions_oh)
                
                dynamics_loss = F.mse_loss(pred_latents, target_latents)
            else:
                dynamics_loss = torch.tensor(0.0).to(self.device)
            
            # --- PPO Loss ---
            # Align lengths (Dynamics loss ignores last step, PPO uses all or alignment?)
            # Usually PPO uses all steps.
            # We can aggregate losses.
            
            dist = torch.distributions.Categorical(policy)
            new_logprobs = dist.log_prob(chunk_actions)
            dist_entropy = dist.entropy().mean()
            
            ratios = torch.exp(new_logprobs - chunk_old_probs)
            advantages = chunk_returns - values.detach().squeeze()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), chunk_returns)
            
            loss = policy_loss + (self.c1 * value_loss) + (self.c2 * dynamics_loss) - (self.c3 * dist_entropy)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            p_loss_avg += policy_loss.item()
            v_loss_avg += value_loss.item()
            d_loss_avg += dynamics_loss.item()
            chunks += 1
            
            # Cleanup to help VRAM
            del chunk_frames, policy, values, latents, loss
            
        return total_loss / chunks, p_loss_avg / chunks, v_loss_avg / chunks, d_loss_avg / chunks

    def update(self, memory):
        # Default update (e.g. from random buffer)
        # Treats memory as batch of independent samples (Seq=1)
        # ... (Existing implementation kept for buffer usage compatibility if needed)
        # But wait, update_with_buffer calls this.
        # Random buffer samples are NOT trajectories. They are (s,a,r,s',d).
        # We need next_state for dynamics target there.
        
        # Recover 'update' for Buffer Sampling (random transitions)
        states = torch.stack([torch.FloatTensor(m[0]) for m in memory]).to(self.device)
        actions = torch.tensor([m[1] for m in memory]).to(self.device)
        rewards = torch.tensor([m[2] for m in memory], dtype=torch.float32).to(self.device)
        dones = torch.tensor([m[4] for m in memory], dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor([m[5] for m in memory]).to(self.device)
        
        # Next States for Value/Dynamics Targets (since random sample)
        next_states = torch.stack([torch.FloatTensor(m[3]) for m in memory]).to(self.device)
        
        with torch.no_grad():
             # Target Value V(s')
             next_seq = next_states.unsqueeze(1)
             _, next_vals, _, _ = self.network(next_seq)
             next_vals = next_vals.squeeze()
             
             # Target Latent z(s')
             # Encoder only for dynamics target
             # Network forward gives latents too.
             # We can run network on next_states to get both
             _, _, _, next_latents = self.network(next_seq)
             target_latents = next_latents.squeeze()
             
        # Returns
        returns = rewards + self.gamma * next_vals * (1 - dones)
        
        # Forward [B, 1, ...]
        states_seq = states.unsqueeze(1)
        policy, values, _, latents = self.network(states_seq)
        
        policy = policy.squeeze(1)
        values = values.squeeze(1)
        latents = latents.squeeze(1)
        
        # Dynamics
        actions_oh = F.one_hot(actions, num_classes=self.action_dim).float()
        pred_latents = self.network.dynamics(latents, actions_oh)
        dynamics_loss = F.mse_loss(pred_latents, target_latents)
        
        # PPO
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

    def update_with_buffer(self, batch_size=64):
        """
        Update using the internal replay buffer.
        """
        if len(self.buffer) < batch_size:
            return 0, 0, 0, 0
            
        memory = self.buffer.sample(batch_size)
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

