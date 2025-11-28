import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared features
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Actor (Policy)
        self.actor = nn.Linear(256, action_dim)
        
        # Critic (Value)
        self.critic = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor: Logits for categorical distribution
        action_logits = self.actor(x)
        
        # Critic: Value estimate
        state_value = self.critic(x)
        
        return action_logits, state_value
