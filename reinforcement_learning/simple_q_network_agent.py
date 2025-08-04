import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleQNetwork(nn.Module):
    """A simple Q-Network for Reinforcement Learning using deep learning."""
    def __init__(self, input_dim, output_dim):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReinforcementLearningAgent:
    """Agent that interacts with the environment and learns using Q-learning."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = SimpleQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.q_network(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += 0.99 * torch.max(self.q_network(torch.FloatTensor(next_state))).item()
        target_f = self.q_network(torch.FloatTensor(state))
        target_f[action] = target
        self.optimizer.zero_grad()
        loss = self.criterion(self.q_network(torch.FloatTensor(state)), target_f)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Simulated environment interaction
state_size = 4
action_size = 2
agent = ReinforcementLearningAgent(state_size, action_size)

# Simulate training for 10 episodes
for episode in range(10):
    state = np.random.rand(state_size)
    for time in range(100):
        action = agent.act(state)
        next_state = np.random.rand(state_size)
        reward = 1 if action == 1 else 0
        done = time == 99
        agent.train(state, action, reward, next_state, done)
        state = next_state
    print(f'Episode {episode + 1} finished')