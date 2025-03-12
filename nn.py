import torch
from torch import nn
from torch import optim
from collections import deque
import random
import pickle
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(DQN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, 64)  # Input: state size, e.g., 6
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)  # Output: 3 actions
        self.relu = nn.ReLU()
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, load_model):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DQN(state_size, action_size, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        if load_model:
            self.load_model()
            self.epsilon = self.epsilon_min  # Reduce exploration if loading trained model

    def save_model(self, filename="models/snake_dqn.pth", memory_filename="memory/snake_memory.pkl"):
        torch.save(self.model.state_dict(), filename)
        with open(memory_filename, 'wb') as f:
            pickle.dump(self.memory, f)
        print(f"Model saved to {filename}, Memory saved to {memory_filename}")

    def load_model(self, filename="models/snake_dqn.pth", memory_filename="memory/snake_memory.pkl"):
        try:
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}, starting fresh")
        try:
            with open(memory_filename, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Memory loaded from {memory_filename}")
        except FileNotFoundError:
            print(f"No saved memory found at {memory_filename}, starting with empty memory")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:  # Explore
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Move to GPU
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert lists to single NumPy arrays first, then to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-learning update
        targets = rewards + self.gamma * self.model(next_states).max(1)[0] * (1 - dones)
        target_f = self.model(states)
        target_f[range(batch_size), actions] = targets

        self.optimizer.zero_grad()
        output = self.model(states)
        loss = self.criterion(output, target_f)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
