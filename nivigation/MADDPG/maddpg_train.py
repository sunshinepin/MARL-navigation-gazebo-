import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import copy
from collections import namedtuple, deque
from maddpg_env import GazeboEnv

# 超参数
robot_num =2
environment_dim=20*robot_num
robot_dim = 4
STATE_DIM = 24*robot_num
ACTION_DIM = 2*robot_num
MAX_ACTION = 1.0
CAPACITY = 10000
BATCH_SIZE = 128
EPISODES = 10000
STEPS = 1000
EXPLORATION_NOISE = 0.1
GAMMA = 0.99
TAU = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

class Critic(nn.Module):
    # def __init__(self, state_dim, action_dim):
    #     super(Critic, self).__init__()
    #     self.layer1 = nn.Linear(state_dim + action_dim, 400)
    #     self.layer2 = nn.Linear(400, 300)
    #     self.layer3 = nn.Linear(300, 1)
    def __init__(self, num_agents, state_dim, action_dim):
            super(Critic, self).__init__()
            total_state_dim = state_dim * num_agents
            total_action_dim = action_dim * num_agents
            self.layer1 = nn.Linear(total_state_dim + total_action_dim, 400)
            self.layer2 = nn.Linear(400, 300)
            self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.bool)

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.replay_buffer = ReplayBuffer(CAPACITY)

    def select_action(self, state, noise_scale=EXPLORATION_NOISE):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = noise_scale * np.random.randn(ACTION_DIM)
        action = action + noise
        return np.clip(action, -MAX_ACTION, MAX_ACTION)

    def update(self):
        if len(self.replay_buffer.buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        state = torch.FloatTensor(state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            # Target actions
            next_action = self.actor_target(next_state)
            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + ((1 - done) * GAMMA * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def train():
    env = GazeboEnv(environment_dim=environment_dim)
    agent = MADDPGAgent(STATE_DIM * 2, ACTION_DIM * 2, MAX_ACTION)  # Adjusting dimensions for all agents combined
    for episode in range(EPISODES):
        states = env.reset()  # Get initial states for all agents
        for step in range(STEPS):
            actions = [agent.select_action(state) for state in states]
            next_states, rewards, dones, _ = env.step(actions)
            for i in range(len(states)):
                agent.replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            states = next_states
            if any(dones):
                break
        agent.update()

train()  # Start training


