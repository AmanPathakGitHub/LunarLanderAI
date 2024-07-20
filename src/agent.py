
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import gym

import pygame

import random
import itertools

from model import LanderModel
from replay_memory import Replay_Memory


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    
    def __init__(self, hyperparameters, model_path):
        
        self.env = gym.make("LunarLander-v2", render_mode="human")
        self.hyperparameters = hyperparameters
        
        self.model = LanderModel(self.hyperparameters["hidden_nodes"]).to(device)
        
        if model_path != None:
            self.model.load_state_dict(torch.load(model_path))
        
        self.target_model = LanderModel(self.hyperparameters["hidden_nodes"]).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.epsilon = self.hyperparameters["epsilon_init"]
        
        self.replay_memory = Replay_Memory(self.hyperparameters["replay_memory_size"])
        self.step_count = 0
        
        self.loss_fn = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        
        self.writer = SummaryWriter()
        
        
    def run(self, training):
        
        highest_reward = 0
        
        if not training:
            self.model.eval()
        
        for episode in itertools.count():
            state, _ = self.env.reset()

            terminated = False
            total_reward = 0
            
          


            while not terminated:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        torch.save(self.model.state_dict(), "models/best.pth")
                        exit(0)
                
                state = torch.tensor(state, dtype=torch.float32).to(device)

                
                if training and random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = self.model(state).argmax()
                
                next_state, reward, terminated, _, info  = self.env.step(action.item())
                
                total_reward += reward
                
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)
                
                state = next_state
                
                # train step
                if training:
                    self.train_step(state, action, next_state, reward, terminated)
                    
                    if self.step_count > self.hyperparameters["max_steps"]:
                        break
                
                if not training:
                    self.env.render()
                
                
            
            
            if training:
                mini_batch = self.replay_memory.sample(self.hyperparameters["batch_size"])
                self.optimise(mini_batch)
            
                self.epsilon = max(self.epsilon * self.hyperparameters["epsilon_decay"], self.hyperparameters["epsilon_min"])
                print(f"Episode: {episode}, Epsilon: {self.epsilon}, Highest Reward: {highest_reward}")
                
                                
                self.writer.add_scalar("Epsilon", self.epsilon, episode)
                self.writer.add_scalar("Reward", total_reward, episode)
                
                if total_reward > highest_reward:
                    highest_reward = total_reward
                    torch.save(self.model.state_dict(), "models/best.pth")
                    
                self.step_count = 0
                    
                   
            
            
        self.env.close()
    
    def train_step(self, state, action, next_state, reward, terminated):
        self.replay_memory.append((state, action, next_state, reward, terminated))
        
        self.step_count += 1
        
        if self.step_count % self.hyperparameters["network_sync_rate"] == 0:
            
            self.target_model.load_state_dict(self.model.state_dict())
        

        self.optimise([(state, action, next_state, reward, terminated)])
        
        
    
    def optimise(self, mini_batch):
        
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.int64).to(device)
        
        current_q = self.model(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.hyperparameters["discount_factor"] * self.target_model(next_states).max(dim=1)[0]
            
        loss = self.loss_fn(current_q, target_q)
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()