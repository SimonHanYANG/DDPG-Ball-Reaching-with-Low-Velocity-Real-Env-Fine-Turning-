# utils/replay_buffer.py
import numpy as np
import torch
from typing import Tuple

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device: torch.device):
        """
        初始化
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_size: 最大容量
            device: 设备
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样批次"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return self.size