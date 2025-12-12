# utils/prioritized_replay_buffer.py
import numpy as np
import torch
from typing import Tuple
import random

class SumTree:
    """求和树数据结构用于优先级采样"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float):
        """根据累积和检索叶节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """返回总优先级"""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """添加数据"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float):
        """根据累积和获取数据"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int, 
                 device: torch.device, alpha: float = 0.6, epsilon: float = 1e-6):
        """
        初始化
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_size: 最大容量
            device: 设备
            alpha: 优先级指数
            epsilon: 防止零优先级
        """
        self.max_size = max_size
        self.device = device
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.tree = SumTree(max_size)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """添加经验（使用最大优先级）"""
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """采样批次（基于优先级）"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, priority, data) = self.tree.get(s)
            
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        
        # 解包经验
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch]).reshape(-1, 1)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights.reshape(-1, 1)).to(self.device)
        
        return states, actions, rewards, next_states, dones, is_weights, indices
    
    def update_priorities(self, indices: list, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries


class HybridReplayBuffer:
    """混合回放缓冲区：结合优先级回放和近期经验优先"""
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int,
                 recent_size: int, device: torch.device, 
                 alpha: float = 0.6, recent_ratio: float = 0.5):
        """
        初始化
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_size: 主缓冲区大小
            recent_size: 近期缓冲区大小
            device: 设备
            alpha: 优先级指数
            recent_ratio: 从近期缓冲区采样的比例
        """
        self.prioritized_buffer = PrioritizedReplayBuffer(
            state_dim, action_dim, max_size, device, alpha
        )
        
        self.recent_buffer = PrioritizedReplayBuffer(
            state_dim, action_dim, recent_size, device, alpha
        )
        
        self.recent_ratio = recent_ratio
        self.device = device
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """同时添加到两个缓冲区"""
        self.prioritized_buffer.add(state, action, reward, next_state, done)
        self.recent_buffer.add(state, action, reward, next_state, done)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """混合采样"""
        recent_batch_size = int(batch_size * self.recent_ratio)
        prioritized_batch_size = batch_size - recent_batch_size
        
        # 从近期缓冲区采样
        if len(self.recent_buffer) >= recent_batch_size:
            recent_samples = self.recent_buffer.sample(recent_batch_size, beta)
        else:
            recent_samples = None
        
        # 从优先级缓冲区采样
        if len(self.prioritized_buffer) >= prioritized_batch_size:
            prioritized_samples = self.prioritized_buffer.sample(prioritized_batch_size, beta)
        else:
            prioritized_samples = None
        
        # 合并样本
        if recent_samples is None and prioritized_samples is None:
            raise ValueError("Not enough samples in buffers")
        elif recent_samples is None:
            return prioritized_samples
        elif prioritized_samples is None:
            return recent_samples
        else:
            # 合并所有张量
            states = torch.cat([recent_samples[0], prioritized_samples[0]], dim=0)
            actions = torch.cat([recent_samples[1], prioritized_samples[1]], dim=0)
            rewards = torch.cat([recent_samples[2], prioritized_samples[2]], dim=0)
            next_states = torch.cat([recent_samples[3], prioritized_samples[3]], dim=0)
            dones = torch.cat([recent_samples[4], prioritized_samples[4]], dim=0)
            is_weights = torch.cat([recent_samples[5], prioritized_samples[5]], dim=0)
            indices = recent_samples[6] + prioritized_samples[6]
            
            return states, actions, rewards, next_states, dones, is_weights, indices
    
    def update_priorities(self, indices: list, priorities: np.ndarray):
        """更新两个缓冲区的优先级"""
        # 分离近期和优先级缓冲区的索引
        recent_count = int(len(indices) * self.recent_ratio)
        
        recent_indices = indices[:recent_count]
        recent_priorities = priorities[:recent_count]
        
        prioritized_indices = indices[recent_count:]
        prioritized_priorities = priorities[recent_count:]
        
        if len(recent_indices) > 0:
            self.recent_buffer.update_priorities(recent_indices, recent_priorities)
        
        if len(prioritized_indices) > 0:
            self.prioritized_buffer.update_priorities(prioritized_indices, prioritized_priorities)
    
    def __len__(self) -> int:
        return len(self.prioritized_buffer)
    