# models/ddpg_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.real_config import RealConfig as Config

class Actor(nn.Module):
    """Actor网络 - 确定性策略"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        
        # 初始化权重
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


class Critic(nn.Module):
    """Critic网络 - Q函数"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Critic, self).__init__()
        
        # Q网络
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class OUNoise:
    """Ornstein-Uhlenbeck噪声（自适应）"""
    
    def __init__(self, size: int, mu: float, theta: float, sigma: float):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.current_sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.current_sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state
    
    def decay_sigma(self, decay_rate: float):
        """衰减噪声"""
        self.current_sigma *= decay_rate


class DDPGAgent:
    """DDPG智能体（支持渐进式解冻和学习率调度）"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        """
        初始化
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置对象
        """
        self.device = config.DEVICE
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.max_action = config.MAX_ACTION
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim, config.HIDDEN_DIM, config.MAX_ACTION).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, config.HIDDEN_DIM, config.MAX_ACTION).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        
        # Critic网络
        self.critic = Critic(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)
        
        # OU噪声（自适应）
        self.noise = OUNoise(action_dim, config.OU_MU, config.OU_THETA, config.OU_SIGMA_START)
        
        # 学习率调度器（移除 verbose 参数）
        if config.LR_SCHEDULER_ENABLED:
            self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer,
                mode='max',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                min_lr=config.LR_SCHEDULER_MIN_LR_ACTOR
            )
            
            self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer,
                mode='max',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                min_lr=config.LR_SCHEDULER_MIN_LR_CRITIC
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None
    
    def freeze_all_layers(self):
        """冻结所有层"""
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: list = None):
        """解冻指定层
        
        Args:
            layer_names: 要解冻的层名称列表，None表示解冻所有层
        """
        if layer_names is None:
            # 解冻所有层
            for param in self.actor.parameters():
                param.requires_grad = True
            for param in self.critic.parameters():
                param.requires_grad = True
        else:
            # 先冻结所有层
            self.freeze_all_layers()
            
            # 解冻指定层
            for name in layer_names:
                layer_name = name.split('.')[0]
                
                if hasattr(self.actor, layer_name):
                    layer = getattr(self.actor, layer_name)
                    for param in layer.parameters():
                        param.requires_grad = True
                
                if hasattr(self.critic, layer_name):
                    layer = getattr(self.critic, layer_name)
                    for param in layer.parameters():
                        param.requires_grad = True
        
        # 重新创建优化器以确保只优化需要梯度的参数
        self._recreate_optimizers()
    
    def _recreate_optimizers(self):
        """重新创建优化器（只包含需要梯度的参数）"""
        # 获取当前学习率
        current_actor_lr = self.actor_optimizer.param_groups[0]['lr']
        current_critic_lr = self.critic_optimizer.param_groups[0]['lr']
        
        # 只选择需要梯度的参数
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        
        # 如果没有可训练参数，使用一个dummy参数避免错误
        if len(actor_params) == 0:
            print("Warning: No trainable actor parameters, using all parameters")
            actor_params = list(self.actor.parameters())
        
        if len(critic_params) == 0:
            print("Warning: No trainable critic parameters, using all parameters")
            critic_params = list(self.critic.parameters())
        
        # 重新创建优化器
        self.actor_optimizer = optim.Adam(actor_params, lr=current_actor_lr)
        self.critic_optimizer = optim.Adam(critic_params, lr=current_critic_lr)
        
        # 重新创建学习率调度器
        if self.actor_scheduler is not None:
            self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer,
                mode='max',
                factor=0.5,
                patience=200,
                min_lr=1e-6
            )
        
        if self.critic_scheduler is not None:
            self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer,
                mode='max',
                factor=0.5,
                patience=200,
                min_lr=3e-6
            )
    
    def set_learning_rate(self, lr_actor: float, lr_critic: float):
        """设置学习率"""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr_actor
        
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr_critic
    
    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if not evaluate:
            # 添加OU噪声用于探索
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def update(self, replay_buffer, batch_size: int, use_prioritized: bool = False, beta: float = 0.4):
        """更新网络（支持优先级回放）"""
        if use_prioritized:
            states, actions, rewards, next_states, dones, is_weights, indices = replay_buffer.sample(batch_size, beta)
        else:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            is_weights = torch.ones((batch_size, 1)).to(self.device)
            indices = None
        
        # 检查是否有可训练的参数
        critic_trainable_params = [p for p in self.critic.parameters() if p.requires_grad]
        actor_trainable_params = [p for p in self.actor.parameters() if p.requires_grad]
        
        # 更新Critic
        if len(critic_trainable_params) > 0:
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q = self.critic_target(next_states, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            current_q = self.critic(states, actions)
            
            # 计算TD误差（用于更新优先级）
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy().flatten()
            
            # 加权MSE损失
            critic_loss = (is_weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_trainable_params, 1.0)
            self.critic_optimizer.step()
            
            critic_loss_value = critic_loss.item()
        else:
            # 如果critic被冻结，仍然计算TD误差用于优先级更新
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_q = self.critic_target(next_states, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * target_q
                current_q = self.critic(states, actions)
                td_errors = torch.abs(current_q - target_q).cpu().numpy().flatten()
            
            critic_loss_value = 0.0
        
        # 更新Actor
        if len(actor_trainable_params) > 0:
            actor_loss = -(self.critic(states, self.actor(states)) * is_weights).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_trainable_params, 1.0)
            self.actor_optimizer.step()
            
            actor_loss_value = actor_loss.item()
        else:
            actor_loss_value = 0.0
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # 更新优先级
        if use_prioritized and indices is not None:
            replay_buffer.update_priorities(indices, td_errors)
        
        # 计算当前Q值（用于监控）
        with torch.no_grad():
            current_q_mean = self.critic(states, actions).mean().item()
        
        return {
            'critic_loss': critic_loss_value,
            'actor_loss': actor_loss_value,
            'q_value': current_q_mean,
            'td_error': td_errors.mean()
        }
    
    def step_scheduler(self, metric: float):
        """步进学习率调度器"""
        if self.actor_scheduler is not None:
            old_actor_lr = self.actor_optimizer.param_groups[0]['lr']
            self.actor_scheduler.step(metric)
            new_actor_lr = self.actor_optimizer.param_groups[0]['lr']
            if old_actor_lr != new_actor_lr:
                print(f"Actor LR reduced: {old_actor_lr:.2e} -> {new_actor_lr:.2e}")
        
        if self.critic_scheduler is not None:
            old_critic_lr = self.critic_optimizer.param_groups[0]['lr']
            self.critic_scheduler.step(metric)
            new_critic_lr = self.critic_optimizer.param_groups[0]['lr']
            if old_critic_lr != new_critic_lr:
                print(f"Critic LR reduced: {old_critic_lr:.2e} -> {new_critic_lr:.2e}")
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_sigma': self.noise.current_sigma
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if 'noise_sigma' in checkpoint:
            self.noise.current_sigma = checkpoint['noise_sigma']