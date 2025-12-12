# real_training_thread.py
import os
import sys
import numpy as np
import torch
import time
import threading
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.real_config import RealConfig
from envs.real_ball_env import RealBallEnv
from models.ddpg_agent import DDPGAgent
from utils.prioritized_replay_buffer import HybridReplayBuffer
from utils.logger import Logger

class EarlyStopping:
    """早停控制器"""
    
    def __init__(self, patience: int = 500, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """检查是否应该早停
        
        Args:
            score: 当前评分（成功率）
        
        Returns:
            是否应该停止
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False


class RealTrainingThread(QThread):
    """真实环境训练线程（集成渐进式Fine-tuning）"""
    
    update_signal = pyqtSignal(dict)
    frame_signal = pyqtSignal(np.ndarray)
    env_ready_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)
    stage_complete_signal = pyqtSignal(int)
    
    def __init__(self, mode='teacher', total_episodes=20000, 
                 curriculum_config=None, pretrained_model=None):
        super().__init__()
        self.mode = mode
        self.total_episodes = total_episodes
        self.curriculum_config = curriculum_config
        self.pretrained_model = pretrained_model
        self.running = False
        self.paused = False
        self.logger = None
        self.env = None
    
    def calculate_success_rate(self, recent_successes, window=100):
        """计算成功率"""
        if len(recent_successes) < window:
            return np.mean(recent_successes) if recent_successes else 0.0
        return np.mean(recent_successes[-window:])
    
    def check_stage_completion(self, current_stage, stage_episode_count, success_rate, 
                              recent_successes, patience_counter, config):
        """检查当前阶段是否完成"""
        # 最少episode要求
        if stage_episode_count < current_stage['min_episodes']:
            return False, 0
        
        # 成功率窗口要求
        if len(recent_successes) < config.CURRICULUM_SUCCESS_WINDOW:
            return False, 0
        
        # 成功率阈值和耐心计数
        if success_rate >= current_stage['success_threshold']:
            patience_counter += 1
            
            if patience_counter >= current_stage['patience']:
                return True, patience_counter
        else:
            patience_counter = 0
        
        # 最大episode限制
        if stage_episode_count >= current_stage['max_episodes']:
            return True, patience_counter
        
        return False, patience_counter
    
    def run(self):
        """运行训练"""
        self.running = True
        
        # 配置
        config = RealConfig()
        
        # 使用传入的课程配置
        if self.curriculum_config is not None:
            config.CURRICULUM_STAGES = self.curriculum_config
            config.CURRICULUM_ENABLED = True
        
        # 创建日志目录
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        # 检查断点
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{self.mode}_latest.pth')
        info_path = checkpoint_path.replace('.pth', '_info.pth')
        start_episode = 0
        total_timesteps = 0
        current_curriculum_idx = 0
        stage_episode_count = 0
        patience_counter = 0
        
        # 初始化日志
        experiment_name = f'{self.mode}_training_real_progressive'
        
        if os.path.exists(checkpoint_path) and os.path.exists(info_path):
            try:
                checkpoint_info = torch.load(info_path, map_location=config.DEVICE, weights_only=False)
                start_episode = checkpoint_info.get('episode', 0)
                total_timesteps = checkpoint_info.get('timesteps', 0)
                current_curriculum_idx = checkpoint_info.get('curriculum_idx', 0)
                stage_episode_count = checkpoint_info.get('stage_episode_count', 0)
                patience_counter = checkpoint_info.get('patience_counter', 0)
                
                self.logger = Logger(config.LOG_DIR, experiment_name)
                log_msg = f"{'='*60}"
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
                log_msg = f"RESUMING {self.mode.upper()} TRAINING (PROGRESSIVE FINE-TUNING)"
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
                log_msg = f"{'='*60}"
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
                log_msg = f"Resume from episode {start_episode}, Stage {current_curriculum_idx + 1}"
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
            except Exception as e:
                self.logger = Logger(config.LOG_DIR, experiment_name)
                log_msg = f"Failed to load checkpoint: {e}. Starting new training."
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
                start_episode = 0
                total_timesteps = 0
        else:
            self.logger = Logger(config.LOG_DIR, experiment_name)
            log_msg = f"{'='*60}"
            self.logger.log_message(log_msg)
            self.log_signal.emit(log_msg)
            log_msg = f"STARTING NEW {self.mode.upper()} TRAINING (PROGRESSIVE FINE-TUNING)"
            self.logger.log_message(log_msg)
            self.log_signal.emit(log_msg)
            log_msg = f"{'='*60}"
            self.logger.log_message(log_msg)
            self.log_signal.emit(log_msg)
        
        self.logger.log_message(f"Device: {config.DEVICE}")
        self.logger.log_message(f"Total Episodes Limit: {self.total_episodes}")
        self.logger.log_message(f"Prioritized Replay: {config.PRIORITIZED_REPLAY_ENABLED}")
        self.logger.log_message(f"Early Stopping: {config.EARLY_STOPPING_ENABLED}")
        self.logger.log_message(f"LR Scheduler: {config.LR_SCHEDULER_ENABLED}")
        
        # 创建环境
        if config.CURRICULUM_ENABLED and self.mode == 'teacher':
            curriculum_stages = config.CURRICULUM_STAGES
            current_stage = curriculum_stages[current_curriculum_idx]
            log_msg = f"Starting {current_stage['name']}"
            self.logger.log_message(log_msg)
            self.log_signal.emit(log_msg)
            
            self.env = RealBallEnv(config, mode=self.mode, curriculum_stage=current_stage)
        else:
            curriculum_stages = None
            self.env = RealBallEnv(config, mode=self.mode, curriculum_stage=None)

        # 发送环境引用
        self.env_ready_signal.emit(self.env)
        
        # 连接帧更新信号
        self.env.frame_updated.connect(self.frame_signal.emit)
        
        # 初始化Stage
        if not self.env.initialize_stage():
            self.logger.log_message("Failed to initialize Stage!")
            self.log_signal.emit("Failed to initialize Stage!")
            self.finished_signal.emit()
            return
        
        # 创建智能体
        state_dim = config.TEACHER_STATE_DIM if self.mode == 'teacher' else config.STUDENT_STATE_DIM
        agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=config.ACTION_DIM,
            config=config
        )
        
        # 加载预训练模型
        if self.pretrained_model and os.path.exists(self.pretrained_model):
            try:
                agent.load(self.pretrained_model)
                self.logger.log_message(f"Loaded pretrained model from {self.pretrained_model}")
                self.log_signal.emit(f"Loaded pretrained model: {os.path.basename(self.pretrained_model)}")
                
                # 不在这里冻结，等到应用阶段设置时再冻结
                
            except Exception as e:
                self.logger.log_message(f"Failed to load pretrained model: {e}")
                self.log_signal.emit(f"Failed to load pretrained model: {e}")
        elif os.path.exists(checkpoint_path):
            try:
                agent.load(checkpoint_path)
                self.logger.log_message(f"Loaded checkpoint from {checkpoint_path}")
                self.log_signal.emit(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                self.logger.log_message(f"Failed to load checkpoint: {e}")
                self.log_signal.emit(f"Failed to load checkpoint: {e}")
        
        # 应用当前阶段的层解冻和学习率
        if curriculum_stages:
            current_stage = curriculum_stages[current_curriculum_idx]
            
            # 解冻指定层
            if 'unfreeze_layers' in current_stage and current_stage['unfreeze_layers'] is not None:
                agent.unfreeze_layers(current_stage['unfreeze_layers'])
                self.logger.log_message(f"Unfrozen layers: {current_stage['unfreeze_layers']}")
                self.log_signal.emit(f"Unfrozen layers: {current_stage['unfreeze_layers']}")
            else:
                agent.unfreeze_layers(None)
                self.logger.log_message("All layers unfrozen")
                self.log_signal.emit("All layers unfrozen")
            
            # 设置学习率
            if 'lr_actor' in current_stage and 'lr_critic' in current_stage:
                agent.set_learning_rate(current_stage['lr_actor'], current_stage['lr_critic'])
                self.logger.log_message(f"LR: Actor={current_stage['lr_actor']}, Critic={current_stage['lr_critic']}")
                self.log_signal.emit(f"LR: Actor={current_stage['lr_actor']:.0e}, Critic={current_stage['lr_critic']:.0e}")
        
        # 创建混合经验回放缓冲区
        if config.PRIORITIZED_REPLAY_ENABLED:
            replay_buffer = HybridReplayBuffer(
                state_dim, 
                config.ACTION_DIM, 
                config.BUFFER_SIZE,
                config.RECENT_BUFFER_SIZE,
                config.DEVICE,
                config.PRIORITY_ALPHA,
                config.RECENT_SAMPLE_RATIO
            )
            self.logger.log_message("Using Hybrid Prioritized Replay Buffer")
        else:
            from utils.replay_buffer import ReplayBuffer
            replay_buffer = ReplayBuffer(state_dim, config.ACTION_DIM, config.BUFFER_SIZE, config.DEVICE)
            self.logger.log_message("Using Standard Replay Buffer")
        
        # 早停控制器
        if config.EARLY_STOPPING_ENABLED:
            early_stopping = EarlyStopping(
                patience=config.EARLY_STOPPING_PATIENCE,
                min_delta=config.EARLY_STOPPING_MIN_DELTA
            )
            self.logger.log_message(f"Early stopping enabled (patience={config.EARLY_STOPPING_PATIENCE})")
        else:
            early_stopping = None
        
        # 训练循环
        recent_successes = []
        best_success_rate = 0.0
        beta_schedule = np.linspace(
            config.PRIORITY_BETA_START, 
            config.PRIORITY_BETA_END, 
            self.total_episodes
        )
        
        for episode in range(start_episode, self.total_episodes):
            if not self.running:
                break
            
            while self.paused:
                time.sleep(0.1)
                if not self.running:
                    break
            
            # 自动随机初始化
            state = self.env.reset(manual_init=False)
            
            if not self.running:
                break
            
            # 重置状态
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # 重置OU噪声
            agent.noise.reset()
            
            while not done and self.running:
                while self.paused:
                    time.sleep(0.1)
                    if not self.running:
                        break
                
                # 检查Ball是否被检测到
                if not self.env.ball_detected:
                    self.log_signal.emit("Ball lost! Episode terminated.")
                    break
                
                # 选择动作
                if total_timesteps < config.WARMUP_STEPS:
                    action = np.random.uniform(-1, 1, config.ACTION_DIM)
                else:
                    action = agent.select_action(state, evaluate=False)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 检查错误
                if 'error' in info:
                    self.log_signal.emit(f"Error: {info['error']}")
                    break
                
                # 存储经验
                replay_buffer.add(state, action, reward, next_state, done)
                
                # 更新
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_timesteps += 1
                
                # 更新网络
                if len(replay_buffer) > config.BATCH_SIZE and total_timesteps % config.UPDATE_EVERY == 0:
                    beta = beta_schedule[min(episode, len(beta_schedule) - 1)]
                    
                    if config.PRIORITIZED_REPLAY_ENABLED:
                        losses = agent.update(replay_buffer, config.BATCH_SIZE, 
                                             use_prioritized=True, beta=beta)
                    else:
                        losses = agent.update(replay_buffer, config.BATCH_SIZE, 
                                             use_prioritized=False)
                    
                    if total_timesteps % 100 == 0:
                        self.logger.log_losses(episode, total_timesteps, losses)
                else:
                    losses = {}
                
                # 噪声衰减
                if hasattr(config, 'OU_SIGMA_DECAY'):
                    agent.noise.decay_sigma(config.OU_SIGMA_DECAY)
            
            # Episode结束
            success = 1 if info.get('termination') == 'success' else 0
            recent_successes.append(success)
            if len(recent_successes) > config.CURRICULUM_SUCCESS_WINDOW:
                recent_successes.pop(0)
            
            success_rate = self.calculate_success_rate(recent_successes, config.CURRICULUM_SUCCESS_WINDOW)
            
            # 记录episode信息
            info['episode_reward'] = episode_reward
            self.logger.log_episode(episode, total_timesteps, info)
            
            # 发送更新信号
            update_data = {
                'episode': episode,
                'total_episodes': self.total_episodes,
                'reward': episode_reward,
                'success_rate': success_rate,
                'distance': info.get('distance', 0),
                'velocity': info.get('velocity', 0),
                'steps': episode_steps,
                'termination': info.get('termination', 'unknown'),
                'entropy': losses.get('entropy', 0),
                'kl_divergence': losses.get('kl_divergence', 0),
                'curriculum_stage': current_curriculum_idx + 1 if curriculum_stages else 0,
                'stage_episode': stage_episode_count,
                'patience': patience_counter,
                'max_init_distance': current_stage.get('max_init_distance', 0) if curriculum_stages else 0,
                'noise_sigma': agent.noise.current_sigma,
                'actor_lr': agent.actor_optimizer.param_groups[0]['lr'],
                'critic_lr': agent.critic_optimizer.param_groups[0]['lr']
            }
            self.update_signal.emit(update_data)
            
            # 打印进度
            if episode % 10 == 0:
                stage_info = f"[Stage {current_curriculum_idx + 1}]" if curriculum_stages else ""
                log_msg = (
                    f"{stage_info} Episode {episode}/{self.total_episodes} | "
                    f"Steps: {episode_steps} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Success Rate: {success_rate:.2%} | "
                    f"Distance: {info.get('distance', 0):.2f}px | "
                    f"Sigma: {agent.noise.current_sigma:.3f} | "
                    f"Termination: {info.get('termination', 'unknown')}"
                )
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
            
            # 学习率调度
            if config.LR_SCHEDULER_ENABLED and episode % config.EARLY_STOPPING_CHECK_INTERVAL == 0:
                agent.step_scheduler(success_rate)
            
            # 早停检查
            if early_stopping and episode % config.EARLY_STOPPING_CHECK_INTERVAL == 0:
                if early_stopping(success_rate):
                    log_msg = f"Early stopping triggered at episode {episode} (success rate: {success_rate:.2%})"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    break
            
            # 保存checkpoint
            if (episode + 1) % config.SAVE_EVERY == 0:
                save_dict = {
                    'episode': episode + 1,
                    'timesteps': total_timesteps,
                    'success_rate': success_rate,
                    'curriculum_idx': current_curriculum_idx,
                    'stage_episode_count': stage_episode_count,
                    'patience_counter': patience_counter
                }
                
                latest_path = os.path.join(config.CHECKPOINT_DIR, f'{self.mode}_latest.pth')
                agent.save(latest_path)
                torch.save(save_dict, latest_path.replace('.pth', '_info.pth'))
                
                log_msg = f"Saved checkpoint at episode {episode + 1}"
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
            
            # 保存最佳模型
            if success_rate > best_success_rate and len(recent_successes) >= config.CURRICULUM_SUCCESS_WINDOW:
                best_success_rate = success_rate
                best_path = os.path.join(config.CHECKPOINT_DIR, f'{self.mode}_best.pth')
                agent.save(best_path)
                log_msg = f"New best model! Success rate: {best_success_rate:.2%}"
                self.logger.log_message(log_msg)
                self.log_signal.emit(log_msg)
            
            # 课程学习阶段检查
            if config.CURRICULUM_ENABLED and self.mode == 'teacher' and curriculum_stages:
                stage_episode_count += 1
                current_stage = curriculum_stages[current_curriculum_idx]
                
                is_complete, patience_counter = self.check_stage_completion(
                    current_stage,
                    stage_episode_count,
                    success_rate,
                    recent_successes,
                    patience_counter,
                    config
                )
                
                if is_complete and current_curriculum_idx < len(curriculum_stages) - 1:
                    # 保存当前阶段模型
                    stage_path = os.path.join(
                        config.CHECKPOINT_DIR,
                        f'{self.mode}_stage{current_curriculum_idx + 1}.pth'
                    )
                    agent.save(stage_path)
                    
                    log_msg = f"{'='*60}"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    log_msg = f"Stage {current_curriculum_idx + 1} COMPLETED!"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    log_msg = f"{'='*60}"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    
                    self.stage_complete_signal.emit(current_curriculum_idx + 1)
                    
                    # 进入下一阶段
                    current_curriculum_idx += 1
                    stage_episode_count = 0
                    patience_counter = 0
                    recent_successes = []
                    current_stage = curriculum_stages[current_curriculum_idx]
                    
                    log_msg = f"Starting {current_stage['name']}"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    
                    # 渐进式解冻
                    if 'unfreeze_layers' in current_stage and current_stage['unfreeze_layers'] is not None:
                        agent.unfreeze_layers(current_stage['unfreeze_layers'])
                        self.logger.log_message(f"Unfrozen layers: {current_stage['unfreeze_layers']}")
                        self.log_signal.emit(f"Unfrozen: {current_stage['unfreeze_layers']}")
                    else:
                        agent.unfreeze_layers(None)
                        self.logger.log_message("All layers unfrozen")
                        self.log_signal.emit("All layers unfrozen")
                    
                    # 更新学习率
                    if 'lr_actor' in current_stage and 'lr_critic' in current_stage:
                        agent.set_learning_rate(current_stage['lr_actor'], current_stage['lr_critic'])
                        self.logger.log_message(f"LR: Actor={current_stage['lr_actor']}, Critic={current_stage['lr_critic']}")
                        self.log_signal.emit(f"New LR: Actor={current_stage['lr_actor']:.0e}")
                    
                    # 创建新环境
                    self.env.close_stage()
                    self.env = RealBallEnv(config, mode=self.mode, curriculum_stage=current_stage)
                    self.env.initialize_stage()
                    
                    # 重新连接信号
                    self.env.frame_updated.connect(self.frame_signal.emit)
                    
                    # 重新发送环境引用
                    self.env_ready_signal.emit(self.env)
                    
                    # 重置早停
                    if early_stopping:
                        early_stopping = EarlyStopping(
                            patience=config.EARLY_STOPPING_PATIENCE,
                            min_delta=config.EARLY_STOPPING_MIN_DELTA
                        )
                
                elif is_complete and current_curriculum_idx == len(curriculum_stages) - 1:
                    log_msg = f"{'='*60}"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    log_msg = "ALL CURRICULUM STAGES COMPLETED!"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    log_msg = f"{'='*60}"
                    self.logger.log_message(log_msg)
                    self.log_signal.emit(log_msg)
                    break
        
        # 训练结束
        self.logger.log_message("Training completed!")
        self.logger.log_message(f"Best success rate: {best_success_rate:.2%}")
        
        # 保存最终模型
        final_path = os.path.join(config.CHECKPOINT_DIR, f'{self.mode}_final.pth')
        agent.save(final_path)
        self.logger.log_message(f"Saved final model to {final_path}")
        
        # 关闭环境
        self.env.close_stage()
        
        # 关闭日志
        self.logger.close()
        
        self.finished_signal.emit()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def stop(self):
        self.running = False
        if self.logger:
            self.logger.log_message("Training stopped by user")
            self.logger.close()
