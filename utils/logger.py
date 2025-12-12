# utils/logger.py
import os
import csv
from datetime import datetime
from typing import Dict, Any

class Logger:
    """训练日志记录器 - 支持追加模式"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # CSV文件路径
        self.episode_csv_file = os.path.join(self.log_dir, 'episode_log.csv')
        self.losses_csv_file = os.path.join(self.log_dir, 'losses_log.csv')
        self.log_file = os.path.join(self.log_dir, 'training.log')
        
        # 文件句柄
        self.episode_csv_file_handle = None
        self.episode_csv_writer = None
        self.losses_csv_file_handle = None
        self.losses_csv_writer = None
        
        # 标记是否已关闭
        self.is_closed = False
        
        self._init_csv()
    
    def _init_csv(self):
        """初始化CSV文件 - 追加模式"""
        # Episode CSV
        episode_file_exists = os.path.isfile(self.episode_csv_file)
        self.episode_csv_file_handle = open(self.episode_csv_file, 'a', newline='')
        self.episode_csv_writer = csv.writer(self.episode_csv_file_handle)
        
        if not episode_file_exists:
            # 新文件，写入表头
            self.episode_csv_writer.writerow([
                'episode', 'timestep', 'reward', 'steps', 'distance', 
                'velocity', 'success'
            ])
            self.episode_csv_file_handle.flush()
        
        # Losses CSV
        losses_file_exists = os.path.isfile(self.losses_csv_file)
        self.losses_csv_file_handle = open(self.losses_csv_file, 'a', newline='')
        self.losses_csv_writer = csv.writer(self.losses_csv_file_handle)
        
        if not losses_file_exists:
            # 新文件，写入表头
            self.losses_csv_writer.writerow([
                'episode', 'timestep', 'critic_loss', 'actor_loss', 'q_value',
                'imitation_loss', 'imitation_weight'
            ])
            self.losses_csv_file_handle.flush()
    
    def log_episode(self, episode: int, timestep: int, info: Dict[str, Any]):
        """记录Episode信息"""
        if self.is_closed:
            return
        
        self.episode_csv_writer.writerow([
            episode,
            timestep,
            info.get('episode_reward', 0),
            info.get('step', 0),
            info.get('distance', 0),
            info.get('velocity', 0),
            1 if info.get('termination') == 'success' else 0
        ])
        self.episode_csv_file_handle.flush()
    
    def log_losses(self, episode: int, timestep: int, losses: Dict[str, float]):
        """记录Loss信息"""
        if self.is_closed:
            return
        
        self.losses_csv_writer.writerow([
            episode,
            timestep,
            losses.get('critic_loss', 0),
            losses.get('actor_loss', 0),
            losses.get('q_value', 0),
            losses.get('imitation_loss', 0),
            losses.get('imitation_weight', 0)
        ])
        self.losses_csv_file_handle.flush()
    
    def log_message(self, message: str):
        """记录消息 - 追加模式"""
        if self.is_closed:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"
        
        # 追加到文件
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        print(log_line.strip())
    
    def close(self):
        """关闭日志文件"""
        if self.is_closed:
            return
        
        self.is_closed = True
        
        if self.episode_csv_file_handle:
            self.episode_csv_file_handle.close()
        
        if self.losses_csv_file_handle:
            self.losses_csv_file_handle.close()