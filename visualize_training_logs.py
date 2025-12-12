# visualize_training_logs.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import json

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class TrainingLogVisualizer:
    """训练日志可视化工具"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化可视化工具
        
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.experiment_name = experiment_name
        
        # 检查目录是否存在
        if not self.log_dir.exists():
            raise ValueError(f"Log directory not found: {self.log_dir}")
        
        # 加载数据
        self.episode_df = None
        self.losses_df = None
        self.load_data()
        
    def load_data(self):
        """加载CSV日志数据"""
        episode_csv = self.log_dir / 'episode_log.csv'
        losses_csv = self.log_dir / 'losses_log.csv'
        
        if episode_csv.exists():
            self.episode_df = pd.read_csv(episode_csv)
            print(f"Loaded {len(self.episode_df)} episodes")
        else:
            print(f"Warning: {episode_csv} not found")
        
        if losses_csv.exists():
            self.losses_df = pd.read_csv(losses_csv)
            print(f"Loaded {len(self.losses_df)} loss records")
        else:
            print(f"Warning: {losses_csv} not found")
    
    def plot_episode_rewards(self, ax=None, window=50):
        """绘制Episode奖励曲线"""
        if self.episode_df is None:
            print("No episode data available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = self.episode_df['episode'].values
        rewards = self.episode_df['reward'].values
        
        # 原始曲线（半透明）
        ax.plot(episodes, rewards, alpha=0.3, linewidth=1, label='Raw Rewards')
        
        # 滑动平均
        if len(rewards) >= window:
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, smoothed, linewidth=2, label=f'{window}-Episode Moving Average')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        ax.axhline(y=mean_reward, color='r', linestyle='--', alpha=0.5, 
                   label=f'Mean: {mean_reward:.2f}')
        ax.text(0.02, 0.98, f'Max: {max_reward:.2f}\nMean: {mean_reward:.2f}\nMin: {np.min(rewards):.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    def plot_success_rate(self, ax=None, window=100):
        """绘制成功率曲线"""
        if self.episode_df is None:
            print("No episode data available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = self.episode_df['episode'].values
        successes = self.episode_df['success'].values
        
        # 计算滑动窗口成功率
        success_rate = pd.Series(successes).rolling(window=window, min_periods=1).mean()
        
        ax.plot(episodes, success_rate, linewidth=2, color='green')
        ax.fill_between(episodes, 0, success_rate, alpha=0.3, color='green')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title(f'Success Rate (Last {window} Episodes)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # 添加里程碑线
        milestones = [0.5, 0.7, 0.8, 0.9]
        colors = ['orange', 'yellow', 'lightgreen', 'darkgreen']
        for milestone, color in zip(milestones, colors):
            ax.axhline(y=milestone, color=color, linestyle='--', alpha=0.5,
                      label=f'{milestone*100:.0f}% Success')
        
        ax.legend(loc='best')
        
        # 添加最终成功率
        final_sr = success_rate.iloc[-1] if len(success_rate) > 0 else 0
        ax.text(0.02, 0.98, f'Final Success Rate: {final_sr:.2%}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                fontsize=12, fontweight='bold')
        
        return ax
    
    def plot_distance_and_velocity(self, ax1=None, ax2=None, window=50):
        """绘制距离和速度曲线"""
        if self.episode_df is None:
            print("No episode data available")
            return
        
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        episodes = self.episode_df['episode'].values
        distances = self.episode_df['distance'].values
        velocities = self.episode_df['velocity'].values
        
        # 距离曲线
        ax1.plot(episodes, distances, alpha=0.3, linewidth=1, label='Raw Distance')
        if len(distances) >= window:
            smoothed_dist = pd.Series(distances).rolling(window=window, min_periods=1).mean()
            ax1.plot(episodes, smoothed_dist, linewidth=2, 
                    label=f'{window}-Episode Moving Average')
        
        ax1.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, 
                   label='Target Tolerance (5px)')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Final Distance (px)', fontsize=12)
        ax1.set_title('Final Distance to Target', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 速度曲线
        ax2.plot(episodes, velocities, alpha=0.3, linewidth=1, label='Raw Velocity')
        if len(velocities) >= window:
            smoothed_vel = pd.Series(velocities).rolling(window=window, min_periods=1).mean()
            ax2.plot(episodes, smoothed_vel, linewidth=2,
                    label=f'{window}-Episode Moving Average')
        
        ax2.axhline(y=8.0, color='r', linestyle='--', alpha=0.5,
                   label='Target Threshold (8px/step)')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Final Velocity (px/step)', fontsize=12)
        ax2.set_title('Final Velocity', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        return ax1, ax2
    
    def plot_losses(self, ax1=None, ax2=None, window=100):
        """绘制损失曲线"""
        if self.losses_df is None:
            print("No loss data available")
            return
        
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        timesteps = self.losses_df['timestep'].values
        critic_loss = self.losses_df['critic_loss'].values
        actor_loss = self.losses_df['actor_loss'].values
        
        # Critic损失
        ax1.plot(timesteps, critic_loss, alpha=0.3, linewidth=1, label='Raw Critic Loss')
        if len(critic_loss) >= window:
            smoothed_critic = pd.Series(critic_loss).rolling(window=window, min_periods=1).mean()
            ax1.plot(timesteps, smoothed_critic, linewidth=2,
                    label=f'{window}-Step Moving Average')
        
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Critic Loss', fontsize=12)
        ax1.set_title('Critic Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Actor损失
        ax2.plot(timesteps, actor_loss, alpha=0.3, linewidth=1, label='Raw Actor Loss')
        if len(actor_loss) >= window:
            smoothed_actor = pd.Series(actor_loss).rolling(window=window, min_periods=1).mean()
            ax2.plot(timesteps, smoothed_actor, linewidth=2,
                    label=f'{window}-Step Moving Average')
        
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Actor Loss', fontsize=12)
        ax2.set_title('Actor Loss Over Time', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        return ax1, ax2
    
    def plot_q_values(self, ax=None, window=100):
        """绘制Q值曲线"""
        if self.losses_df is None or 'q_value' not in self.losses_df.columns:
            print("No Q-value data available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        timesteps = self.losses_df['timestep'].values
        q_values = self.losses_df['q_value'].values
        
        ax.plot(timesteps, q_values, alpha=0.3, linewidth=1, label='Raw Q-Value')
        if len(q_values) >= window:
            smoothed_q = pd.Series(q_values).rolling(window=window, min_periods=1).mean()
            ax.plot(timesteps, smoothed_q, linewidth=2,
                   label=f'{window}-Step Moving Average')
        
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Q-Value', fontsize=12)
        ax.set_title('Q-Value Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_episode_length(self, ax=None, window=50):
        """绘制Episode长度曲线"""
        if self.episode_df is None:
            print("No episode data available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = self.episode_df['episode'].values
        steps = self.episode_df['steps'].values
        
        ax.plot(episodes, steps, alpha=0.3, linewidth=1, label='Raw Episode Length')
        if len(steps) >= window:
            smoothed_steps = pd.Series(steps).rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, smoothed_steps, linewidth=2,
                   label=f'{window}-Episode Moving Average')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Steps', fontsize=12)
        ax.set_title('Episode Length Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_steps = np.mean(steps)
        ax.text(0.02, 0.98, f'Mean Steps: {mean_steps:.1f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    def plot_success_vs_distance(self, ax=None):
        """绘制成功率与初始距离的关系（散点图）"""
        if self.episode_df is None:
            print("No episode data available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # 假设距离越大说明初始位置越远
        distances = self.episode_df['distance'].values
        successes = self.episode_df['success'].values
        
        # 创建距离区间
        bins = np.linspace(0, max(distances), 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # 计算每个区间的成功率
        success_rates = []
        for i in range(len(bins) - 1):
            mask = (distances >= bins[i]) & (distances < bins[i+1])
            if mask.sum() > 0:
                success_rates.append(successes[mask].mean())
            else:
                success_rates.append(0)
        
        ax.bar(bin_centers, success_rates, width=(bins[1]-bins[0])*0.8, 
               alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Final Distance (px)', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate vs Final Distance', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_training_summary(self):
        """绘制训练摘要（包含所有关键指标）"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 第一行
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_episode_rewards(ax1)
        
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_success_vs_distance(ax2)
        
        # 第二行
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_success_rate(ax3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        if self.losses_df is not None:
            self.plot_losses(ax4, ax5)
        
        # 第三行
        ax6 = fig.add_subplot(gs[2, 0])
        ax7 = fig.add_subplot(gs[2, 1])
        self.plot_distance_and_velocity(ax6, ax7)
        
        ax8 = fig.add_subplot(gs[2, 2])
        self.plot_episode_length(ax8)
        
        fig.suptitle(f'Training Summary: {self.experiment_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def generate_statistics_report(self):
        """生成统计报告"""
        if self.episode_df is None:
            print("No episode data available")
            return
        
        print("\n" + "="*60)
        print(f"TRAINING STATISTICS REPORT: {self.experiment_name}")
        print("="*60)
        
        # Episode统计
        print("\n[Episode Statistics]")
        print(f"Total Episodes: {len(self.episode_df)}")
        print(f"Total Timesteps: {self.episode_df['timestep'].max() if 'timestep' in self.episode_df else 'N/A'}")
        
        # 奖励统计
        print("\n[Reward Statistics]")
        rewards = self.episode_df['reward'].values
        print(f"Mean Reward: {np.mean(rewards):.2f}")
        print(f"Max Reward: {np.max(rewards):.2f}")
        print(f"Min Reward: {np.min(rewards):.2f}")
        print(f"Std Reward: {np.std(rewards):.2f}")
        
        # 成功率统计
        print("\n[Success Statistics]")
        successes = self.episode_df['success'].values
        overall_sr = np.mean(successes)
        last_100_sr = np.mean(successes[-100:]) if len(successes) >= 100 else overall_sr
        last_50_sr = np.mean(successes[-50:]) if len(successes) >= 50 else overall_sr
        
        print(f"Overall Success Rate: {overall_sr:.2%}")
        print(f"Last 100 Episodes: {last_100_sr:.2%}")
        print(f"Last 50 Episodes: {last_50_sr:.2%}")
        
        # 距离和速度统计
        print("\n[Distance & Velocity Statistics]")
        distances = self.episode_df['distance'].values
        velocities = self.episode_df['velocity'].values
        
        print(f"Mean Final Distance: {np.mean(distances):.2f} px")
        print(f"Mean Final Velocity: {np.mean(velocities):.2f} px/step")
        
        # 成功episode的统计
        success_mask = successes == 1
        if success_mask.sum() > 0:
            print(f"\n[Successful Episodes Only]")
            print(f"Count: {success_mask.sum()}")
            print(f"Mean Steps: {self.episode_df.loc[success_mask, 'steps'].mean():.1f}")
            print(f"Mean Reward: {self.episode_df.loc[success_mask, 'reward'].mean():.2f}")
        
        # 失败episode的统计
        fail_mask = successes == 0
        if fail_mask.sum() > 0:
            print(f"\n[Failed Episodes Only]")
            print(f"Count: {fail_mask.sum()}")
            print(f"Mean Final Distance: {self.episode_df.loc[fail_mask, 'distance'].mean():.2f} px")
            print(f"Mean Steps: {self.episode_df.loc[fail_mask, 'steps'].mean():.1f}")
        
        print("\n" + "="*60 + "\n")
    
    def save_all_plots(self, output_dir: str = None):
        """保存所有图表到文件"""
        if output_dir is None:
            output_dir = self.log_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存摘要图
        print("Generating training summary plot...")
        summary_fig = self.plot_training_summary()
        summary_path = output_dir / f'{self.experiment_name}_summary.png'
        summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {summary_path}")
        
        # 保存PDF版本
        pdf_path = output_dir / f'{self.experiment_name}_summary.pdf'
        summary_fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        
        plt.close(summary_fig)
        
        # 保存单独的图表
        individual_plots = [
            ('rewards', self.plot_episode_rewards),
            ('success_rate', self.plot_success_rate),
            ('episode_length', self.plot_episode_length),
            ('q_values', self.plot_q_values),
            ('success_vs_distance', self.plot_success_vs_distance)
        ]
        
        for name, plot_func in individual_plots:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_func(ax)
                plot_path = output_dir / f'{self.experiment_name}_{name}.png'
                fig.savefig(plot_path, dpi=200, bbox_inches='tight')
                print(f"Saved: {plot_path}")
                plt.close(fig)
            except Exception as e:
                print(f"Failed to save {name}: {e}")
        
        print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training logs')
    parser.add_argument('--log_dir', type=str, default='logs_real',
                       help='Log directory')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., teacher_training_real_progressive)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots in window')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save plots to file')
    
    args = parser.parse_args()
    
    try:
        # 创建可视化工具
        visualizer = TrainingLogVisualizer(args.log_dir, args.experiment)
        
        # 生成统计报告
        visualizer.generate_statistics_report()
        
        # 保存图表
        if not args.no_save:
            visualizer.save_all_plots(args.output_dir)
        
        # 显示图表
        if args.show:
            visualizer.plot_training_summary()
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

'''
# 可视化teacher训练日志
python visualize_training_logs.py --experiment teacher_training_real_progressive

# 可视化student训练日志
python visualize_training_logs.py --experiment student_training_real_progressive

# 保存并显示图表
python visualize_training_logs.py --experiment teacher_training_real_progressive --show

# 指定输出目录
python visualize_training_logs.py --experiment teacher_training_real_progressive --output_dir ./my_plots
'''
if __name__ == '__main__':
    main()