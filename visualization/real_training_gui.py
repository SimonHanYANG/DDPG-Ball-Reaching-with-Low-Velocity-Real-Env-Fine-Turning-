# visualization/real_training_gui.py
import os
import sys
import numpy as np
import cv2
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSpinBox, QTextEdit, QGroupBox, QProgressBar,
                             QDoubleSpinBox, QCheckBox, QTabWidget, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.real_config import RealConfig
from real_training_thread import RealTrainingThread
from camera_thread import CameraThread
from image_process_thread import ImageProcessThread
from yolo_thread import YOLODetectionThread

class RealTrainingGUI(QMainWindow):
    """真实环境训练GUI"""
    
    def __init__(self):
        super().__init__()
        self.config = RealConfig()
        self.training_thread = None
        self.camera_thread = None
        self.image_process_thread = None
        self.yolo_thread = None
        
        # 数据存储
        self.episode_rewards = []
        self.success_rates = []
        self.distance_values = []
        self.velocity_values = []
        
        # 当前环境引用
        self.current_env = None
        
        # Stage控制
        self.stage = None
        
        self.initUI()
        self.init_camera_and_detection()
    
    def initUI(self):
        """初始化UI"""
        self.setWindowTitle('Ball Reaching - Real Environment Training Fine-Tuning')
        self.setGeometry(100, 100, 1800, 900)
        
        # 主widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 左侧：控制面板
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 中间：相机视图
        middle_panel = self.create_camera_panel()
        main_layout.addWidget(middle_panel, 2)
        
        # 右侧：图表
        right_panel = self.create_chart_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("Training Control")
        layout = QVBoxLayout()
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 基本设置标签页
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        
        # 模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Teacher', 'Student'])
        mode_layout.addWidget(self.mode_combo)
        basic_layout.addLayout(mode_layout)
        
        # 训练轮数
        episodes_layout = QHBoxLayout()
        episodes_layout.addWidget(QLabel("Max Episodes:"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1000, 50000)
        self.episodes_spin.setValue(20000)
        self.episodes_spin.setSingleStep(1000)
        episodes_layout.addWidget(self.episodes_spin)
        basic_layout.addLayout(episodes_layout)
        
        # 预训练模型
        pretrain_layout = QVBoxLayout()
        self.pretrain_checkbox = QCheckBox("Load Pretrained Model (from simulation)")
        pretrain_layout.addWidget(self.pretrain_checkbox)
        
        pretrain_btn_layout = QHBoxLayout()
        self.pretrain_path_label = QLabel("No model selected")
        self.pretrain_path_label.setWordWrap(True)
        self.pretrain_path_label.setStyleSheet("color: gray; font-size: 10px;")
        pretrain_btn_layout.addWidget(self.pretrain_path_label)
        
        self.browse_pretrain_btn = QPushButton("Browse")
        self.browse_pretrain_btn.clicked.connect(self.browse_pretrained_model)
        pretrain_btn_layout.addWidget(self.browse_pretrain_btn)
        pretrain_layout.addLayout(pretrain_btn_layout)
        
        basic_layout.addLayout(pretrain_layout)
        
        # 课程学习
        self.curriculum_checkbox = QCheckBox("Enable Curriculum Learning")
        self.curriculum_checkbox.setChecked(True)
        basic_layout.addWidget(self.curriculum_checkbox)
        
        basic_layout.addStretch()
        basic_tab.setLayout(basic_layout)
        tab_widget.addTab(basic_tab, "Basic")
        
        # Stage手动控制标签页
        stage_tab = self.create_stage_control_tab()
        tab_widget.addTab(stage_tab, "Stage Control")
        
        layout.addWidget(tab_widget)
        
        # 开始按钮
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        layout.addWidget(self.start_btn)
        
        # 暂停按钮
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_training)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        # 确认初始化按钮 (训练时使用)
        # self.confirm_init_btn = QPushButton("Confirm Ball Position")
        # self.confirm_init_btn.clicked.connect(self.confirm_manual_init)
        # self.confirm_init_btn.setEnabled(False)
        # self.confirm_init_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        # layout.addWidget(self.confirm_init_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 统计信息
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.episode_label = QLabel("Episode: 0 / 0")
        stats_layout.addWidget(self.episode_label)
        
        self.stage_label = QLabel("Stage: N/A")
        stats_layout.addWidget(self.stage_label)
        
        self.reward_label = QLabel("Reward: 0.00")
        stats_layout.addWidget(self.reward_label)
        
        self.success_label = QLabel("Success Rate: 0.0%")
        stats_layout.addWidget(self.success_label)
        
        self.distance_label = QLabel("Distance: 0.0px")
        stats_layout.addWidget(self.distance_label)
        
        self.velocity_label = QLabel("Velocity: 0.0px/step")
        stats_layout.addWidget(self.velocity_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 日志
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        panel.setLayout(layout)
        return panel
    
    def create_stage_control_tab(self):
        """创建Stage手动控制标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 说明
        info_label = QLabel("Use arrow buttons to manually position the ball:")
        layout.addWidget(info_label)
        
        # 方向按钮
        direction_group = QGroupBox("Direction Control")
        direction_layout = QVBoxLayout()
        
        # 上
        up_layout = QHBoxLayout()
        up_layout.addStretch()
        self.up_btn = QPushButton("↑")
        self.up_btn.setFixedSize(60, 60)
        self.up_btn.clicked.connect(lambda: self.move_stage(0, -self.config.STAGE_STEP_SIZE))
        up_layout.addWidget(self.up_btn)
        up_layout.addStretch()
        direction_layout.addLayout(up_layout)
        
        # 左右
        middle_layout = QHBoxLayout()
        self.left_btn = QPushButton("←")
        self.left_btn.setFixedSize(60, 60)
        self.left_btn.clicked.connect(lambda: self.move_stage(-self.config.STAGE_STEP_SIZE, 0))
        middle_layout.addWidget(self.left_btn)
        
        middle_layout.addStretch()
        
        self.right_btn = QPushButton("→")
        self.right_btn.setFixedSize(60, 60)
        self.right_btn.clicked.connect(lambda: self.move_stage(self.config.STAGE_STEP_SIZE, 0))
        middle_layout.addWidget(self.right_btn)
        direction_layout.addLayout(middle_layout)
        
        # 下
        down_layout = QHBoxLayout()
        down_layout.addStretch()
        self.down_btn = QPushButton("↓")
        self.down_btn.setFixedSize(60, 60)
        self.down_btn.clicked.connect(lambda: self.move_stage(0, self.config.STAGE_STEP_SIZE))
        down_layout.addWidget(self.down_btn)
        down_layout.addStretch()
        direction_layout.addLayout(down_layout)
        
        direction_group.setLayout(direction_layout)
        layout.addWidget(direction_group)
        
        # 步长设置
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size (px):"))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(1.0, 50.0)
        self.step_spin.setValue(self.config.STAGE_STEP_SIZE)
        self.step_spin.setSingleStep(1.0)
        step_layout.addWidget(self.step_spin)
        layout.addLayout(step_layout)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_camera_panel(self):
        """创建相机面板"""
        panel = QGroupBox("Camera View")
        layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 800)
        self.video_label.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.video_label)
        
        panel.setLayout(layout)
        return panel
    
    def create_chart_panel(self):
        """创建图表面板"""
        panel = QGroupBox("Training Metrics")
        layout = QVBoxLayout()
        
        # Reward图表
        self.reward_figure = Figure(figsize=(5, 2))
        self.reward_canvas = FigureCanvas(self.reward_figure)
        self.reward_ax = self.reward_figure.add_subplot(111)
        self.reward_ax.set_title('Episode Reward')
        self.reward_ax.set_xlabel('Episode')
        self.reward_ax.set_ylabel('Reward')
        layout.addWidget(self.reward_canvas)
        
        # Success Rate图表
        self.success_figure = Figure(figsize=(5, 2))
        self.success_canvas = FigureCanvas(self.success_figure)
        self.success_ax = self.success_figure.add_subplot(111)
        self.success_ax.set_title('Success Rate (Last 100 Episodes)')
        self.success_ax.set_xlabel('Episode')
        self.success_ax.set_ylabel('Success Rate')
        layout.addWidget(self.success_canvas)
        
        # Distance图表
        self.distance_figure = Figure(figsize=(5, 2))
        self.distance_canvas = FigureCanvas(self.distance_figure)
        self.distance_ax = self.distance_figure.add_subplot(111)
        self.distance_ax.set_title('Final Distance (px)')
        self.distance_ax.set_xlabel('Episode')
        self.distance_ax.set_ylabel('Distance (px)')
        layout.addWidget(self.distance_canvas)
        
        # Velocity图表
        self.velocity_figure = Figure(figsize=(5, 2))
        self.velocity_canvas = FigureCanvas(self.velocity_figure)
        self.velocity_ax = self.velocity_figure.add_subplot(111)
        self.velocity_ax.set_title('Final Velocity (px/step)')
        self.velocity_ax.set_xlabel('Episode')
        self.velocity_ax.set_ylabel('Velocity (px/step)')
        layout.addWidget(self.velocity_canvas)
        
        panel.setLayout(layout)
        return panel
    
    def init_camera_and_detection(self):
        """初始化相机和检测线程"""
        try:
            # 图像处理线程
            self.image_process_thread = ImageProcessThread()
            
            # 相机线程
            self.camera_thread = CameraThread(self.image_process_thread)
            self.camera_thread.error_signal.connect(self.on_camera_error)
            
            # YOLO检测线程
            self.yolo_thread = YOLODetectionThread(self.config.YOLO_ENGINE_PATH)
            self.yolo_thread.detection_result_signal.connect(self.on_detection_result)
            self.yolo_thread.status_signal.connect(self.log_message)
            
            # 连接信号
            self.image_process_thread.processed_image_signal.connect(self.yolo_thread.update_frame)
            
            # 启动线程
            self.image_process_thread.start()
            self.camera_thread.start()
            self.yolo_thread.start()
            self.yolo_thread.start_detection()
            
            self.log_message("Camera and detection initialized", "info")
            
        except Exception as e:
            self.log_message(f"Failed to initialize camera: {str(e)}", "error")
    
    def browse_pretrained_model(self):
        """浏览预训练模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pretrained Model",
            "checkpoints",
            "Model Files (*.pth)"
        )
        
        if file_path:
            self.pretrain_path_label.setText(os.path.basename(file_path))
            self.pretrain_path_label.setStyleSheet("color: green; font-size: 10px;")
            self.pretrained_model_path = file_path
    
    def move_stage(self, dx_px, dy_px):
        """手动移动Stage
        
        Args:
            dx_px: X方向移动 (pixel)
            dy_px: Y方向移动 (pixel)
        """
        if not self.stage:
            from stage import Stage
            try:
                self.stage = Stage()
            except Exception as e:
                self.log_message(f"Failed to initialize Stage: {str(e)}", "error")
                return
        
        # 获取当前步长 (pixel)
        step_size = self.step_spin.value() if hasattr(self, 'step_spin') else self.config.STAGE_STEP_SIZE
        dx_px = dx_px / self.config.STAGE_STEP_SIZE * step_size
        dy_px = dy_px / self.config.STAGE_STEP_SIZE * step_size
        
        # 转换为微米
        dx_um = dx_px * self.config.PIXEL_TO_UM_X
        dy_um = dy_px * self.config.PIXEL_TO_UM_Y
        
        try:
            self.stage.move_xy_relative(dx_um, dy_um)
            self.log_message(f"Stage moved: ({dx_px:.1f}, {dy_px:.1f}) px = ({dx_um:.0f}, {dy_um:.0f}) μm", "info")
        except Exception as e:
            self.log_message(f"Stage move failed: {str(e)}", "error")
    
    # def confirm_manual_init(self):
    #     """确认手动初始化完成"""
    #     if self.training_thread:
    #         self.training_thread.confirm_manual_init()
    #         self.confirm_init_btn.setEnabled(False)
    #         self.log_message("Ball position confirmed. Episode starting...", "info")
    
    def start_training(self):
        """开始训练"""
        mode = self.mode_combo.currentText().lower()
        episodes = self.episodes_spin.value()
        
        # 预训练模型
        pretrained_model = None
        if self.pretrain_checkbox.isChecked() and hasattr(self, 'pretrained_model_path'):
            pretrained_model = self.pretrained_model_path
        
        # 课程学习配置
        curriculum_config = None
        if self.curriculum_checkbox.isChecked() and mode == 'teacher':
            curriculum_config = self.config.CURRICULUM_STAGES
        
        # **加载断点续训的历史数据**
        self.load_checkpoint_data(mode)
        
        self.log_message(f"Starting {mode} training (max {episodes} episodes)...", "info")
        
        # 创建训练线程
        self.training_thread = RealTrainingThread(
            mode=mode,
            total_episodes=episodes,
            curriculum_config=curriculum_config,
            pretrained_model=pretrained_model
        )
        
        self.training_thread.update_signal.connect(self.update_stats)
        self.training_thread.frame_signal.connect(self.update_training_frame)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.log_signal.connect(self.log_message)
        # self.training_thread.request_manual_init_signal.connect(self.on_request_manual_init)
        
        # 获取环境引用以便绘制
        self.training_thread.env_ready_signal.connect(self.on_env_ready)
        
        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # 开始训练
        self.training_thread.start()

    def load_checkpoint_data(self, mode):
        """加载断点续训的历史数据"""
        import csv
        
        experiment_name = f'{mode}_training_real'
        episode_csv = os.path.join(self.config.LOG_DIR, experiment_name, 'episode_log.csv')
        
        if os.path.exists(episode_csv):
            try:
                with open(episode_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.episode_rewards.append(float(row['reward']))
                        # 计算成功率
                        success = float(row['success'])
                        if len(self.success_rates) == 0:
                            self.success_rates.append(success)
                        else:
                            # 滚动窗口计算
                            window = 100
                            recent = self.success_rates[-window:] if len(self.success_rates) >= window else self.success_rates
                            recent_sum = sum(recent) + success
                            self.success_rates.append(recent_sum / (len(recent) + 1))
                        
                        self.distance_values.append(float(row['distance']))
                        self.velocity_values.append(float(row['velocity']))
                
                self.log_message(f"Loaded {len(self.episode_rewards)} episodes from checkpoint", "info")
                
                # 更新图表
                self.update_charts()
                
            except Exception as e:
                self.log_message(f"Failed to load checkpoint data: {e}", "warning")

    def on_env_ready(self, env):
        """环境准备好时获取引用"""
        self.current_env = env
    
    def update_training_frame(self, frame):
        """更新训练帧 - 直接显示从环境发来的帧"""
        self.update_frame(frame)
    
    def on_detection_result(self, frame, boxes, scores, class_ids, tracks):
        """处理检测结果"""
        # 绘制检测框（蓝色）
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 绘制JPDAF跟踪轨迹（绿色） - 只在非训练时显示
        if not self.training_thread or not self.training_thread.isRunning():
            for track_id, track_data in tracks.items():
                trajectory = track_data['trajectory']
                for i in range(len(trajectory) - 1):
                    pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
                    pt2 = (int(trajectory[i+1][0]), int(trajectory[i+1][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # 更新环境检测
        if self.current_env:
            self.current_env.update_detection(frame, boxes, tracks)
        
        # 显示（只在非训练时直接显示）
        if not self.training_thread or not self.training_thread.isRunning():
            self.update_frame(frame)
    
    # def on_request_manual_init(self):
    #     """收到手动初始化请求"""
    #     self.confirm_init_btn.setEnabled(True)
    #     self.log_message("Please position ball using arrow buttons, then click 'Confirm'", "info")
    
    def pause_training(self):
        """暂停训练"""
        if self.training_thread:
            if self.pause_btn.text() == "Pause":
                self.training_thread.pause()
                self.pause_btn.setText("Resume")
                self.log_message("Training paused", "info")
            else:
                self.training_thread.resume()
                self.pause_btn.setText("Pause")
                self.log_message("Training resumed", "info")
    
    def stop_training(self):
        """停止训练"""
        if self.training_thread:
            self.training_thread.stop()
            self.log_message("Stopping training...", "info")
    
    def training_finished(self):
        """训练结束"""
        self.log_message("Training finished!", "info")
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        # self.confirm_init_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
    
    def update_frame(self, frame):
        """更新视频帧"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_stats(self, data):
        """更新统计信息"""
        self.episode_label.setText(f"Episode: {data['episode']} / {data['total_episodes']}")
        self.stage_label.setText(f"Stage: {data.get('curriculum_stage', 'N/A')}")
        self.reward_label.setText(f"Reward: {data['reward']:.2f}")
        self.success_label.setText(f"Success Rate: {data['success_rate']*100:.1f}%")
        self.distance_label.setText(f"Distance: {data['distance']:.2f}px")
        self.velocity_label.setText(f"Velocity: {data['velocity']:.2f}px/step")
        
        # 更新进度条
        progress = int(100 * data['episode'] / data['total_episodes'])
        self.progress_bar.setValue(progress)
        
        # 更新数据
        self.episode_rewards.append(data['reward'])
        self.success_rates.append(data['success_rate'])
        self.distance_values.append(data['distance'])
        self.velocity_values.append(data['velocity'])
        
        # 更新图表
        if data['episode'] % 10 == 0:
            self.update_charts()
    
    def update_charts(self):
        """更新图表"""
        # Reward
        self.reward_ax.clear()
        self.reward_ax.plot(self.episode_rewards, 'b-', alpha=0.3)
        if len(self.episode_rewards) >= 10:
            smooth = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            self.reward_ax.plot(range(9, len(self.episode_rewards)), smooth, 'b-', linewidth=2)
        self.reward_ax.set_title('Episode Reward')
        self.reward_ax.set_xlabel('Episode')
        self.reward_ax.set_ylabel('Reward')
        self.reward_ax.grid(True, alpha=0.3)
        self.reward_canvas.draw()
        
        # Success Rate
        self.success_ax.clear()
        self.success_ax.plot(self.success_rates, 'g-', linewidth=2)
        self.success_ax.set_title('Success Rate')
        self.success_ax.set_xlabel('Episode')
        self.success_ax.set_ylabel('Success Rate')
        self.success_ax.set_ylim([0, 1.1])
        self.success_ax.grid(True, alpha=0.3)
        self.success_canvas.draw()
        
        # Distance
        self.distance_ax.clear()
        self.distance_ax.plot(self.distance_values, 'r-', alpha=0.3)
        if len(self.distance_values) >= 10:
            smooth = np.convolve(self.distance_values, np.ones(10)/10, mode='valid')
            self.distance_ax.plot(range(9, len(self.distance_values)), smooth, 'r-', linewidth=2)
        self.distance_ax.set_title('Final Distance')
        self.distance_ax.set_xlabel('Episode')
        self.distance_ax.set_ylabel('Distance (px)')
        self.distance_ax.grid(True, alpha=0.3)
        self.distance_canvas.draw()
        
        # Velocity
        self.velocity_ax.clear()
        self.velocity_ax.plot(self.velocity_values, 'm-', alpha=0.3)
        if len(self.velocity_values) >= 10:
            smooth = np.convolve(self.velocity_values, np.ones(10)/10, mode='valid')
            self.velocity_ax.plot(range(9, len(self.velocity_values)), smooth, 'm-', linewidth=2)
        self.velocity_ax.set_title('Final Velocity')
        self.velocity_ax.set_xlabel('Episode')
        self.velocity_ax.set_ylabel('Velocity (px/step)')
        self.velocity_ax.grid(True, alpha=0.3)
        self.velocity_canvas.draw()
    
    def log_message(self, message, msg_type="info"):
        """添加日志消息"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_camera_error(self, error_msg):
        """相机错误处理"""
        self.log_message(f"Camera Error: {error_msg}", "error")
    
    def closeEvent(self, event):
        """关闭事件"""
        # 停止所有线程
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.wait()
        
        if self.yolo_thread:
            self.yolo_thread.stop()
            self.yolo_thread.wait()
        
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        if self.image_process_thread:
            self.image_process_thread.stop()
            self.image_process_thread.wait()
        
        if self.stage:
            self.stage.close()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = RealTrainingGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()