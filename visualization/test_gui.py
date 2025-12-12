# visualization/test_gui.py
import os
import sys
import numpy as np
import cv2
import time
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QTextEdit, QGroupBox, QFileDialog, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QTabWidget, QRadioButton)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.real_config import RealConfig
from envs.real_ball_env import RealBallEnv
from models.ddpg_agent import DDPGAgent
from camera_thread import CameraThread
from image_process_thread import ImageProcessThread
from yolo_thread import YOLODetectionThread

class ClickableLabel(QLabel):
    """可点击的标签用于选择目标点"""
    clicked = pyqtSignal(QPoint)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.pos())

class TestGUI(QMainWindow):
    """测试GUI - 支持手动目标点选择"""
    
    def __init__(self):
        super().__init__()
        self.config = RealConfig()
        self.camera_thread = None
        self.image_process_thread = None
        self.yolo_thread = None
        self.stage = None
        
        # 环境和智能体
        self.env = None
        self.agent = None
        self.model_loaded = False
        self.model_mode = 'teacher'  # 默认teacher模式
        
        # 测试状态
        self.testing = False
        self.current_target = None
        self.episode_count = 0
        self.total_rewards = []
        self.total_successes = []
        
        self.initUI()
        self.init_camera_and_detection()
        self.init_env_and_stage()
    
    def initUI(self):
        """初始化UI"""
        self.setWindowTitle('Ball Reaching - Interactive Testing')
        self.setGeometry(100, 100, 1600, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 左侧：控制面板
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右侧：相机视图
        right_panel = self.create_camera_panel()
        main_layout.addWidget(right_panel, 3)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("Test Control")
        layout = QVBoxLayout()
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 模型选择标签页
        model_tab = QWidget()
        model_layout = QVBoxLayout()
        
        model_group = QGroupBox("Model Selection")
        model_group_layout = QVBoxLayout()
        
        # 模型类型选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Model Type:"))
        self.model_mode_combo = QComboBox()
        self.model_mode_combo.addItems(['Teacher', 'Student'])
        self.model_mode_combo.currentTextChanged.connect(self.on_model_mode_changed)
        mode_layout.addWidget(self.model_mode_combo)
        model_group_layout.addLayout(mode_layout)
        
        # 模型路径
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: gray;")
        model_group_layout.addWidget(self.model_path_label)
        
        browse_btn = QPushButton("Browse Model")
        browse_btn.clicked.connect(self.browse_model)
        model_group_layout.addWidget(browse_btn)
        
        # 提示信息
        hint_label = QLabel("Hint: Teacher model uses 12-dim state,\nStudent model uses 22-dim state")
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        hint_label.setWordWrap(True)
        model_group_layout.addWidget(hint_label)
        
        model_group.setLayout(model_group_layout)
        model_layout.addWidget(model_group)
        
        # 测试模式
        mode_group = QGroupBox("Test Mode")
        mode_layout = QVBoxLayout()
        
        self.auto_mode_radio = QRadioButton("Auto Mode (Random Targets)")
        self.manual_mode_radio = QRadioButton("Manual Mode (Click to Set Target)")
        self.manual_mode_radio.setChecked(True)
        
        mode_layout.addWidget(self.auto_mode_radio)
        mode_layout.addWidget(self.manual_mode_radio)
        
        mode_group.setLayout(mode_layout)
        model_layout.addWidget(mode_group)
        
        # 自动模式设置
        auto_group = QGroupBox("Auto Mode Settings")
        auto_layout = QVBoxLayout()
        
        episodes_layout = QHBoxLayout()
        episodes_layout.addWidget(QLabel("Episodes:"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 100)
        self.episodes_spin.setValue(10)
        episodes_layout.addWidget(self.episodes_spin)
        auto_layout.addLayout(episodes_layout)
        
        auto_group.setLayout(auto_layout)
        model_layout.addWidget(auto_group)
        
        model_layout.addStretch()
        model_tab.setLayout(model_layout)
        tab_widget.addTab(model_tab, "Model")
        
        # Stage控制标签页
        stage_tab = self.create_stage_control_tab()
        tab_widget.addTab(stage_tab, "Stage Control")
        
        layout.addWidget(tab_widget)
        
        # 控制按钮
        self.start_btn = QPushButton("Start Test")
        self.start_btn.clicked.connect(self.start_test)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Test")
        self.stop_btn.clicked.connect(self.stop_test)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        # 重置按钮
        reset_btn = QPushButton("Reset Episode")
        reset_btn.clicked.connect(self.reset_episode)
        layout.addWidget(reset_btn)
        
        # 统计信息
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.episode_label = QLabel("Episode: 0")
        stats_layout.addWidget(self.episode_label)
        
        self.reward_label = QLabel("Current Reward: 0.00")
        stats_layout.addWidget(self.reward_label)
        
        self.success_label = QLabel("Success: No")
        stats_layout.addWidget(self.success_label)
        
        self.distance_label = QLabel("Distance: 0.0px")
        stats_layout.addWidget(self.distance_label)
        
        self.steps_label = QLabel("Steps: 0")
        stats_layout.addWidget(self.steps_label)
        
        stats_layout.addWidget(QLabel("--- Totals ---"))
        
        self.total_episodes_label = QLabel("Total Episodes: 0")
        stats_layout.addWidget(self.total_episodes_label)
        
        self.avg_reward_label = QLabel("Avg Reward: 0.00")
        stats_layout.addWidget(self.avg_reward_label)
        
        self.success_rate_label = QLabel("Success Rate: 0.0%")
        stats_layout.addWidget(self.success_rate_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 日志
        log_group = QGroupBox("Test Log")
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
        """创建Stage控制标签页"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel("Use arrow buttons to manually position the ball:")
        layout.addWidget(info_label)
        
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
        panel = QGroupBox("Camera View - Click to Set Target")
        layout = QVBoxLayout()
        
        # 使用可点击的标签
        self.video_label = ClickableLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(1200, 800)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.clicked.connect(self.on_video_clicked)
        layout.addWidget(self.video_label)
        
        # 提示信息
        hint_label = QLabel("Click on the camera view to set target position")
        hint_label.setStyleSheet("color: blue; font-weight: bold;")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint_label)
        
        panel.setLayout(layout)
        return panel
    
    def init_camera_and_detection(self):
        """初始化相机和检测"""
        try:
            self.image_process_thread = ImageProcessThread()
            
            self.camera_thread = CameraThread(self.image_process_thread)
            self.camera_thread.error_signal.connect(self.on_camera_error)
            
            self.yolo_thread = YOLODetectionThread(self.config.YOLO_ENGINE_PATH)
            self.yolo_thread.detection_result_signal.connect(self.on_detection_result)
            self.yolo_thread.status_signal.connect(self.log_message)
            
            self.image_process_thread.processed_image_signal.connect(self.yolo_thread.update_frame)
            
            self.image_process_thread.start()
            self.camera_thread.start()
            self.yolo_thread.start()
            self.yolo_thread.start_detection()
            
            self.log_message("Camera and detection initialized", "info")
            
        except Exception as e:
            self.log_message(f"Failed to initialize camera: {str(e)}", "error")
    
    def init_env_and_stage(self):
        """初始化环境和Stage"""
        try:
            # 使用默认teacher模式初始化
            self.env = RealBallEnv(self.config, mode='teacher', curriculum_stage=None)
            
            if not self.env.initialize_stage():
                self.log_message("Failed to initialize Stage!", "error")
                return
            
            # 获取Stage引用
            self.stage = self.env.stage
            
            self.log_message("Environment and Stage initialized", "info")
            
        except Exception as e:
            self.log_message(f"Failed to initialize environment: {str(e)}", "error")
    
    def on_model_mode_changed(self, mode_text):
        """模型类型改变时的回调"""
        self.model_mode = mode_text.lower()
        
        # 重新创建环境
        if self.env:
            self.env.close_stage()
            self.env = RealBallEnv(self.config, mode=self.model_mode, curriculum_stage=None)
            self.env.initialize_stage()
            self.stage = self.env.stage
        
        # 如果已加载模型，清除
        if self.model_loaded:
            self.model_loaded = False
            self.start_btn.setEnabled(False)
            self.model_path_label.setText("Please reload model for new mode")
            self.model_path_label.setStyleSheet("color: orange;")
            self.log_message(f"Switched to {mode_text} mode. Please reload model.", "info")
    
    def browse_model(self):
        """浏览模型文件"""
        # 根据模型类型设置默认目录
        if self.model_mode == 'teacher':
            default_dir = "checkpoints_real"
        else:
            default_dir = "checkpoints_real"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self.model_mode.capitalize()} Model",
            default_dir,
            "Model Files (*.pth)"
        )
        
        if file_path:
            self.model_path_label.setText(os.path.basename(file_path))
            self.model_path = file_path
            
            # 加载模型
            self.load_model(file_path)
    
    def load_model(self, model_path):
        """加载模型"""
        try:
            # 根据当前模式确定state_dim
            if self.model_mode == 'teacher':
                state_dim = self.config.TEACHER_STATE_DIM
            else:
                state_dim = self.config.STUDENT_STATE_DIM
            
            self.log_message(f"Loading {self.model_mode} model (state_dim={state_dim})...", "info")
            
            # 创建智能体
            self.agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=self.config.ACTION_DIM,
                config=self.config
            )
            
            # 加载模型权重
            self.agent.load(model_path)
            
            self.model_loaded = True
            self.start_btn.setEnabled(True)
            self.model_path_label.setStyleSheet("color: green;")
            
            self.log_message(
                f"{self.model_mode.capitalize()} model loaded successfully: {os.path.basename(model_path)}",
                "info"
            )
            
        except Exception as e:
            self.log_message(f"Failed to load model: {str(e)}", "error")
            self.model_loaded = False
            self.start_btn.setEnabled(False)
            self.model_path_label.setStyleSheet("color: red;")
            self.model_path_label.setText(f"Error: {str(e)[:50]}...")
    
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
    
    def on_video_clicked(self, pos: QPoint):
        """视频点击事件 - 仅设置目标点，不移动球"""
        if not self.manual_mode_radio.isChecked():
            return
        
        if not self.env or not self.env.ball_detected:
            self.log_message("Please wait for ball detection first!", "warning")
            return
        
        # 计算实际坐标
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        
        # 获取pixmap实际大小
        pixmap = self.video_label.pixmap()
        if not pixmap:
            return
        
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        
        # 计算缩放比例
        scale_x = self.config.CAMERA_WIDTH / pixmap_width
        scale_y = self.config.CAMERA_HEIGHT / pixmap_height
        
        # 计算偏移（居中显示）
        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2
        
        # 计算实际图像坐标
        img_x = (pos.x() - offset_x) * scale_x
        img_y = (pos.y() - offset_y) * scale_y
        
        # 转换为相对于中心的坐标
        center_x = self.config.CAMERA_WIDTH / 2
        center_y = self.config.CAMERA_HEIGHT / 2
        
        target_x = img_x - center_x
        target_y = img_y - center_y
        
        # 检查是否在安全区域内
        if abs(target_x) > self.config.SAFETY_RADIUS or abs(target_y) > self.config.SAFETY_RADIUS:
            self.log_message("Target outside safety zone!", "warning")
            return
        
        # ===== 只设置目标位置，不移动Stage =====
        # 保存当前球的位置作为起始点
        start_ball_pos = self.env.ball_pos.copy()
        
        # 设置新目标
        self.env.target_pos = np.array([target_x, target_y])
        self.current_target = self.env.target_pos.copy()
        
        # 重置episode统计（但不重置球的物理位置）
        self.env.step_count = 0
        self.env.episode_reward = 0.0
        self.env.full_trajectory = [start_ball_pos]  # 轨迹从当前位置开始
        self.env.distance_history = []
        self.env.prev_distance = self.env._calc_distance_to_target()
        
        self.episode_count += 1
        
        # 计算距离
        distance_to_target = np.linalg.norm(self.env.target_pos - start_ball_pos)
        
        self.log_message(
            f"Episode {self.episode_count}: New TARGET set at ({target_x:.1f}, {target_y:.1f}) px (RED cross). "
            f"Ball at ({start_ball_pos[0]:.1f}, {start_ball_pos[1]:.1f}) px (GREEN). "
            f"Distance: {distance_to_target:.1f} px",
            "info"
        )
        
        # 如果正在测试，启动执行
        if self.testing:
            # 等待一下让检测更新
            time.sleep(0.1)
            self.execute_episode()

    def start_test(self):
        """开始测试"""
        if not self.model_loaded:
            self.log_message("Please load a model first!", "error")
            return
        
        if not self.env or not self.env.ball_detected:
            self.log_message("Please wait for ball detection!", "warning")
            return
        
        self.testing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        if self.auto_mode_radio.isChecked():
            # 自动模式
            self.run_auto_test()
        else:
            # 手动模式
            self.log_message("Manual mode: Click on video to set targets", "info")
    
    def run_auto_test(self):
        """运行自动测试"""
        num_episodes = self.episodes_spin.value()
        
        self.log_message(f"Starting auto test with {num_episodes} episodes...", "info")
        
        # 创建定时器用于自动测试
        self.auto_test_timer = QTimer()
        self.auto_test_timer.timeout.connect(self.auto_test_step)
        self.auto_test_episodes_left = num_episodes
        self.auto_test_timer.start(100)
    
    def auto_test_step(self):
        """自动测试步骤"""
        if self.auto_test_episodes_left <= 0:
            self.auto_test_timer.stop()
            self.stop_test()
            return
        
        # 保存当前球位置
        current_ball_pos = self.env.ball_pos.copy()
        
        # 生成随机目标
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(50, self.config.SAFETY_RADIUS * 0.85)
        
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        
        # 只设置目标，不移动Stage
        self.env.target_pos = np.array([target_x, target_y])
        self.current_target = self.env.target_pos.copy()
        
        # 重置episode统计
        self.env.step_count = 0
        self.env.episode_reward = 0.0
        self.env.full_trajectory = [current_ball_pos]
        self.env.distance_history = []
        self.env.prev_distance = self.env._calc_distance_to_target()
        
        self.episode_count += 1
        self.auto_test_episodes_left -= 1
        
        distance_to_target = np.linalg.norm(self.env.target_pos - current_ball_pos)
        
        self.log_message(
            f"Auto episode {self.episode_count}: Target at ({target_x:.1f}, {target_y:.1f}), "
            f"Ball at ({current_ball_pos[0]:.1f}, {current_ball_pos[1]:.1f}), "
            f"Distance: {distance_to_target:.1f}px",
            "info"
        )
        
        # 等待检测更新
        time.sleep(0.1)
        
        # 执行episode
        self.execute_episode()
    
    def execute_episode(self):
        """执行一个episode"""
        if not self.env.ball_detected:
            self.log_message("Ball lost!", "error")
            return
        
        state = self.env._get_state()
        done = False
        
        while not done and self.testing:
            QApplication.processEvents()
            
            if not self.env.ball_detected:
                self.log_message("Ball lost during episode!", "error")
                break
            
            # 选择动作
            action = self.agent.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            if 'error' in info:
                self.log_message(f"Error: {info['error']}", "error")
                break
            
            state = next_state
            
            # 更新显示
            self.update_test_stats(info)
            
            time.sleep(0.05)
        
        # Episode结束
        if done:
            success = 1 if info.get('termination') == 'success' else 0
            self.total_rewards.append(self.env.episode_reward)
            self.total_successes.append(success)
            
            self.log_message(
                f"Episode {self.episode_count} finished | "
                f"Reward: {self.env.episode_reward:.2f} | "
                f"Success: {'Yes' if success else 'No'} | "
                f"Distance: {info.get('distance', 0):.2f}px",
                "info"
            )
    
    def update_test_stats(self, info):
        """更新测试统计"""
        self.episode_label.setText(f"Episode: {self.episode_count}")
        self.reward_label.setText(f"Current Reward: {self.env.episode_reward:.2f}")
        self.success_label.setText(f"Success: {'Yes' if info.get('termination') == 'success' else 'No'}")
        self.distance_label.setText(f"Distance: {info.get('distance', 0):.2f}px")
        self.steps_label.setText(f"Steps: {self.env.step_count}")
        
        self.total_episodes_label.setText(f"Total Episodes: {self.episode_count}")
        if len(self.total_rewards) > 0:
            self.avg_reward_label.setText(f"Avg Reward: {np.mean(self.total_rewards):.2f}")
            self.success_rate_label.setText(f"Success Rate: {np.mean(self.total_successes)*100:.1f}%")
    
    def reset_episode(self):
        """重置episode（不移动球，只设置新的随机目标）"""
        if self.env:
            # 保存当前球位置
            current_ball_pos = self.env.ball_pos.copy()
            
            # 随机生成目标
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(50, self.config.SAFETY_RADIUS * 0.85)
            
            target_x = distance * np.cos(angle)
            target_y = distance * np.sin(angle)
            
            # 只设置目标，不移动Stage
            self.env.target_pos = np.array([target_x, target_y])
            
            # 重置统计
            self.env.step_count = 0
            self.env.episode_reward = 0.0
            self.env.full_trajectory = [current_ball_pos]
            self.env.distance_history = []
            self.env.prev_distance = self.env._calc_distance_to_target()
            
            distance_to_target = np.linalg.norm(self.env.target_pos - current_ball_pos)
            
            self.log_message(
                f"Episode reset: Target at ({target_x:.1f}, {target_y:.1f}), "
                f"Ball at ({current_ball_pos[0]:.1f}, {current_ball_pos[1]:.1f}), "
                f"Distance: {distance_to_target:.1f}px",
                "info"
            )
    
    def stop_test(self):
        """停止测试"""
        self.testing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if hasattr(self, 'auto_test_timer'):
            self.auto_test_timer.stop()
        
        # 显示最终统计
        if len(self.total_rewards) > 0:
            self.log_message("=" * 60, "info")
            self.log_message("FINAL RESULTS:", "info")
            self.log_message(f"Total Episodes: {self.episode_count}", "info")
            self.log_message(f"Average Reward: {np.mean(self.total_rewards):.2f}", "info")
            self.log_message(f"Success Rate: {np.mean(self.total_successes)*100:.1f}%", "info")
            self.log_message("=" * 60, "info")
    
    def on_detection_result(self, frame, boxes, scores, class_ids, tracks):
        """处理检测结果"""
        # 绘制检测框（蓝色）
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 更新环境检测
        if self.env:
            self.env.update_detection(frame, boxes, tracks)
            
            # 绘制环境覆盖（包括目标、安全区、轨迹等）
            frame = self.env.render_overlay(frame, show_trajectory=True)
        
        self.update_frame(frame)
    
    def update_frame(self, frame):
        """更新视频帧"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)
    
    def log_message(self, message, msg_type="info"):
        """添加日志"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_camera_error(self, error_msg):
        """相机错误"""
        self.log_message(f"Camera Error: {error_msg}", "error")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.testing = False
        
        if hasattr(self, 'auto_test_timer'):
            self.auto_test_timer.stop()
        
        if self.yolo_thread:
            self.yolo_thread.stop()
            self.yolo_thread.wait()
        
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        if self.image_process_thread:
            self.image_process_thread.stop()
            self.image_process_thread.wait()
        
        if self.env:
            self.env.close_stage()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = TestGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()