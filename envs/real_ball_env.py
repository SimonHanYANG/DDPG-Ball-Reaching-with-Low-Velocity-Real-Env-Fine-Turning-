# envs/real_ball_env.py
import numpy as np
import cv2
import time
from typing import Tuple, Dict, Optional
from PyQt6.QtCore import QObject, pyqtSignal
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.real_config import RealConfig
from stage import Stage

class RealBallEnv(QObject):
    """真实环境 - 基于相机和Stage的小球到达任务
    
    核心设计：
    - 所有距离、位置、速度计算都使用 pixel 为单位
    - 只在 Stage 控制时才转换为 μm
    - Ball 位置：相对于图像中心的 pixel 坐标
    """
    
    # 信号
    detection_updated = pyqtSignal(object, list, dict)  # frame, boxes, tracks
    status_message = pyqtSignal(str, str)  # message, type
    frame_updated = pyqtSignal(np.ndarray)  # 新增：每次step后都发送帧
    
    def __init__(self, config: RealConfig, mode='teacher', curriculum_stage: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.mode = mode
        self.curriculum_stage = curriculum_stage
        
        # Stage控制器
        self.stage = None
        
        # 当前图像帧
        self.current_frame = None
        self.current_frame_lock = False  # 添加锁标志
        
        # Ball状态 (pixel坐标，相对于图像中心)
        self.ball_pos = np.zeros(2)  # [x_px, y_px]
        self.ball_vel = np.zeros(2)  # [vx_px/step, vy_px/step]
        self.ball_detected = False
        self.ball_track_id = None
        
        # 目标状态 (pixel坐标，相对于图像中心)
        self.target_pos = np.zeros(2)  # [x_px, y_px]
        
        # 历史记录
        self.trajectory_history = []
        self.full_trajectory = []
        self.distance_history = []
        
        # Episode统计
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_distance = 0.0
        self.prev_action = np.zeros(2)
        self.prev_ball_pos = np.zeros(2)
        
        # Stage位置缓存 (μm坐标)
        self.stage_pos_um = np.zeros(2)
        
        # 等待帧更新的标志
        self.waiting_for_new_frame = False
        self.frame_update_count = 0
    
    def initialize_stage(self):
        """初始化Stage控制器"""
        try:
            self.stage = Stage()
            self.status_message.emit("Stage initialized successfully", "info")
            
            # 获取当前Stage位置
            x, y = self.stage.get_xy_position()
            self.stage_pos_um = np.array([x, y])
            
            return True
        except Exception as e:
            self.status_message.emit(f"Failed to initialize Stage: {str(e)}", "error")
            return False
    
    def close_stage(self):
        """关闭Stage"""
        if self.stage:
            self.stage.close()
            self.stage = None
    
    def pixel_to_um(self, pixel_offset: np.ndarray) -> np.ndarray:
        """像素偏移转换为微米偏移
        
        Args:
            pixel_offset: [dx_px, dy_px]
        
        Returns:
            [dx_um, dy_um]
        """
        dx_um = pixel_offset[0] * self.config.PIXEL_TO_UM_X
        dy_um = pixel_offset[1] * self.config.PIXEL_TO_UM_Y
        return np.array([dx_um, dy_um])
    
    def um_to_pixel(self, um_offset: np.ndarray) -> np.ndarray:
        """微米偏移转换为像素偏移
        
        Args:
            um_offset: [dx_um, dy_um]
        
        Returns:
            [dx_px, dy_px]
        """
        dx_px = um_offset[0] / self.config.PIXEL_TO_UM_X
        dy_px = um_offset[1] / self.config.PIXEL_TO_UM_Y
        return np.array([dx_px, dy_px])
    
    def update_detection(self, frame: np.ndarray, boxes: list, tracks: dict):
        """更新检测和跟踪结果
        
        Args:
            frame: 当前图像帧
            boxes: 检测框列表 [[x1,y1,x2,y2], ...]
            tracks: 跟踪结果字典 {track_id: {'trajectory': [...], 'color': (r,g,b)}}
        """
        # 更新当前帧
        self.current_frame = frame.copy()
        self.frame_update_count += 1
        
        # 图像中心 (pixel)
        img_center_x = self.config.CAMERA_WIDTH // 2
        img_center_y = self.config.CAMERA_HEIGHT // 2
        
        # 更新Ball位置
        self.prev_ball_pos = self.ball_pos.copy()
        
        if len(boxes) > 0:
            # 选择最接近上一次位置的检测框
            if self.ball_detected and np.linalg.norm(self.ball_pos) > 0:
                distances = []
                for box in boxes:
                    cx = (box[0] + box[2]) / 2 - img_center_x
                    cy = (box[1] + box[3]) / 2 - img_center_y
                    dist = np.linalg.norm(np.array([cx, cy]) - self.ball_pos)
                    distances.append(dist)
                
                best_idx = np.argmin(distances)
                box = boxes[best_idx]
            else:
                # 首次检测，选择第一个
                box = boxes[0]
            
            # 更新位置 (转换为相对于图像中心的坐标, pixel)
            cx = (box[0] + box[2]) / 2 - img_center_x
            cy = (box[1] + box[3]) / 2 - img_center_y
            self.ball_pos = np.array([cx, cy])
            self.ball_detected = True
            
            # 计算速度 (pixel/step)
            if np.linalg.norm(self.prev_ball_pos) > 0:
                self.ball_vel = (self.ball_pos - self.prev_ball_pos) / self.config.DT
            
            # 更新轨迹
            self.trajectory_history.append(self.ball_pos.copy())
            if len(self.trajectory_history) > 3:
                self.trajectory_history.pop(0)
            
            self.full_trajectory.append(self.ball_pos.copy())
        else:
            self.ball_detected = False
            self.status_message.emit("Ball not detected!", "warning")
        
        # 发送帧更新信号
        if self.waiting_for_new_frame:
            self.waiting_for_new_frame = False
    
    def reset(self, manual_init: bool = False) -> np.ndarray:
        """重置环境
    
        Args:
            manual_init: 已废弃，保留参数用于向后兼容
        
        Returns:
            初始状态
        """
        # 生成目标位置 (pixel, 相对于图像中心)
        angle = np.random.uniform(0, 2 * np.pi)
        target_distance = np.random.uniform(50, self.config.SAFETY_RADIUS * 0.85)
        
        self.target_pos = np.array([
            target_distance * np.cos(angle),
            target_distance * np.sin(angle)
        ])
        
        # 自动随机初始化小球位置
        init_angle = np.random.uniform(0, 2 * np.pi)
        
        # 根据课程阶段或随机生成初始距离
        if self.curriculum_stage:
            max_dist = self.curriculum_stage.get('max_init_distance', self.config.SAFETY_RADIUS * 0.85)
            min_dist = self.curriculum_stage.get('min_init_distance', 50)
        else:
            max_dist = self.config.SAFETY_RADIUS * 0.85
            min_dist = 50
        
        init_distance = np.random.uniform(min_dist, max_dist)
        
        # 计算初始位置 (pixel)
        init_pos = np.array([
            init_distance * np.cos(init_angle),
            init_distance * np.sin(init_angle)
        ])
        
        # 如果当前小球已检测，计算需要移动的偏移
        if self.ball_detected:
            offset_px = init_pos - self.ball_pos
        else:
            # 如果未检测到小球，假设从中心开始
            offset_px = init_pos
        
        # 转换为微米并移动Stage
        offset_um = self.pixel_to_um(offset_px)
        
        try:
            self.stage.move_xy_relative(offset_um[0], offset_um[1])
            # 更新Stage位置缓存
            x, y = self.stage.get_xy_position()
            self.stage_pos_um = np.array([x, y])
            
            self.status_message.emit(
                f"Ball initialized at ({init_pos[0]:.1f}, {init_pos[1]:.1f}) px, "
                f"Target at ({self.target_pos[0]:.1f}, {self.target_pos[1]:.1f}) px",
                "info"
            )
        except Exception as e:
            self.status_message.emit(f"Failed to initialize ball position: {str(e)}", "error")
        
        # 等待Stage移动完成并获取新的检测结果
        import time
        time.sleep(0.3)  # 给Stage时间移动
        
        # 清空历史
        self.trajectory_history = [self.ball_pos.copy()] if self.ball_detected else []
        self.full_trajectory = [self.ball_pos.copy()] if self.ball_detected else []
        self.distance_history = []
        
        self.prev_distance = self._calc_distance_to_target() if self.ball_detected else 0
        self.prev_action = np.zeros(2)
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步
        
        Args:
            action: 动作 (pixel/step), 范围 [-1, 1], 需要缩放
        
        Returns:
            next_state, reward, done, info
        """
        if not self.ball_detected:
            return self._get_state(), -100.0, False, {'error': 'ball_not_detected'}
        
        # 动作归一化并缩放 (pixel/step)
        action_px = np.clip(action, -1, 1) * self.config.MAX_ACTION
        
        # 转换为微米
        action_um = self.pixel_to_um(action_px)
        
        # 执行Stage移动
        try:
            self.stage.move_xy_relative(action_um[0], action_um[1])
            
            # 更新Stage位置缓存
            x, y = self.stage.get_xy_position()
            self.stage_pos_um = np.array([x, y])
            
        except Exception as e:
            self.status_message.emit(f"Stage move failed: {str(e)}", "error")
            return self._get_state(), -100.0, True, {'error': 'stage_error'}
        
        # 等待新帧到来（最多等待0.5秒）
        self.waiting_for_new_frame = True
        start_frame_count = self.frame_update_count
        timeout = 0.5
        start_time = time.time()
        
        while self.waiting_for_new_frame and (time.time() - start_time) < timeout:
            time.sleep(0.01)  # 短暂休眠，等待帧更新
            if self.frame_update_count > start_frame_count:
                self.waiting_for_new_frame = False
                break
        
        # 发送当前帧用于显示
        if self.current_frame is not None:
            overlay_frame = self.render_overlay(self.current_frame.copy(), show_trajectory=True)
            self.frame_updated.emit(overlay_frame)
        
        # 计算状态 (所有单位: pixel)
        distance = self._calc_distance_to_target()
        velocity_mag = np.linalg.norm(self.ball_vel)
        
        # 距离变化 (pixel)
        delta_distance = distance - self.prev_distance
        self.distance_history.append(delta_distance)
        if len(self.distance_history) > 20:
            self.distance_history.pop(0)
        
        # 检查终止条件
        done = False
        info = {}
        
        # 1. 越界
        if self._is_out_of_bounds():
            done = True
            info['termination'] = 'boundary_collision'
            reward = -200.0
            self.status_message.emit("Ball out of bounds!", "error")
        
        # 2. 成功到达
        elif (distance <= self.config.TARGET_TOLERANCE and 
              velocity_mag <= self.config.TARGET_VELOCITY_THRESHOLD):
            done = True
            info['termination'] = 'success'
            
            base_reward = 300.0
            speed_bonus = 100.0 * (1.0 - velocity_mag / self.config.TARGET_VELOCITY_THRESHOLD)
            efficiency_bonus = 100.0 * (1.0 - self.step_count / self.config.MAX_STEPS)
            reward = base_reward + speed_bonus + efficiency_bonus
            
            self.status_message.emit(f"Success! Reward: {reward:.1f}", "success")
        
        # 3. 超时
        elif self.step_count >= self.config.MAX_STEPS:
            done = True
            info['termination'] = 'timeout'
            reward = -50.0 - distance * 0.5
            self.status_message.emit("Episode timeout", "warning")
        
        else:
            # 计算step reward
            reward = self._calculate_step_reward(action_px, distance, velocity_mag, delta_distance)
        
        # 更新状态
        self.prev_distance = distance
        self.prev_action = action_px
        self.step_count += 1
        self.episode_reward += reward
        
        # 统计信息
        info.update({
            'distance': distance,
            'velocity': velocity_mag,
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'delta_distance': delta_distance,
            'approach_rate': self._calc_approach_rate()
        })
        
        return self._get_state(), reward, done, info
    
    def _calculate_step_reward(self, action: np.ndarray, distance: float, 
                               velocity: float, delta_distance: float) -> float:
        """计算每步奖励 - 所有单位: pixel"""
        reward = 0.0
        
        # 距离变化奖励
        if delta_distance < 0:
            approach_weight = 1.0 + (self.config.SAFETY_RADIUS - distance) / self.config.SAFETY_RADIUS
            reward += 150.0 * abs(delta_distance) * approach_weight
            
            if self._count_consecutive_approach() >= 5:
                reward += 20.0
        else:
            retreat_weight = 1.0 + (self.config.SAFETY_RADIUS - distance) / self.config.SAFETY_RADIUS
            reward -= 200.0 * delta_distance * retreat_weight
            
            if self._count_consecutive_retreat() >= 3:
                reward -= 30.0
        
        # 速度控制奖励 (pixel/step)
        if distance < 30:
            target_vel = 3.0
            if velocity <= target_vel:
                reward += 30.0 * (1.0 - velocity / target_vel)
            else:
                reward -= 40.0 * (velocity - target_vel) / self.config.MAX_ACTION
        elif distance < 100:
            target_vel = 8.0
            if velocity > target_vel:
                reward -= 20.0 * (velocity - target_vel) / self.config.MAX_ACTION
        
        # 方向奖励
        to_target = self.target_pos - self.ball_pos
        to_target_norm = np.linalg.norm(to_target)
        
        if to_target_norm > 1e-6:
            ideal_direction = to_target / to_target_norm
            actual_movement = self.ball_pos - self.prev_ball_pos
            actual_movement_norm = np.linalg.norm(actual_movement)
            
            if actual_movement_norm > 1e-6:
                actual_direction = actual_movement / actual_movement_norm
                direction_alignment = np.dot(ideal_direction, actual_direction)
                
                if direction_alignment > 0.7:
                    reward += 15.0 * direction_alignment
                elif direction_alignment < -0.3:
                    reward -= 25.0 * abs(direction_alignment)
        
        # 边界安全 (pixel)
        boundary_dist = self._calc_distance_to_boundary()
        if boundary_dist < 80:
            boundary_penalty = 50.0 * ((80 - boundary_dist) / 80) ** 2
            reward -= boundary_penalty
        
        # 动作效率
        action_mag = np.linalg.norm(action)
        
        if action_mag < 0.2 * self.config.MAX_ACTION and distance > 20:
            reward -= 15.0
        
        if distance < 30 and action_mag > 0.6 * self.config.MAX_ACTION:
            reward -= 10.0
        
        # Student额外惩罚
        if self.mode == 'student':
            action_change = np.linalg.norm(action - self.prev_action)
            if action_change > 0.7 * self.config.MAX_ACTION:
                reward -= 8.0 * (action_change / self.config.MAX_ACTION)
        
        # 距离阈值奖励 (pixel)
        if distance < 50 and self.prev_distance >= 50:
            reward += 40.0
        elif distance < 20 and self.prev_distance >= 20:
            reward += 60.0
        elif distance < 10 and self.prev_distance >= 10:
            reward += 80.0
        
        return reward
    
    def _get_state(self) -> np.ndarray:
        """获取状态 - 所有单位: pixel"""
        if not self.ball_detected:
            if self.mode == 'teacher':
                return np.zeros(self.config.TEACHER_STATE_DIM, dtype=np.float32)
            else:
                return np.zeros(self.config.STUDENT_STATE_DIM, dtype=np.float32)
        
        distance = self._calc_distance_to_target()
        boundary_dist = self._calc_distance_to_boundary()
        velocity_mag = np.linalg.norm(self.ball_vel)
        
        to_target = self.target_pos - self.ball_pos
        to_target_norm = np.linalg.norm(to_target)
        
        if velocity_mag > 1e-6 and to_target_norm > 1e-6:
            cos_angle = np.dot(self.ball_vel, to_target) / (velocity_mag * to_target_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
        else:
            cos_angle = 0.0
        
        delta_d = distance - self.prev_distance
        action_norm = self.prev_action / self.config.MAX_ACTION
        approach_rate = self._calc_approach_rate()
        
        teacher_state = np.array([
            distance / self.config.SAFETY_RADIUS,
            boundary_dist / self.config.SAFETY_RADIUS,
            velocity_mag / self.config.MAX_ACTION,
            cos_angle,
            delta_d / self.config.SAFETY_RADIUS,
            to_target[0] / self.config.SAFETY_RADIUS,
            to_target[1] / self.config.SAFETY_RADIUS,
            action_norm[0],
            action_norm[1],
            float(self.step_count) / self.config.MAX_STEPS,
            approach_rate,
            float(self._count_consecutive_approach()) / 10.0
        ], dtype=np.float32)
        
        if self.mode == 'teacher':
            return teacher_state
        
        # Student额外状态
        ball_pos_norm = self.ball_pos / self.config.SAFETY_RADIUS
        target_pos_norm = self.target_pos / self.config.SAFETY_RADIUS
        
        trajectory_features = np.zeros(4)
        if len(self.trajectory_history) >= 2:
            delta = (self.trajectory_history[-1] - self.trajectory_history[-2]) / self.config.SAFETY_RADIUS
            trajectory_features[0:2] = delta
        if len(self.trajectory_history) >= 3:
            delta = (self.trajectory_history[-2] - self.trajectory_history[-3]) / self.config.SAFETY_RADIUS
            trajectory_features[2:4] = delta
        
        perception_conf = 1.0
        
        student_extra = np.array([
            ball_pos_norm[0],
            ball_pos_norm[1],
            target_pos_norm[0],
            target_pos_norm[1],
            trajectory_features[0],
            trajectory_features[1],
            trajectory_features[2],
            trajectory_features[3],
            perception_conf,
            float(self._count_consecutive_retreat()) / 10.0
        ], dtype=np.float32)
        
        return np.concatenate([teacher_state, student_extra])
    
    def _calc_distance_to_target(self) -> float:
        """计算到目标的距离 (pixel)"""
        return np.linalg.norm(self.ball_pos - self.target_pos)
    
    def _calc_distance_to_boundary(self) -> float:
        """计算到边界的距离 (pixel)"""
        return self.config.SAFETY_RADIUS - np.linalg.norm(self.ball_pos)
    
    def _is_out_of_bounds(self) -> bool:
        """检查是否越界"""
        return np.linalg.norm(self.ball_pos) > self.config.SAFETY_RADIUS
    
    def _calc_approach_rate(self) -> float:
        """计算接近率"""
        if len(self.distance_history) < 2:
            return 0.0
        recent = self.distance_history[-10:]
        approach_steps = [d for d in recent if d < 0]
        return len(approach_steps) / len(recent) if len(recent) > 0 else 0.0
    
    def _count_consecutive_approach(self) -> int:
        """计算连续接近步数"""
        count = 0
        for delta in reversed(self.distance_history):
            if delta < 0:
                count += 1
            else:
                break
        return count
    
    def _count_consecutive_retreat(self) -> int:
        """计算连续远离步数"""
        count = 0
        for delta in reversed(self.distance_history):
            if delta > 0:
                count += 1
            else:
                break
        return count
    
    def render_overlay(self, frame: np.ndarray, show_trajectory: bool = True) -> np.ndarray:
            """在帧上绘制覆盖信息 - 所有单位: pixel
            
            Args:
                frame: 原始相机图像
                show_trajectory: 是否显示轨迹
            
            Returns:
                带覆盖信息的图像
            """
            img = frame.copy()
            
            center_x = self.config.CAMERA_WIDTH // 2
            center_y = self.config.CAMERA_HEIGHT // 2
            
            def to_screen(pos):
                """将相对坐标转换为屏幕坐标"""
                x = int(center_x + pos[0])
                y = int(center_y + pos[1])
                return (x, y)
            
            # 绘制安全区 (蓝色边框)
            safety_r = int(self.config.SAFETY_RADIUS)
            cv2.rectangle(img,
                        (center_x - safety_r, center_y - safety_r),
                        (center_x + safety_r, center_y + safety_r),
                        (255, 0, 0), 3)
            cv2.putText(img, 'Safety Zone (800x800px)', 
                        (center_x - safety_r + 10, center_y - safety_r + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 绘制目标 (红色十字) - 修改颜色
            target_screen = to_screen(self.target_pos)
            cv2.circle(img, target_screen, int(self.config.TARGET_TOLERANCE), 
                    (0, 0, 255), 2, cv2.LINE_AA)  # 改为红色
            cv2.line(img, (target_screen[0] - 15, target_screen[1]), 
                    (target_screen[0] + 15, target_screen[1]), (0, 0, 255), 3)  # 改为红色
            cv2.line(img, (target_screen[0], target_screen[1] - 15), 
                    (target_screen[0], target_screen[1] + 15), (0, 0, 255), 3)  # 改为红色
            cv2.putText(img, 'TARGET', 
                        (target_screen[0] + 20, target_screen[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 改为红色
            
            # 绘制轨迹（绿色）- 保持绿色
            if show_trajectory and len(self.full_trajectory) > 1:
                for i in range(len(self.full_trajectory) - 1):
                    pt1 = to_screen(self.full_trajectory[i])
                    pt2 = to_screen(self.full_trajectory[i + 1])
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 绘制Ball位置（黄色圆点）
            if self.ball_detected:
                ball_screen = to_screen(self.ball_pos)
                cv2.circle(img, ball_screen, 8, (0, 255, 255), -1, cv2.LINE_AA)
                
                # 绘制速度向量 (红色箭头) - 修改颜色
                vel_mag = np.linalg.norm(self.ball_vel)
                if vel_mag > 0.5:
                    vel_end = self.ball_pos + self.ball_vel * 5.0
                    vel_screen = to_screen(vel_end)
                    cv2.arrowedLine(img, ball_screen, vel_screen, 
                                    (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.2)  # 改为红色
            
            # 信息面板 (所有单位: pixel)
            distance = self._calc_distance_to_target() if self.ball_detected else 0
            velocity_mag = np.linalg.norm(self.ball_vel) if self.ball_detected else 0
            
            info_text = [
                f"Distance: {distance:.1f}px",
                f"Velocity: {velocity_mag:.1f}px/step",
                f"Boundary Dist: {self._calc_distance_to_boundary():.1f}px" if self.ball_detected else "Boundary Dist: -",
                f"Step: {self.step_count}/{self.config.MAX_STEPS}",
                f"Reward: {self.episode_reward:.1f}",
                f"Ball Detected: {'YES' if self.ball_detected else 'NO'}"
            ]
            
            y_pos = 30
            for i, text in enumerate(info_text):
                cv2.putText(img, text, (10, y_pos + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, text, (10, y_pos + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            return img
    
    @property
    def state_dim(self) -> int:
        return self.config.TEACHER_STATE_DIM if self.mode == 'teacher' else self.config.STUDENT_STATE_DIM
    
    @property
    def action_dim(self) -> int:
        return self.config.ACTION_DIM