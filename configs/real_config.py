# configs/real_config.py
import torch
import numpy as np

class RealConfig:
    """真实环境配置 - 所有距离计算使用 pixel，Stage 控制使用 μm"""
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 相机参数
    CAMERA_WIDTH = 1600
    CAMERA_HEIGHT = 1200
    FOV_WIDTH = 1600.0
    FOV_HEIGHT = 1200.0
    
    # 安全区域 (以视野中心为中心的 800x800 pixel)
    SAFETY_WIDTH = 800.0
    SAFETY_HEIGHT = 800.0
    SAFETY_RADIUS = min(SAFETY_WIDTH, SAFETY_HEIGHT) / 2.0  # 400px
    
    # 目标容忍度和速度阈值 (pixel)
    TARGET_TOLERANCE = 5.0
    TARGET_VELOCITY_THRESHOLD = 8.0
    MAX_STEPS = 300
    DT = 1.0
    
    # 动作空间 - Stage移动速度 (pixel/step)
    MAX_ACTION = 15.0
    ACTION_DIM = 2
    
    # YOLO检测参数
    YOLO_ENGINE_PATH = "weights/yolov8-weights/best.engine"
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    DETECTION_CLASS = 'ball_free'  # 检测自由球
    
    # JPDAF跟踪参数
    JPDAF_PROCESS_NOISE = 20.0
    JPDAF_MEASURE_NOISE = 2.0
    JPDAF_DETECT_PROB = 0.7
    JPDAF_GATE_PROB = 0.95
    
    # Stage控制参数
    STAGE_STEP_SIZE = 5.0  # 手动控制时的步长 (pixel)
    STAGE_MAX_VELOCITY = 50.0  # Stage最大速度 (μm/s)
    STAGE_ACCELERATION = 100.0  # Stage加速度 (μm/s²)
    
    # 坐标转换参数 (pixel to μm)
    PIXEL_TO_UM_X = 3600.0 / 1600.0
    PIXEL_TO_UM_Y = 2700.0 / 1200.0
    
    # Few-Shot 课程学习参数 (所有距离单位: pixel)
    CURRICULUM_ENABLED = True
    CURRICULUM_STAGES = [
        {
            'name': 'Stage 1: Warm-up (20-60px)',
            'max_init_distance': 60,
            'min_init_distance': 20,
            'success_threshold': 0.90,
            'min_episodes': 300,
            'max_episodes': 800,
            'patience': 100,
            'unfreeze_layers': ['actor.fc3', 'critic.fc3'],  # 只解冻输出层
            'lr_actor': 1e-4,
            'lr_critic': 3e-4
        },
        {
            'name': 'Stage 2: Easy (40-120px)',
            'max_init_distance': 120,
            'min_init_distance': 40,
            'success_threshold': 0.80,
            'min_episodes': 500,
            'max_episodes': 1200,
            'patience': 150,
            'unfreeze_layers': ['actor.fc2', 'actor.fc3', 'critic.fc2', 'critic.fc3'],  # 解冻后两层
            'lr_actor': 8e-5,
            'lr_critic': 2e-4
        },
        {
            'name': 'Stage 3: Medium (80-200px)',
            'max_init_distance': 200,
            'min_init_distance': 80,
            'success_threshold': 0.70,
            'min_episodes': 800,
            'max_episodes': 1800,
            'patience': 200,
            'unfreeze_layers': None,  # 全部解冻
            'lr_actor': 5e-5,
            'lr_critic': 1.5e-4
        },
        {
            'name': 'Stage 4: Hard (120-300px)',
            'max_init_distance': 300,
            'min_init_distance': 120,
            'success_threshold': 0.60,
            'min_episodes': 1000,
            'max_episodes': 2500,
            'patience': 250,
            'unfreeze_layers': None,  # 全部解冻
            'lr_actor': 3e-5,
            'lr_critic': 1e-4
        },
        {
            'name': 'Stage 5: Expert (150-400px)',
            'max_init_distance': 400,
            'min_init_distance': 150,
            'success_threshold': 0.55,
            'min_episodes': 1200,
            'max_episodes': 3000,
            'patience': 300,
            'unfreeze_layers': None,
            'lr_actor': 2e-5,
            'lr_critic': 8e-5
        }
    ]
    
    CURRICULUM_SUCCESS_WINDOW = 100
    
    # 智能早停参数
    EARLY_STOPPING_ENABLED = True
    EARLY_STOPPING_PATIENCE = 500  # 全局早停耐心
    EARLY_STOPPING_MIN_DELTA = 0.01  # 最小改进阈值
    EARLY_STOPPING_CHECK_INTERVAL = 50  # 检查间隔
    
    # 学习率调度参数
    LR_SCHEDULER_ENABLED = True
    LR_SCHEDULER_FACTOR = 0.5  # 衰减因子
    LR_SCHEDULER_PATIENCE = 200  # 学习率衰减耐心
    LR_SCHEDULER_MIN_LR_ACTOR = 1e-6
    LR_SCHEDULER_MIN_LR_CRITIC = 3e-6
    
    # 优先经验回放参数
    PRIORITIZED_REPLAY_ENABLED = True
    PRIORITY_ALPHA = 0.6  # 优先级指数
    PRIORITY_BETA_START = 0.4  # 重要性采样起始值
    PRIORITY_BETA_END = 1.0  # 重要性采样结束值
    PRIORITY_BETA_FRAMES = 100000  # Beta退火步数
    PRIORITY_EPSILON = 1e-6  # 防止零优先级
    
    # 近期经验优先参数
    RECENT_BUFFER_SIZE = 10000  # 近期经验缓冲区大小
    RECENT_SAMPLE_RATIO = 0.5  # 从近期经验采样的比例
    
    # 状态空间维度
    TEACHER_STATE_DIM = 12
    STUDENT_STATE_DIM = 22
    
    # DDPG超参数
    GAMMA = 0.98
    TAU = 0.005
    LR_ACTOR = 3e-4  # 默认学习率（会被课程覆盖）
    LR_CRITIC = 1e-3
    
    # OU噪声参数（自适应）
    OU_THETA = 0.15
    OU_SIGMA_START = 0.3  # 初始噪声
    OU_SIGMA_END = 0.05  # 最终噪声
    OU_SIGMA_DECAY = 0.9999  # 噪声衰减率
    OU_MU = 0.0
    
    # 经验回放
    BUFFER_SIZE = 200000
    BATCH_SIZE = 128
    
    # 网络结构
    HIDDEN_DIM = 256
    
    # 训练参数
    TOTAL_EPISODES = 20000  # 总体限制（通常会提前停止）
    STUDENT_EPISODES = 10000
    WARMUP_STEPS = 1000  # 减少预热步数
    UPDATE_EVERY = 1
    SAVE_EVERY = 100
    
    # 模仿学习参数
    IMITATION_START = 5000
    IMITATION_END = 8000
    IMITATION_WEIGHT_START = 0.8
    IMITATION_WEIGHT_END = 0.1
    
    # 日志路径
    LOG_DIR = "logs_real"
    CHECKPOINT_DIR = "checkpoints_real"
    
    # 可视化参数
    VIS_WIDTH = 1600
    VIS_HEIGHT = 1200
    VIS_FPS = 30
    
    # 安全检查 (pixel)
    BOUNDARY_MARGIN = 50
    EMERGENCY_STOP_DISTANCE = 30