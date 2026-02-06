import os # 操作系统接口，用于路径操作等
import jax # Google开发的用于高性能数值计算的库
import jax.numpy as jnp # JAX中的NumPy API
import numpy as np # 用于数值计算的基础库

# 从 franka_env 库中导入各种环境包装器 (Wrappers)
from kinova_env.envs.wrappers import (
    Quat2EulerWrapper, # 将四元数 (Quaternion) 转换为欧拉角 (Euler Angles) 的包装器
    SpacemouseIntervention, # 允许使用 Spacemouse 设备进行人工干预的包装器
    GamepadIntervention,
    MultiCameraBinaryRewardClassifierWrapper, # 基于多摄像头的二元奖励分类器的包装器
    GripperCloseEnv # 用于处理机械臂夹爪关闭动作的包装器
)
# 导入相对坐标系环境
from kinova_env.envs.relative_env import RelativeFrame
# 导入 Franka 环境的默认配置
from kinova_env.envs.kinova_env import DefaultEnvConfig
# 导入 SERL (Scalable, Efficient, and Robust Learning) 库的观察空间包装器
from serl_launcher.serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
# 导入 SERL 库的块化/分块 (Chunking) 包装器
from serl_launcher.serl_launcher.wrappers.chunking import ChunkingWrapper
# 导入奖励分类器加载函数
from serl_launcher.serl_launcher.networks.reward_classifier import load_classifier_func

# 导入实验的默认训练配置
from experiments.config import DefaultTrainingConfig
# 导入 RAM 插入实验特定的环境类
from experiments.apple_put.wrapper import APPLEEnv


# 修改过后的
class EnvConfig(DefaultEnvConfig):
    # --- 基础连接与硬件配置 ---
    # 优先使用 TestConfig 的 IP，若有特定 SERVER_URL 需求请手动微调
    SERVER_IP = "192.168.8.10"
    SERVER_URL = "http://127.0.0.2:5000/"
    
    # 合并摄像头配置：保留 EnvConfig 的详细参数，但索引/数量参考 TestConfig
    REALSENSE_CAMERAS = {
        "side_1": {"index": 1},
        "wrist_1": {"index": 2},
    }

    # 图像裁剪逻辑 (保留原 EnvConfig 特有参数)
    IMAGE_CROP = {
        # "side_1": lambda img: img[200:500, 300:600],
        # "wrist_1": lambda img: img[150:450, 350:1100],
        # "wrist_2": lambda img: img[100:500, 400:900],
    }

    # --- 位姿定义 (以 TestConfig 为准) ---
    # 初始位姿
    RESET_POSE = np.array([
        0.300, -0.000, 0.250,           
        np.deg2rad(180.00), np.deg2rad(0.00), np.deg2rad(90.00)                
    ])
    
    # 目标位姿与抓取位姿
    GRASP_POSE = np.array([0.300, 0.000, 0.200, np.deg2rad(180), 0, np.deg2rad(90)])
    TARGET_POSE = np.array([0.290, 0.213, 0.984, np.deg2rad(180), 0, np.deg2rad(90)])
    # TARGET_POSE = np.array([0.290, 0.213, 0.184, np.deg2rad(180), 0, np.deg2rad(90)])

    # --- 安全限位与控制参数 (以 TestConfig 为准) ---
    ABS_POSE_LIMIT_LOW = np.array([
        0.240, -0.270, 0.020, 
        np.deg2rad(180.00 - 30), np.deg2rad(0.00 - 30), np.deg2rad(90.00 - 30)
    ])
    ABS_POSE_LIMIT_HIGH = np.array([
        0.500, 0.270, 0.500, 
        np.deg2rad(180.00 + 30), np.deg2rad(0.00 + 30), np.deg2rad(90.00 + 30)
    ])

    # 动作缩放 (平移, 旋转, 夹爪)
    ACTION_SCALE = np.array([0.05, 15, 1.0])

    # --- 任务逻辑参数 ---
    MAX_EPISODE_LENGTH = 100
    DISPLAY_IMAGE = True
    
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05


    # RANDOM_XY_RANGE = 0.1
    # RANDOM_RZ_RANGE = 0.3

    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    JOINT_RESET_PERIOD = 5

    RANDOM_RESET = True

    # --- 机械臂底层控制参数 (保留 EnvConfig 特有细节) ---
    # 若 TestConfig 中有简易定义则覆盖，否则保留 EnvConfig 的字典格式
    COMPLIANCE_PARAM = {
        "stiffness": [200, 200, 200, 20, 20, 20], # TestConfig 值
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }

    PRECISION_PARAM = {
        "stiffness": [600, 600, 600, 50, 50, 50], # TestConfig 值
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


# 定义训练配置类，继承自 DefaultTrainingConfig
class TrainConfig(DefaultTrainingConfig):
    # 观察空间中使用的图像键 (即摄像机名称)
    image_keys = ["side_1", "wrist_1"]
    # image_keys = ["side_1"]
    
    # 奖励分类器使用的图像键
    classifier_keys = ["side_1", "wrist_1"]
    # classifier_keys = ["side_1"]
    
    # 观察空间中使用的本体感知 (Proprioceptive) 数据键
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    
    # 缓冲区保存频率
    buffer_period = 1000
    # buffer_period = 50

    # 日志记录频率
    log_period: int = 100

    # 最大训练步数
    # max_steps: int = 600
    
    # 模型检查点保存频率
    # checkpoint_period = 5000
    checkpoint_period = 1000
    
    # 每次更新 (Update) 之间的步数
    steps_per_update = 50
    
    # 编码器类型，此处使用预训练的 ResNet
    encoder_type = "resnet-pretrained"
    
    # 实验设置模式
    setup_mode = "single-arm-fixed-gripper"

    # 获取环境实例的方法
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # 1. 创建基础环境 (RAMEnv)
        env = APPLEEnv(
            fake_env=fake_env, # 是否使用模拟环境
            save_video=save_video, # 是否保存视频
            config=EnvConfig(), # 使用上面定义的配置
        )

        # 2. 堆叠环境包装器 (Wrappers)
        
        # 2.1. GripperCloseEnv: 处理夹爪动作
        env = GripperCloseEnv(env)
        
        # 2.2. SpacemouseIntervention: 启用人工干预 (如果不是模拟环境)
        if not fake_env:
            # env = KeyboardIntervention(env)
            env = GamepadIntervention(env)
            
        # 2.3. RelativeFrame: 将动作转换为相对坐标系下的位移
        env = RelativeFrame(env)
        
        # 2.4. Quat2EulerWrapper: 将机械臂的四元数位姿转换为欧拉角位姿
        env = Quat2EulerWrapper(env)
        
        # 2.5. SERLObsWrapper: 结构化观察空间，确保包含所需的本体感知数据
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        
        # 2.6. ChunkingWrapper: 观察和动作的分块/历史处理
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # 3. 可选：奖励分类器包装器
        if classifier:
            # 3.1. 加载奖励分类器函数 (神经网络模型)
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0), # 随机种子
                sample=env.observation_space.sample(), # 获取观察空间的样本用于初始化
                image_keys=self.classifier_keys, # 使用的图像键
                checkpoint_path=os.path.abspath("classifier_ckpt/"), # 分类器模型的检查点路径
            )

            # 3.2. 定义奖励函数
            def reward_func(obs):
                # Sigmoid 函数
                from datetime import datetime
                import time
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                classifier_output = sigmoid(classifier(obs))
                if classifier_output * 100 > 20:
                    print(f"\nclassifier output: {classifier_output * 100}")  # Debug 输出分类器结果
                # 使用 .item() 确保将 (1,) 维度的数组转为 Python 标量
                # is_successful = (sigmoid(classifier(obs)) > 0
                # .85).item()
                is_successful = (sigmoid(classifier(obs)) > 0.75).item()
                # 同样确保状态判断也是标量
                # 假设 obs['state'] 的形状是 (1, N)，obs['state'][0, 6] 已经是标量了，
                # 但为了保险可以写成 float(obs['state'][0, 6])
                # state_condition = obs['state'][0, 6] > 0.04 # 保证与初始位置的z轴差异>0.4
                state_condition = True # 暂时不加 z 位置限制
                return int(is_successful and state_condition)
                # return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6].item() > 0.04)

            # 3.3. MultiCameraBinaryRewardClassifierWrapper: 使用定义的奖励函数包装环境
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
            
        # 4. 返回最终配置好的环境
        return env

'''
import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.ram_insertion.wrapper import RAMEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 0.06, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
'''