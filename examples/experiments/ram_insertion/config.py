import os # 操作系统接口，用于路径操作等
import jax # Google开发的用于高性能数值计算的库
import jax.numpy as jnp # JAX中的NumPy API
import numpy as np # 用于数值计算的基础库
# 从 franka_env 库中导入各种环境包装器 (Wrappers)
from kinova_env.envs.wrappers import (
    Quat2EulerWrapper, # 将四元数 (Quaternion) 转换为欧拉角 (Euler Angles) 的包装器
    SpacemouseIntervention, # 允许使用 Spacemouse 设备进行人工干预的包装器
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
from experiments.ram_insertion.wrapper import RAMEnv

# 定义环境配置类，继承自 DefaultEnvConfig
class EnvConfig(DefaultEnvConfig):
    # Franka 机械臂控制服务器的 URL
    SERVER_URL = "http://127.0.0.2:5000/"
    
    # 启用 Realsense 深度摄像头的配置
    REALSENSE_CAMERAS = {
        "wrist_1": { # 腕部摄像头 1
            "serial_number": "127122270146",
            "dim": (1280, 720), # 原始分辨率
            "exposure": 40000, # 曝光时间
        },
        "wrist_2": { # 腕部摄像头 2
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    
    # 图像裁剪 (Crop) 区域的定义
    IMAGE_CROP = {
        # 对 wrist_1 摄像头图像进行裁剪，截取 [150:450] 行，[350:1100] 列
        "wrist_1": lambda img: img[150:450, 350:1100],
        # 对 wrist_2 摄像头图像进行裁剪，截取 [100:500] 行，[400:900] 列
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    
    # 目标抓取位姿 (Target Pose)，以 [x, y, z, roll, pitch, yaw] (欧拉角) 形式表示
    # 这是 RAM 插槽的位置
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    
    # 抓取位姿 (Grasp Pose)，可能是机械臂夹爪抓住 RAM 时的位置
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    
    # 重置位姿 (Reset Pose)，机械臂在每轮实验开始时的初始位置
    # 在 TARGET_POSE 的基础上 Z 轴提高 0.05m，pitch 增加 0.05 rad
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    
    # 机械臂末端执行器 (TCP) 绝对位姿的下限限制
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    
    # 机械臂末端执行器 (TCP) 绝对位姿的上限限制
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    
    # 是否启用随机重置
    RANDOM_RESET = True
    
    # 随机重置时，XY 平面上的随机范围 (m)
    RANDOM_XY_RANGE = 0.02
    
    # 随机重置时，Z 轴旋转 (Yaw) 的随机范围 (rad)
    RANDOM_RZ_RANGE = 0.05
    
    # 动作的缩放比例 (Action Scale)
    ACTION_SCALE = (0.01, 0.06, 1) # 可能是 (平移缩放, 旋转缩放, 夹爪缩放)
    
    # 是否显示图像
    DISPLAY_IMAGE = True
    
    # 最大回合长度 (步数)
    MAX_EPISODE_LENGTH = 100
    
    # Franka 机械臂的阻抗控制 (Compliance Control) 参数
    COMPLIANCE_PARAM = {
        # 平移刚度、阻尼等
        "translational_stiffness": 2000,
        "translational_damping": 89,
        # 旋转刚度、阻尼等
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        # ... 还有各种积分项 (Ki) 和剪裁 (clip) 限制，用于约束控制器的输出
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
    
    # Franka 机械臂的精确控制 (Precision Control) 参数
    # 这些参数通常比 COMPLIANCE_PARAM 更严格或具有更大的容忍度，用于不同的控制阶段。
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        # ... 注意剪裁 (clip) 值通常更大，意味着允许更大的运动
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
    image_keys = ["wrist_1", "wrist_2"]
    
    # 奖励分类器使用的图像键
    classifier_keys = ["wrist_1", "wrist_2"]
    
    # 观察空间中使用的本体感知 (Proprioceptive) 数据键
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    
    # 缓冲区保存频率
    buffer_period = 1000
    
    # 模型检查点保存频率
    checkpoint_period = 5000
    
    # 每次更新 (Update) 之间的步数
    steps_per_update = 50
    
    # 编码器类型，此处使用预训练的 ResNet
    encoder_type = "resnet-pretrained"
    
    # 实验设置模式
    setup_mode = "single-arm-fixed-gripper"

    # 获取环境实例的方法
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # 1. 创建基础环境 (RAMEnv)
        env = RAMEnv(
            fake_env=fake_env, # 是否使用模拟环境
            save_video=save_video, # 是否保存视频
            config=EnvConfig(), # 使用上面定义的配置
        )
        
        # 2. 堆叠环境包装器 (Wrappers)
        
        # 2.1. GripperCloseEnv: 处理夹爪动作
        env = GripperCloseEnv(env)
        
        # 2.2. SpacemouseIntervention: 启用人工干预 (如果不是模拟环境)
        if not fake_env:
            env = SpacemouseIntervention(env)
            
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
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                
                # 奖励逻辑：
                # 1. 分类器输出的 Sigmoid 值必须大于 0.85 (高置信度)
                # 2. 机械臂末端执行器的 Z 坐标 (obs['state'][0, 6]) 必须大于 0.04 (防止误判)
                # obs['state'] 包含了本体感知数据，例如 TCP 位姿 (x, y, z, roll, pitch, yaw)
                # 假设 TCP pose 是前六个元素，gripper pose 是第七个元素，Z 坐标在索引 6（如果是 7 维状态）或索引 2 (如果是 6 维位姿)
                # 根据代码的上下文，更可能是 Z 坐标。
                # 'state' 的结构依赖于 SERLObsWrapper 的实现，但通常包含 'tcp_pose'。
                # 假设 obs['state'][0, 6] 是一个与 Z 轴高度相关的状态量，比如夹爪的 Z 位置。
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

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