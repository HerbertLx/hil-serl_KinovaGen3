# %%
import os
import sys

# --- 路径设置 ---
# 1. 设置 hil-serl 包路径
target_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl"
if os.path.exists(target_path) and target_path not in sys.path:
    sys.path.insert(0, target_path)
    print(f"✅ 成功添加 hil-serl 路径: {target_path}")

# 2. 设置 kinova_manage.py 所在路径 (根据你的描述)
manager_path = "/home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control"
if os.path.exists(manager_path) and manager_path not in sys.path:
    sys.path.insert(0, manager_path)
    print(f"✅ 成功添加 KinovaManager 路径: {manager_path}")

from kinova_manage import KinovaManager



# %%
import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from kinova_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from kinova_env.spacemouse.keyboard_expert import KeyBoardExpert
from kinova_env.spacemouse.gamepad_expert import GamepadExpert
from kinova_env.spacemouse.visionpro_expert import VisionProExpert

import requests
from scipy.spatial.transform import Rotation as R
from kinova_env.envs.kinova_env import KinovaEnv
from typing import List

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        # print(f"\n In MultiCameraBinaryRewardClassifierWrapper, before, info['succeed'] = {info['succeed']}")  # --- DEBUG ---
        from datetime import datetime
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
        # print(f"\n In MultiCameraBinaryRewardClassifierWrapper, after, info['succeed'] = {info['succeed']}")  # --- DEBUG ---
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        # print(f"\nIn Quat2EulerWrapper, before, tcp_pose = {tcp_pose}")
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        # print(f"In Quat2EulerWrapper, after, tcp_pose = {tcp_pose}")
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class SpacemouseIntervention(gym.ActionWrapper):
    """
    SpacemouseIntervention 类:
    一个 Gymnasium 动作包装器，用于实现人类专家通过 SpaceMouse 对策略动作的实时干预。
    
    类逻辑:
    1. 实时监听 SpaceMouse 的输入。
    2. 如果检测到 SpaceMouse 有明显的物理移动或按键操作，则判定为“人类干预”。
    3. 在干预发生时，覆盖掉神经网络（Policy）输出的动作，改用专家的动作。
    4. 将干预状态记录在 info 字典中，方便后续数据记录（如行为克隆或数据采样）。
    """

    def __init__(self, env, action_indices=None):
        """
        初始化干预包装器。
        
        :param env: 待包装的 Gymnasium 环境。
        :param action_indices: list[int] 或 None。可选参数，用于指定只允许 SpaceMouse 控制哪些维度的动作。
        """
        super().__init__(env)

        # 检查环境动作空间，判定是否包含夹爪控制位（通常第 7 位是夹爪）
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        # 初始化专家设备驱动（SpaceMouse）
        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False  # 记录鼠标左右键状态
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        处理动作覆盖逻辑（干预核心）。

        输入参数:
        :param action: np.ndarray, 原始策略（Policy）输出的归一化动作信号。

        输出参数:
        :return (new_action, intervened): 
            - new_action: 如果发生干预，返回专家动作；否则返回原始动作。
            - intervened: bool, 标志位，True 表示当前帧发生了人工干预。

        函数逻辑:
        1. 获取 SpaceMouse 的 6 自由度位移信号 (expert_a) 和按键状态 (buttons)。
        2. 判断位移信号的模长，如果大于 0.001，则认为人正在移动鼠标，触发干预。
        3. 处理夹爪逻辑：
           - 如果按下左键，生成一个 [-1, -0.9] 的闭合信号。
           - 如果按下右键，生成一个 [0.9, 1] 的开启信号。
        4. 如果设置了 action_indices，则只保留指定维度的专家输入，其余维度置零。
        5. 返回最终确定的动作和干预状态。
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False
        
        # 判定是否有物理位移干预
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        # 处理夹爪干预逻辑
        if self.gripper_enabled:
            if self.left:  # 闭合夹爪逻辑
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # 开启夹爪逻辑
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            # 将 6 维位移和 1 维夹爪拼成 7 维动作
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        # 如果指定了动作索引映射（如只用鼠标控制平移，不控制旋转）
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # 判定最终返回哪方的指令
        if intervened:
            return expert_a, True

        return action, False

    def step(self, action: np.ndarray):
        """
        执行环境步进，并注入干预信息。

        输入参数:
        :param action: np.ndarray, 策略输出的原始动作。

        输出参数:
        :return (obs, rew, done, truncated, info): 标准 Gymnasium 返回值。
            - info 中额外包含：
                - intervene_action: 实际执行的人类干预动作（仅在干预时存在）。
                - left/right: 鼠标按键状态。
        """
        # 调用内部 action 函数判定是否需要替换动作
        new_action, replaced = self.action(action)

        # 执行环境物理步进
        obs, rew, done, truncated, info = self.env.step(new_action)

        # 记录辅助信息，用于后续的 IL (模仿学习) 采样
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        
        return obs, rew, done, truncated, info

class KeyboardIntervention(gym.ActionWrapper):
    """
    KeyboardIntervention 类:
    一个 Gymnasium 动作包装器，用于实现人类专家通过 Keyboard（键盘）对策略动作的实时干预。
    
    类逻辑:
    1. 实时监听 KeyboardExpert 的输入信号。
    2. 如果检测到键盘有按键按下（位移键或功能键），则判定为“人类干预”。
    3. 在干预发生时，覆盖掉神经网络（Policy）输出的动作，改用键盘模拟的专家动作。
    4. 将干预状态记录在 info 字典中，方便后续模仿学习数据采样。
    """

    def __init__(self, env, action_indices=None):
        """
        初始化键盘干预包装器。
        
        :param env: 待包装的 Gymnasium 环境。
        :param action_indices: list[int] 或 None。可选参数，用于指定只允许键盘控制哪些维度的动作（如仅控制位置）。
        """
        super().__init__(env)

        # 检查环境动作空间，判定是否包含夹爪控制位（通常第 7 位是夹爪）
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        # 实例化键盘驱动
        self.expert = KeyBoardExpert()
        
        # 内部状态：记录当前是否有开/合动作
        self.left, self.right = False, False  
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        处理动作覆盖逻辑（干预核心）。

        输入参数:
        :param action: np.ndarray, 原始策略（Policy）输出的动作信号。

        输出参数:
        :return (new_action, intervened): 
            - new_action: 最终执行的动作（专家动作或原始动作）。
            - intervened: bool, 当前帧是否发生了键盘干预。

        函数逻辑:
        1. 从 KeyboardExpert 获取 6 自由度增量和按键状态。
        2. 判定位移：如果键盘对应的 XYZ/旋转轴有输出，触发干预。
        3. 判定夹爪：
           - 对应 KeyboardExpert 中的按钮映射（如 F1/F2）。
           - 左键（buttons[0]）触发闭合 [-1, -0.9]；右键（buttons[1]）触发开启 [0.9, 1]。
        4. 组装并过滤动作维度。
        """
        # 获取键盘最新的动作向量 [6维] 和 按钮列表 [4维]
        expert_a, buttons = self.expert.get_action()
        
        # 修正：显式取前两个按钮，防止维度 unpacking 错误
        self.left = bool(buttons[0])
        self.right = bool(buttons[1])
        
        intervened = False
        
        # 1. 判定位移/旋转干预 (只要有按键被按下，norm 就会大于 0)
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        # 2. 判定夹爪干预逻辑
        if self.gripper_enabled:
            if self.left:  # 闭合夹爪逻辑
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # 开启夹爪逻辑
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            
            # 将键盘 6 维动作和 1 维夹爪拼成完整的动作空间维度 (通常为 7)
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        # 3. 维度过滤 (如果只想让键盘干预特定轴)
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # 4. 返回选择后的动作
        if intervened:
            return expert_a, True

        return action, False

    def step(self, action: np.ndarray):
        """
        执行环境步进，并在 info 中记录干预细节。

        输入参数:
        :param action: np.ndarray, 神经网络输出的原始动作。

        返回:
        - obs, rew, done, truncated, info
        """
        # 核心逻辑：获取可能被键盘覆盖后的新动作
        new_action, replaced = self.action(action)

        # 执行物理步进
        obs, rew, done, truncated, info = self.env.step(new_action)

        # 记录关键信息用于数据分析
        if replaced:
            info["intervene_action"] = new_action
        
        info["intervened"] = replaced
        info["left"] = self.left
        info["right"] = self.right
        
        return obs, rew, done, truncated, info

class GamepadIntervention(gym.ActionWrapper):
    """
    GamepadIntervention 类:
    使用 Xbox 手柄实现对机器人策略动作的实时干预。
    """

    def __init__(self, env, action_indices=None):
        """
        :param env: 待包装的环境
        :param action_indices: 可选，指定手柄只控制哪些维度（如 [0,1,2,6] 仅控制位置和夹爪）
        """
        super().__init__(env)

        # 判定动作空间是否包含夹爪 (通常第 7 位)
        self.gripper_enabled = self.action_space.shape[0] > 6

        # 实例化手柄驱动
        self.expert = GamepadExpert()
        
        self.btn_a = False  # 闭合
        self.btn_b = False  # 开启
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        干预核心逻辑：
        1. 获取手柄动作向量 [6维] 和 按钮状态
        2. 只要摇杆/扳机/十字键有输入，或 A/B 按下，即判定为干预
        """
        expert_a, buttons = self.expert.get_action()
        
        # 映射：A键(buttons[0]) 闭合，B键(buttons[1]) 开启
        self.btn_a = bool(buttons[0])
        self.btn_b = bool(buttons[1])
        
        intervened = False
        
        # 1. 判定 6 自由度轴是否有实质性输入 (阈值略高于 Expert 内部死区以保持稳定)
        if np.linalg.norm(expert_a) > 0.01:
            intervened = True


        # 2. 判定夹爪干预逻辑
        if self.gripper_enabled:
            if self.btn_a:
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.btn_b:
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            
            # 拼接为 7 维动作 [x, y, z, roll, pitch, yaw, gripper]
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        # 3. 维度过滤 (如果只允许干预部分维度)
        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        # 4. 返回：如果发生干预则返回手柄动作，否则返回原始 action
        if intervened:
            return expert_a, True

        return action, False

    def step(self, action: np.ndarray):
        """执行 step 并记录干预状态"""
        print(f"original action = {action}")
        new_action, replaced = self.action(action)
        print(f"new_action = {new_action}, replaced = {replaced}")

        obs, rew, done, truncated, info = self.env.step(new_action)
        # 在 info 中存储干预详情，用于后续数据分析或模仿学习采样
        info["intervened"] = replaced
        if replaced:
            info["intervene_action"] = new_action
        
        info["btn_a"] = self.btn_a
        info["btn_b"] = self.btn_b
        
        return obs, rew, done, truncated, info

    def close(self):
        """释放手柄进程"""
        self.expert.close()
        return super().close()

class VisionProIntervention(gym.ActionWrapper):
    """
    VisionProIntervention 类:
    使用 Apple Vision Pro 实现对机器人策略动作的实时干预。
    
    干预逻辑：
    - 激活开关：左手捏合 (is_active 为 True) 时，判定为人工干预。
    - 动作覆盖：激活时，使用右手位姿和左手 Roll 计算的 6/7 维动作。
    """

    def __init__(self, env, avp_ip="192.168.1.223", action_indices=None):
        """
        :param env: 待包装的环境
        :param avp_ip: Vision Pro 的 IP 地址
        :param action_indices: 可选，指定干预只作用于哪些维度
        """
        super().__init__(env)

        # 1. 判定动作空间维度 (通常 7 维包含夹爪)
        self.action_dim = self.action_space.shape[0]
        self.gripper_enabled = self.action_dim > 6

        # 2. 实例化 Vision Pro 专家模块
        self.expert = VisionProExpert(avp_ip=avp_ip)
        
        self.is_active = False  # 左手捏合状态
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        干预核心逻辑：
        1. 获取 Vision Pro 动作向量 [6维]、按钮状态 [0位为夹爪] 以及 激活状态
        2. 如果 is_active 为 True，则判定为人工干预，覆盖模型动作
        """
        # 获取子进程数据
        expert_a, buttons, is_active = self.expert.get_action()
        self.is_active = is_active
        
        intervened = False

        # --- 判定干预状态 ---
        # 只有当左手捏合激活，且右手确实有移动输入时才触发覆盖（或者只要激活就覆盖）
        if self.is_active:
            intervened = True

        if intervened:
            # 构建最终返回的动作向量
            final_expert_a = expert_a # 初始为 [vx, vy, vz, wx, wy, wz]

            # --- 夹爪干预逻辑 ---
            if self.gripper_enabled:
                # 映射：buttons[0]=1 (右手捏合) -> 闭合 (-1.0), 否则开启 (1.0)
                # 注意：这里根据你环境的定义调整正负号
                gripper_action = -1.0 if buttons[0] == 1 else 1.0
                gripper_action = np.array([gripper_action])
                
                # 拼接为 7 维动作
                final_expert_a = np.concatenate((final_expert_a, gripper_action), axis=0)

            # --- 维度过滤 ---
            if self.action_indices is not None:
                # 如果只想让 VisionPro 干预特定维度（例如只干预位置，旋转仍由模型控制）
                filtered_action = action.copy()
                for idx in self.action_indices:
                    if idx < len(final_expert_a):
                        filtered_action[idx] = final_expert_a[idx]
                return filtered_action, True

            return final_expert_a, True

        # 如果没有激活干预，返回模型原始动作
        return action, False

    def step(self, action: np.ndarray):
        """执行环境步进并记录干预详情"""
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        
        # 存储干预信息，方便后续进行 DAgger 或模仿学习数据筛选
        info["intervened"] = replaced
        if replaced:
            info["intervene_action"] = new_action
        
        info["vp_active"] = self.is_active # 左手是否捏合
        
        return obs, rew, done, truncated, info

    def close(self):
        """释放 Vision Pro 连接进程"""
        self.expert.close()
        return super().close()

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info
    
# %%
import time
import numpy as np
from kinova_env.envs.kinova_env import TestConfig

def test_movement_with_keyboard():
    print("=== 测试: 键盘干预模式 ===")
    print("控制说明:")
    print("  移动: WASD (XY轴), QE (Z轴)")
    print("  旋转: IJKL (Roll/Pitch), UO (Yaw)")
    print("  夹爪: F1 (闭合), F2 (开启)")
    print("  提示: 没有任何按键按下时，将执行程序默认的‘Z轴往复’动作。")
    
    env = None
    try:
        # 1. 实例化原始环境
        base_env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        
        # 2. 使用键盘干预包装器
        # 如果你只想让键盘控制位置，可以设置 action_indices=[0,1,2,6]
        env = KeyboardIntervention(base_env)
        
        env.reset()
        
        gripper = 0 
        for i in range(100): # 增加循环次数方便测试键盘
            # 生成程序默认动作：每隔20步切换一次夹爪状态，并尝试向上微动
            if i % 20 == 0:
                gripper = -gripper
            
            # 基础自动动作 (默认动作)
            default_action = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, gripper]) 

            # 3. 执行 step
            # 如果此时你按下键盘，env.step 内部会自动把 default_action 替换为键盘动作
            obs, reward, done, truncated, info = env.step(default_action)
            
            # 4. 打印反馈
            current_z = obs['state']['tcp_pose'][2]
            if info.get("intervened", False):
                print(f"\r[人机干预中] 实际动作: {info['intervene_action'][:3].round(3)} | Z: {current_z:.4f}", end="")
            else:
                print(f"\r[自动模式] 目标夹爪: {gripper:>4} | Z: {current_z:.4f}", end="")
            
            # 维持环境频率
            time.sleep(0.1) 

    except Exception as e:
        import traceback
        print(f"\n❌ 失败: {e}")
        traceback.print_exc()
        
    finally:
        if env: 
            # 注意：包装器的 close 会处理 KeyboardExpert 进程的关闭
            env.close()
        print("\n=== 测试结束 ===")

# test_movement_with_keyboard()
# %%
import time
from kinova_env.envs.kinova_env import KinovaEnv, TestConfig

def test_movement_with_gamepad():
    print("=== 测试: Xbox 手柄实时干预模式 ===")
    print("控制提示:")
    print("  - 左摇杆推前/后/左/右")
    print("  - 右扳机(RT)上升 / 左扳机(LT)下降")
    print("  - A键(闭合) / B键(开启)")
    print("  - 没有任何输入时：机器人将执行默认动作 (每步 Z 轴 +0.1)")
    print("-" * 40)
    
    env = None
    try:
        # 1. 实例化 Kinova 基础环境 (假设 hz=10)
        test_config = TestConfig()
        test_config.ACTION_SCALE = np.array([10, 0.5, 1.0]) 
        base_env = KinovaEnv(hz=10, config=test_config, fake_env=False)

        
        # 2. 包装手柄干预层
        env = GamepadIntervention(base_env)
        
        obs, _ = env.reset()
        
        for i in range(2000): # 测试 200 步
            # 3. 定义“自动模式”动作：每一步自动向上移动 0.1
            # 对应动作向量 [x, y, z, r, p, y, gripper]
            default_action = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]) 

            # 4. 执行 step (包装器内部会自动判断是否被手柄覆盖)
            obs, reward, done, truncated, info = env.step(default_action)
            
            # 5. 打印反馈信息
            # 获取当前 TCP 的 Z 坐标 (假设 obs 结构如您之前 Keyboard 代码所示)
            current_z = obs['state']['tcp_pose'][2]
            
            if info.get("intervened", False):
                act = info['intervene_action']
                print(f"\r🎮 [人工干预] 指令 Z: {act[2]:>5.2f} | 实际 TCP_Z: {current_z:.4f}", end="")
            else:
                print(f"\r🤖 [自动模式] 指令 Z: {default_action[2]:>5.2f} | 实际 TCP_Z: {current_z:.4f}", end="")
            
            if done:
                print("\n任务完成，重置环境...")
                env.reset()

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
    finally:
        if env: 
            env.close()
        print("\n=== 测试结束 ===")

# test_movement_with_gamepad()

import time
from kinova_env.envs.kinova_env import KinovaEnv, TestConfig

def test_movement_with_visionpro():
    print("=== 测试: Vision Pro 实时全姿态干预模式 ===")
    print("控制提示:")
    print("  1. 激活干预: 左手捏合 (Left Pinch) - 机器人将完全跟随你的右手位姿")
    print("  2. 右手移动: 控制末端 XYZ 平移")
    print("  3. 右手转动: 控制末端 Pitch(俯仰) 和 Yaw(偏航)")
    print("  4. 左手翻转: 控制末端 Roll(顺逆时针)")
    print("  5. 夹爪控制: 右手捏合 (Close: -1, Open: 1)")
    print("-" * 60)
    
    env = None
    try:
        # 1. 实例化环境配置
        test_config = TestConfig()
        test_config.MAX_EPISODE_LENGTH = 300
        # 增加旋转动作的缩放系数，确保旋转在 UI 上可见
        test_config.ACTION_SCALE = np.array([0.4, 10, 1.0]) 
        base_env = KinovaEnv(hz=15, config=test_config, fake_env=False)
        
        # 2. 包装 Vision Pro 干预层
        env = VisionProIntervention(base_env, avp_ip="192.168.1.223")
        
        obs, _ = env.reset()
        print("🚀 环境已就绪，开始 6-DOF 数据监测...")

        for i in range(10000):
            # 3. 默认动作：维持当前姿态，仅在 Z 轴微升
            # [vx, vy, vz, wx, wy, wz, gripper]
            default_action = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]) 

            # 4. 执行 Step
            obs, reward, done, truncated, info = env.step(default_action)
            
            # 5. 状态反馈：补充完整的 6 自由度 + 夹爪信息
            if info.get("intervened", False):
                act = info['intervene_action']
                
                # 动作拆解
                v_xyz = act[0:3]     # 线速度
                w_py = act[3:5]      # 右手控制的 Pitch, Yaw
                w_roll = act[5]      # 左手控制的 Roll
                gripper = act[6]     # 夹爪
                
                # 状态获取 (假设 obs 包含当前的欧拉角)
                # tcp_pose: [x, y, z, r, p, y]
                current_pose = obs['state']['tcp_pose']
                
                grip_str = "CLOSED" if gripper < 0 else "OPEN"
                
                # 格式化输出：涵盖所有干预维度
                print(f"\r🟢 [人工干预] "
                      f"PosV:[{v_xyz[0]:.2f}, {v_xyz[1]:.2f}, {v_xyz[2]:.2f}] | "
                      f"OriV(P/Y/R):[{w_py[0]:.1f}, {w_py[1]:.1f}, {w_roll:.1f}] | "
                      f"Grip:{grip_str} | "
                      f"Z:{current_pose[2]:.3f}", end="")
            else:
                # 自动模式输出
                print(f"\r🤖 [自动模式] Vz=0.02 巡航中... | 按住左手进行干预 | Z:{obs['state']['tcp_pose'][2]:.3f}" + " " * 40, end="")
            
            if done or truncated:
                print("\n任务结束，重置环境...")
                obs, _ = env.reset()

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env: 
            env.close()
        print("\n=== 测试结束 ===")

# 执行测试
test_movement_with_visionpro()