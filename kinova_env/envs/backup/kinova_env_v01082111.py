"""
Gym Interface for Kinova Gen3 (Using KinovaManager)
Refactored to use kinova_manage.py for simplified control logic.
"""

# %%
import sys
import os

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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# %% imports
import numpy as np
import gymnasium as gym
import cv2
import copy
import time
import threading
import queue
from datetime import datetime
from collections import OrderedDict
from typing import Dict
from scipy.spatial.transform import Rotation



# --- 导入辅助类 ---
from kinova_env.camera.video_capture import VideoCapture
from kinova_env.camera.rs_capture import RSCapture
from kinova_env.utils.rotations import euler_2_quat, quat_2_euler

# %% 
# 1. 图像显示线程 (保持不变)
# =========================================================
class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True 
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


# %%
# 2. 环境配置类
# =========================================================
class DefaultEnvConfig:
    # 修改为实际 IP
    SERVER_IP: str = "192.168.8.10"
        
    REALSENSE_CAMERAS = {
        "side_1": { "index": 4 },  # 对应 /dev/video4
        "wrist_1": { "index": 6 }  # 对应 /dev/video6
    }
    
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))

    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 2.0
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    
    # 各种参数
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)

class USBCaptureAdapter:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print(f"⚠️ 警告: 无法打开摄像头 {name} (index {index})")

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def close(self):
        if self.cap:
            self.cap.release()

# %%
# 3. KinovaEnv (基于 KinovaManager 重构)
# =========================================================
class KinovaEnv(gym.Env):
    def __init__(
        self,
        hz=25,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        
        self.config = config
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        self.last_gripper_time = time.time()

        # 转换重置位姿 (Euler -> Quat)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        
        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # ---------------------------------------------------
        # 初始化机器人 (使用 KinovaManager)
        # ---------------------------------------------------
        from kinova_manage import KinovaManager
        self.robot = KinovaManager(ip_address=config.SERVER_IP)

        if not fake_env:
            try:
                self.robot.connect()
            except Exception as e:
                print(f"❌ 机器人连接失败: {e}")
        # 初始化状态变量
        self.currpos = np.zeros(7) # XYZ + Quat
        self.currvel = np.zeros(6)
        self.currforce = np.zeros(3)
        self.currtorque = np.zeros(3)
        self.curr_gripper_pos = np.zeros(1)
        
        # 第一次更新状态
        self._update_currpos()
        self.cycle_count = 0
        self.curr_path_length = 0 
        self.terminate = False
        
        # Random Reset params
        self.randomreset = config.RANDOM_RESET

        # Boundary Box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_pose": gym.spaces.Box(0, 1, shape=(1,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            }),
            "images": gym.spaces.Dict({
                key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                for key in config.REALSENSE_CAMERAS
            }),
        })

        if fake_env: return

        # 相机初始化
        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "Kinova View")
            self.displayer.start()

        # 键盘监听
        if not fake_env:
            from pynput import keyboard
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
                    print("🛑 Emergency Stop Triggered!")
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("✅ Initialized KinovaEnv (Using KinovaManager)")

    # ---------------------------------------------------------
    # 状态更新 (调用 Manager)
    # ---------------------------------------------------------
    def _update_currpos(self):
        """
        从 KinovaManager 获取状态并转换为 Gym 格式
        """
        # 使用 Manager 的 get_status 接口
        status = self.robot.get_status()
        if status is None:
            return 
            
        base = status.base
        
        # 1. Pose: XYZ (m)
        xyz = np.array([base.tool_pose_x, base.tool_pose_y, base.tool_pose_z])
        # 2. Pose: Euler (Deg) -> Quaternion
        # 注意: KinovaManager 返回的是度数，Gym 需要四元数
        euler_deg = np.array([base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z])
        try:
            r = Rotation.from_euler('xyz', euler_deg, degrees=True)
            quat = r.as_quat() # [x, y, z, w]
            # print(f"\nxyz = {xyz}, quat = {quat}")
            # print(f"\nnp.concatenate([xyz, quat]) = {np.concatenate([xyz, quat])}")
            self.currpos = np.concatenate([xyz, quat])
        except Exception as e:
            print(f"State math error: {e}")
        # 3. Velocity
        self.currvel = np.array([
            base.tool_twist_linear_x, base.tool_twist_linear_y, base.tool_twist_linear_z,
            base.tool_twist_angular_x, base.tool_twist_angular_y, base.tool_twist_angular_z
        ])
        # 4. Force/Torque
        self.currforce = np.array([base.tool_external_wrench_force_x, base.tool_external_wrench_force_y, base.tool_external_wrench_force_z])
        self.currtorque = np.array([base.tool_external_wrench_torque_x, base.tool_external_wrench_torque_y, base.tool_external_wrench_torque_z])
        # 5. Gripper (0-100 -> 0-1)
        # 假设只有一个夹爪
        grip = status.interconnect.gripper_feedback.motor
        if len(grip) > 0:
            self.curr_gripper_pos = np.array([grip[0].position / 100.0])
        else:
            self.curr_gripper_pos = np.array([0.0])

        # print(f"\nbase.tool_pose_x: {base.tool_pose_x}, tool_pose_y: {base.tool_pose_y}, tool_pose_z: {base.tool_pose_z}")

    def update_currpos(self):
        """
        从 KinovaManager 获取状态并转换为 Gym 格式
        """
        # 使用 Manager 的 get_status 接口
        status = self.robot.get_status()
        if status is None:
            return 
            
        base = status.base
        # 1. Pose: XYZ (m)
        xyz = np.array([base.tool_pose_x, base.tool_pose_y, base.tool_pose_z])

        # 2. Pose: Euler (Deg) -> Quaternion
        # 注意: KinovaManager 返回的是度数，Gym 需要四元数
        euler_deg = np.array([base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z])
        try:
            r = Rotation.from_euler('xyz', euler_deg, degrees=True)
            quat = r.as_quat() # [x, y, z, w]
            # print(f"\nxyz = {xyz}, quat = {quat}")
            # print(f"\nnp.concatenate([xyz, quat]) = {np.concatenate([xyz, quat])}")
            self.currpos = np.concatenate([xyz, quat])
        except Exception as e:
            print(f"State math error: {e}")

        # 3. Velocity
        self.currvel = np.array([
            base.tool_twist_linear_x, base.tool_twist_linear_y, base.tool_twist_linear_z,
            base.tool_twist_angular_x, base.tool_twist_angular_y, base.tool_twist_angular_z
        ])

        # 4. Force/Torque
        self.currforce = np.array([base.tool_external_wrench_force_x, base.tool_external_wrench_force_y, base.tool_external_wrench_force_z])
        self.currtorque = np.array([base.tool_external_wrench_torque_x, base.tool_external_wrench_torque_y, base.tool_external_wrench_torque_z])
        
        # 5. Gripper (0-100 -> 0-1)
        # 假设只有一个夹爪
        grip = status.interconnect.gripper_feedback.motor
        if len(grip) > 0:
            self.curr_gripper_pos = np.array([grip[0].position / 100.0])
        else:
            self.curr_gripper_pos = np.array([0.0])
        
        # print(f"\nbase.tool_pose_x: {base.tool_pose_x}, tool_pose_y: {base.tool_pose_y}, tool_pose_z: {base.tool_pose_z}")


    def _get_obs(self) -> dict:
        images = self.get_im()
        # print(f"\nIn _get_obs, currpos = {self.currpos}")
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    # ---------------------------------------------------------
    # 核心 Step 函数
    # ---------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple:
        """
        执行环境的一个步进动作，控制机器人移动并更新夹爪状态。
        
        功能逻辑:
        1. 接收并限幅 Action，结合 action_scale 计算目标位姿增量（Delta Pose）。
        2. 位置更新：在当前 Cartesian 坐标基础上累加 XYZ 偏移。
        3. 姿态更新：利用四元数左乘旋转增量（Rotation Vector 转四元数），避免万向节死锁。
        4. 安全限幅：调用 clip_safety_box 确保目标位姿不超出定义的限制区域。
        5. 硬件指令下发：
           - 调用 _send_pos_command 发送 7 位目标位姿（自动处理欧拉角转换）。
           - 调用 _send_gripper_command 发送夹爪信号（支持二值化或连续控制）。
        6. 维持设定的环境控制频率（hz），并返回最新的状态观测。

        Args:
            action (np.ndarray): 形状为 (7,) 的动作向量。
                - action[0:3]: XYZ 轴位移增量 [-1, 1]。
                - action[3:6]: 旋转向量增量 [-1, 1]。
                - action[6]: 夹爪信号 [-1, 1]。

        Returns:
            tuple: (ob, reward, done, truncated, info)
                - ob (dict): 包含图像和机器人状态。
                - reward (int): 奖励值。
                - done (bool): 是否终止。
                - truncated (bool): 是否截断。
                - info (dict): 包含成功标志等辅助信息。
        """
        start_time = time.time()
        print(f"\nStart time: {start_time}")
        
        # 0. 动作限幅，确保输入在合理范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. 计算目标位姿 (Target Pose Calculation)
        target_pos = self.currpos.copy()
        
        # 1.1 位置增量更新 (X, Y, Z)
        xyz_delta = action[:3] * self.action_scale[0]
        target_pos[:3] += xyz_delta

        # 1.2 姿态增量更新 (Rotation Vector -> Quaternion)
        rot_delta_vec = action[3:6] * self.action_scale[1]
        if np.linalg.norm(rot_delta_vec) > 1e-6:
            r_delta = Rotation.from_rotvec(rot_delta_vec)
            r_curr = Rotation.from_quat(self.currpos[3:])
            # 采用左乘方式：r_next = r_delta * r_curr
            target_pos[3:] = (r_delta * r_curr).as_quat()

        # 2. 安全限幅 (Safety Box Constraint)
        # 确保 target_pos 不会超出机器人工作的物理安全边界
        target_pos = self.clip_safety_box(target_pos)

        # 3. 硬件指令下发 (Hardware Communication)
        
        # 3.1 下发机械臂位姿指令
        # 内部自动处理：四元数 -> 欧拉角转换
        print(f"Before _send_pos_command, time: {time.time()}")
        self._send_pos_command(target_pos)
        print(f"After _send_pos_command, time: {time.time()}")
        # 3.2 下发夹爪控制指令
        # 根据 action[6] 发送控制信号。此处 scale[2] 用于缩放信号强度
        # mode 可以根据需求设为 "binary" (带保护) 或 "continuous" (直接线性映射)
        gripper_sig = action[6] * self.action_scale[2]
        self._send_gripper_command(gripper_sig, mode="binary")

        self.curr_path_length += 1
        
        # 4. 频率控制 (Frequency Regulation)
        dt = time.time() - start_time
        target_period = 1.0 / self.hz
        if dt < target_period:
            time.sleep(target_period - dt)

        # 5. 更新状态并获取观测 (Observation & Rewards)
        self._update_currpos()  # 同步当前硬件实际位置
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        
        # 终止条件判定
        if self.curr_path_length >= self.max_episode_length:
            print(f"Max_episode_length {self.max_episode_length} reached!")
        done = (self.curr_path_length >= self.max_episode_length) or (reward > 0) or self.terminate
        
        if done:
            print(f"\nEpisode done at step {self.curr_path_length}, reward: {reward}")
        # print(f"\nIn step, currpos = {self.currpos}")

        return ob, int(reward), done, False, {"succeed": bool(reward)}

    # ---------------------------------------------------------
    # Reset 函数
    # ---------------------------------------------------------
    def reset(self, joint_reset=False, **kwargs):
        if self.save_video:
            self.save_video_recording()
        
        self.cycle_count += 1
        # 周期性强制关节重置
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        # 调用 Manager 获取当前状态 (仅用于确认连接)
        self._update_currpos()

        # 执行重置
        self.go_to_reset(joint_reset=joint_reset)
        
        # 初始化 Gym 变量
        self.curr_path_length = 0
        self._update_currpos()
        self.terminate = False

        return self._get_obs(), {"succeed": False}

    def go_to_reset(self, joint_reset=False):
        """
        重置机器人至初始状态。
        
        逻辑说明:
        1. 可选的关节空间重置：直接驱动 7 个电机旋转至 Home 位姿。
        2. 笛卡尔空间重置：利用 _send_pos_command 将末端移动至 resetpos。
        3. 夹爪重置：利用 _send_gripper_command 将夹爪完全打开。
        
        Args:
            joint_reset (bool): 是否执行关节空间的 Home 位重置。
        """
        print("🔄 Resetting Robot...")
        
        # 1. 执行关节重置 (Joint Space Reset)
        if joint_reset:
            print("   Executing Joint Reset...")
            # 目标关节角度 (度) + 夹爪位置 (0 为全开)
            home_joints = [355.22, 4.94, 190.63, 241.29, 181.31, 51.82, 277.83, 0.0]
            self.robot.move_angular(home_joints, dual_grip=False)
            # 给硬件预留反应时间
            time.sleep(1.0)

        # 2. 执行末端位姿重置 (Cartesian Space Reset)
        # self.resetpos 存储的是 7 位位姿 [x, y, z, qx, qy, qz, qw]
        print(f"   Moving to Start Cartesian: {self.resetpos[:3]}")
        
        # 调用封装好的位置发送函数 (内部会自动处理四元数到欧拉角的转换)
        self._send_pos_command(self.resetpos)
        
        # 3. 执行夹爪重置 (Gripper Reset)
        # 发送 1.0 (开启信号)，mode="binary" 会触发 _send_gripper_command 中的开启逻辑
        print("   Opening Gripper...")
        self._send_gripper_command(1.0, mode="binary")
        
        # 4. 最终同步与等待
        # 确保重置动作完成后再开始后续采集
        time.sleep(1.0)
        self._update_currpos()
        print("✅ Robot Reset Complete.")

    # ---------------------------------------------------------
    # 辅助函数
    # ---------------------------------------------------------
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """
        对目标位姿进行安全边界裁切，并在触发边界时输出提示。
        """
        # 备份裁切前的原始位置以便对比
        original_xyz = pose[:3].copy()
        
        # 执行裁切 (位置部分)
        pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        
        # 检查是否发生了裁切 (计算欧氏距离或直接对比)
        if not np.allclose(original_xyz, pose[:3], atol=1e-5):
            print(f"\n⚠️  [Safety Box] 触发位置截断!")
            print(f"   原始目标: X:{original_xyz[0]:.4f}, Y:{original_xyz[1]:.4f}, Z:{original_xyz[2]:.4f}")
            print(f"   裁切结果: X:{pose[0]:.4f}, Y:{pose[1]:.4f}, Z:{pose[2]:.4f}")
            print(f"   限位范围: X:[{self.xyz_bounding_box.low[0]:.2f}, {self.xyz_bounding_box.high[0]:.2f}], "
                  f"Y:[{self.xyz_bounding_box.low[1]:.2f}, {self.xyz_bounding_box.high[1]:.2f}], "
                  f"Z:[{self.xyz_bounding_box.low[2]:.2f}, {self.xyz_bounding_box.high[2]:.2f}]")
        
        # 简化处理：目前仅限制位置 (XYZ)，旋转由其他逻辑控制
        return pose

    def compute_reward(self, obs: dict) -> bool:
        """
        计算当前状态与目标状态之间的奖励值（基于欧氏距离和角度偏差）。
        
        功能逻辑:
        1. 提取当前末端执行器的位姿（TCP Pose），包含位置（XYZ）和姿态（四元数）。
        2. 位置偏差计算：直接计算当前 XYZ 与目标 XYZ 的差值。
        3. 姿态偏差计算：
           - 将当前的四元数和目标的欧拉角分别转换为旋转矩阵。
           - 计算两个旋转矩阵之间的相对旋转矩阵。
           - 将相对旋转矩阵转换回欧拉角，得到各个轴向（Roll, Pitch, Yaw）的角度偏差。
        4. 阈值判定：将位置偏差和角度偏差合并，判断所有维度的偏差是否都小于预设的阈值（_REWARD_THRESHOLD）。

        输入参数:
            :param obs: dict, 环境观测字典。必须包含 obs["state"]["tcp_pose"]，其形状为 (7,)，即 [x, y, z, qx, qy, qz, qw]。

        输出参数:
            :return: bool, 如果当前位姿与目标的 6 自由度偏差均在阈值内，返回 True（代表完成任务/获得奖励），否则返回 False。
        """
        # 1. 获取当前位姿并提取旋转矩阵
        current_pose = obs["state"]["tcp_pose"]
        # 将当前四元数 [qx, qy, qz, qw] 转换为 3x3 旋转矩阵
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        
        # 2. 获取目标姿态并提取旋转矩阵
        # self._TARGET_POSE[3:] 存储的是目标欧拉角 (单位通常为弧度)
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        
        # 3. 计算旋转偏差 (Relative Rotation)
        # diff_rot = R_current^T * R_target，代表从当前姿态转到目标姿态所需的旋转
        diff_rot = current_rot.T @ target_rot
        
        # 4. 将相对旋转转换为欧拉角偏差 (Euler Error)
        # 得到在 XYZ 三个轴上的角度差值
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        
        # 5. 合并位置偏差与角度偏差
        # np.hstack 将位置差 [Δx, Δy, Δz] 和 角度差 [Δroll, Δpitch, Δyaw] 拼接成 6 维向量
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        
        # 6. 判定是否满足奖励条件
        # 只有当 6 个维度的绝对误差全部小于对应的 self._REWARD_THRESHOLD 时，才判定为 True
        return np.all(delta < self._REWARD_THRESHOLD)

    def init_cameras(self, name_config_dict=None):
        if hasattr(self, 'cap') and self.cap is not None:
            self.close_cameras()
        self.cap = OrderedDict()
        for cam_name, cfg in name_config_dict.items():
            if "index" in cfg:
                self.cap[cam_name] = VideoCapture(USBCaptureAdapter(cam_name, cfg["index"]))

    def get_im(self) -> Dict[str, np.ndarray]:
        images, display_images = {}, {}
        for key, cap in self.cap.items():
            rgb = cap.read()
            if rgb is None: continue
            cropped = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
            target_size = self.observation_space["images"][key].shape[:2][::-1]
            resized = cv2.resize(cropped, target_size)
            images[key] = resized[..., ::-1].copy()
            display_images[key] = resized.copy()
            
            if self.save_video: 
                # (Video logic simplified for brevity)
                pass

        if self.display_image and display_images:
            try: self.img_queue.put_nowait(display_images)
            except queue.Full: pass
        return images
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """
        保持接口一致：将机器人移动到目标位姿。
        不再在 Python 层手动插值，直接调用底层控制器以获得更平滑的运动。
        """
        # 1. 统一输入格式：如果是 6 位 (Euler)，转换为 7 位 (Quat) 以便后续统一处理
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        
        # 2. 直接调用发送命令，由 Kinova 底层完成平滑轨迹规划
        self._send_pos_command(goal)
        
        # 3. 模拟必要的等待时间或状态更新
        # 如果你希望它像 Franka 一样阻塞直到 timeout 结束，可以保留 sleep
        # 但为了不卡顿，通常直接更新状态即可
        self._update_currpos()
        self.nextpos = goal.copy()

    def _send_pos_command(self, pos: np.ndarray):
        """
        内部逻辑：将 7 位位姿 (XYZ + Quat) 转换为 6 位 (XYZ + Euler Deg)
        然后通过 KinovaManager 发送。
        """
        # 1. 位置部分 (X, Y, Z)
        xyz = pos[:3]
        
        # 2. 姿态部分处理：如果是 7 位 (四元数)，转回 6 位 (欧拉角)
        # 利用你代码中的辅助类 quat_2_euler (假设返回的是弧度，需要转为度)
        if len(pos) == 7:
            # 使用你提供的 quat_2_euler 转换
            euler_rad = quat_2_euler(pos[3:])
            euler_deg = np.rad2deg(euler_rad)
        else:
            # 如果已经是 6 位，则假设后三位是欧拉角弧度
            euler_deg = np.rad2deg(pos[3:])

        # 3. 组装 KinovaManager 需要的列表 [X, Y, Z, TX, TY, TZ, Gripper]
        # 保持当前夹爪位置 (0-100)
        cmd_list = [
            xyz[0], xyz[1], xyz[2], 
            euler_deg[0], euler_deg[1], euler_deg[2], 
            self.curr_gripper_pos[0] * 100.0 
        ]
        
        print(f"Before sending command, time: {time.time()}")
        # 4. 直接调用原来的核心函数
        # move_cartesian 内部会自动进行平滑控制，不会像 Python 循环那样卡顿
        self.robot.move_cartesian(cmd_list, dual_grip=False, skip_gripper=True)
        print(f"After sending command, time: {time.time()}\n")
        
    def _send_gripper_command(self, pos: float, mode="binary"):
        """
        功能: 
            将高层动作空间（Action Space）的夹爪信号转换为底层 Kinova 硬件控制指令。
            支持“二值化开关”和“连续位置”两种控制模式。

        参数:
            :param pos: float, 控制信号。
                - 在 "binary" 模式下: 
                    * <= -0.5 判定为“闭合”意图；
                    * >= 0.5 判定为“开启”意图；
                    * (-0.5, 0.5) 之间为死区，不触发动作。
                - 在 "continuous" 模式下: 
                    * 取值范围 [-1, 1]，会被线性映射到 [0, 100] 的绝对位置百分比。
            :param mode: str, 控制模式。
                - "binary": (默认) 二值化控制，适合大多数离散抓取任务，带状态自锁和冷却保护。
                - "continuous": 连续位置控制，直接映射到 0.0-100.0。

        实现逻辑:
            1. 频率保护：检查当前时间与 `last_gripper_time` 的差值，若小于 `GRIPPER_SLEEP` 则跳过，防止电机过热。
            2. 状态检查：在 "binary" 模式下，结合 `curr_gripper_pos` 进行自锁判断。
               - 只有当夹爪处于“打开”状态且收到“闭合”指令时，才执行闭合操作。
               - 只有当夹爪处于“非全开”状态且收到“开启”指令时，才执行开启操作。
            3. 硬件映射：调用 `KinovaManager.control_gripper` 发送指令。
               - 0.0 代表完全张开，100.0 代表完全闭合。

        输出格式: 
            None (直接修改硬件状态并更新内部时间戳)
        """
        # 冷却时间与配置检查
        gripper_sleep = getattr(self.config, 'GRIPPER_SLEEP', 1.0)
        
        if mode == "binary":
            # 逻辑判定：
            # pos <= -0.5 视为“闭合”意图 (对应 Manager 的 value=100.0)
            # pos >= 0.5  视为“开启”意图 (对应 Manager 的 value=0.0)
            
            # 1. 闭合逻辑 (从打开到闭合)
            if (pos <= -0.5) and (self.curr_gripper_pos[0] < 0.85) and (time.time() - self.last_gripper_time > gripper_sleep):
                # 注意：KinovaManager 100.0 是全闭，0.0 是全开
                # 这里我们直接调用 control_gripper，使用 dual_grip=False 来传绝对值
                self.robot.control_gripper(100.0, dual_grip=False)
                self.last_gripper_time = time.time()
                print("🦾 Gripper: Closing...")
                time.sleep(gripper_sleep)
                
            # 2. 开启逻辑 (从闭合到开启)
            elif (pos >= 0.5) and (self.curr_gripper_pos[0] > 0.15) and (time.time() - self.last_gripper_time > gripper_sleep):
                self.robot.control_gripper(0.0, dual_grip=False)
                self.last_gripper_time = time.time()
                print("🦾 Gripper: Opening...")
                time.sleep(gripper_sleep)
                
        elif mode == "continuous":
            # 连续模式：将 [-1, 1] 映射到 [0, 100]
            val_0_100 = ((pos + 1.0) / 2.0) * 100.0
            val_0_100 = np.clip(val_0_100, 0, 100)
            self.robot.control_gripper(val_0_100, dual_grip=False)
    
    def close_cameras(self):
        if self.cap:
            for cap in self.cap.values(): cap.close()

    def _recover(self):
        '''
        这里应该是原来机械臂请错的函数，但是这里就不做了
        '''
        pass

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def close(self):
        if hasattr(self, 'listener'): self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            self.displayer.join()
        if hasattr(self, 'robot'):
            self.robot.disconnect()

# %% 测试配置
# =========================================================

# %% 更新后的测试配置 (基于物理安全范围)
# =========================================================

class TestConfig(DefaultEnvConfig):
    SERVER_IP = "192.168.8.10"
    REALSENSE_CAMERAS = {
        "side_1": {"index": 4},
        "wrist_1": {"index": 6}
    }
    
    # 初始位姿保持不变 [x, y, z, r, p, y] (单位: m, rad)
    RESET_POSE = np.array([
        0.400, -0.000, 0.250,           
        np.deg2rad(180.00), np.deg2rad(0.00), np.deg2rad(90.00)               
    ])
    
    TARGET_POSE = RESET_POSE.copy()

    # --- 根据你提供的安全范围进行修改 ---
    # 顺序为 [x, y, z, roll, pitch, yaw]
    # 姿态部分 (r,p,y) 暂时保留较大的允许范围 (约 +- 1弧度/57度)，防止移动时报错
    ABS_POSE_LIMIT_LOW = np.array([
        0.240,   # x min
        -0.270,  # y min
        0.040,   # z min
        RESET_POSE[3] - 10.0, 
        RESET_POSE[4] - 10.0, 
        RESET_POSE[5] - 10.0
    ])
    
    ABS_POSE_LIMIT_HIGH = np.array([
        0.500,   # x max
        0.270,   # y max
        0.500,   # z max
        RESET_POSE[3] + 10.0, 
        RESET_POSE[4] + 10.0, 
        RESET_POSE[5] + 10.0
    ])

    # 动作缩放: 每次 step 最大位移 2cm, 最大旋转约 3度, 夹爪全量程
    ACTION_SCALE = np.array([0.02, 0.05, 1.0]) 
    
# %% 测试 1: 重置与状态读取
# =========================================================

def test_reset_manager_integration():
    print("=== 测试: Manager 集成与 Reset ===")
    env = None
    try:
        env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        obs, _ = env.reset(joint_reset=False) # 设为 True 测试关节重置
        print("✅ Reset 完成")
        
        pose = obs['state']['tcp_pose']
        grip = obs['state']['gripper_pose']
        print(f"当前位姿 (XYZ+Q): {pose}")
        print(f"夹爪状态 (0-1): {grip}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        if env: env.close()

# test_reset_manager_integration()

# %% 测试 2: 笛卡尔移动
# =========================================================

def test_movement():
    print("=== 测试: Step 移动 ===")
    env = None
    try:
        env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        env.reset()
        
        # 向上移动 (Z+)
        print("执行向上移动...")
        gripper = -1.0 # 1 is open the gripper, while -1 is close the gripper
        for i in range(5):
            gripper = -gripper

            if i == 4:
                gripper = 0

            action = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 10.0, gripper]) # Gripper close

            obs, _, _, _, _ = env.step(action)
            print(f"Step {i}: Z = {obs['state']['tcp_pose'][2]:.4f}")
            time.sleep(3)

    except Exception as e:
        print(f"❌ 失败: {e}")
    finally:
        if env: env.close()

# test_movement()
# %% 测试 3: 摄像机功能测试
# =========================================================

# %% 修改后的摄像机测试函数 (避免死锁版)
# =========================================================

def test_camera_setup_safe():
    """
    安全测试摄像机。
    利用 KinovaEnv 内部自带的 ImageDisplayer 线程进行显示。
    """
    print("=== 开始安全测试摄像机 (Safe Camera Test) ===")
    
    config = TestConfig()
    
    # 注意：确保 config.DISPLAY_IMAGE = True，这样 Env 才会启动显示线程
    config.DISPLAY_IMAGE = True
    
    env = None
    try:
        # 初始化环境（内部会启动 ImageDisplayer 线程）
        # 如果只想测相机，fake_env 可以设为 True 避免连接机器人
        env = KinovaEnv(hz=10, config=config, fake_env=False)
        
        print("🎥 画面应该已经在 [Kinova View] 窗口中显示。")
        print("提示：请在【控制台/Jupyter】点击停止按钮(Interrupt) 或按 Ctrl+C 来结束测试。")
        print("注意：不要手动点击窗口的 X 按钮。")

        # 主线程只需要保持运行，并不断触发 get_im() 即可
        # 图像会自动通过队列传给 ImageDisplayer 线程显示
        start_time = time.time()
        while time.time() - start_time < 60:  # 测试运行 60 秒
            obs = env.get_im()  # get_im 内部会自动把图放进 img_queue
            
            if not obs:
                print("⚠️ 无法获取图像，请检查硬件。")
                break
                
            time.sleep(0.1) # 降低主线程 CPU 占用

    except KeyboardInterrupt:
        print("\n用户停止测试。")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        if env:
            # env.close() 会安全地通知 ImageDisplayer 退出并关闭窗口
            env.close()
        print("=== 摄像机测试结束，窗口已安全关闭 ===")

# 调用新函数
# test_camera_setup_safe()

# %% KinovaManager 基础功能与笛卡尔移动测试
# =========================================================

def test_kinova_basic_control():
    # 1. 初始化并连接
    # 使用你文档中的默认 IP
    manager = KinovaManager(ip_address="192.168.8.10")
    
    try:
        print("🔗 正在连接机器人...")
        manager.connect()
        
        # 2. 获取并打印当前状态 (使用文档提到的 print_status)
        print("\n📊 --- 当前机器人状态 ---")
        current_status = manager.get_status()
        manager.print_status(current_status)
        
        # # 提取当前坐标作为安全参考
        base = current_status.base
        curr_x = base.tool_pose_x
        curr_y = base.tool_pose_y
        curr_z = base.tool_pose_z
        curr_tx = base.tool_pose_theta_x
        curr_ty = base.tool_pose_theta_y
        curr_tz = base.tool_pose_theta_z

        # # 3. 准备移动指令 (仅向上移动 3 厘米，这是最安全的动作)
        # # 目标格式: [X, Y, Z, TX, TY, TZ, Gripper]
        # # 我们保持旋转不变，夹爪保持现状 (假设现状是 0.0 全开)
        safe_target = [curr_x, curr_y, curr_z + 0.03, curr_tx, curr_ty, curr_tz, 0.0]
        # safe_target = [0.25, 0.26, 0.13, curr_tx, curr_ty, curr_tz, 0.0]  # 直接设定一个安全位置测试
        # safe_target = [0.39, -0.12, 0.3, curr_tx, curr_ty, curr_tz, 0.0]  # 直接设定一个安全位置测试
        
        print(f"\n🚀 准备执行安全移动测试:")
        print(f"   从 Z: {curr_z:.4f} 移动到 Z: {safe_target[2]:.4f} ")
        
        input("⚠️ 请确保机械臂周围无障碍物，按回车键 [Enter] 开始移动...")
        
        # 4. 执行笛卡尔绝对位置移动
        # dual_grip=False 表示夹爪我们传的是 0-100 的绝对值
        manager.move_cartesian(safe_target, dual_grip=False)
        
        print("\n✅ 移动完成！再次检查位置:")
        new_status = manager.get_status()
        manager.print_status(new_status)

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        
    finally:
        # 5. 断开连接
        print("\n断开连接并清理环境...")
        manager.disconnect()

# test_kinova_basic_control()


''' safety box info
safety range
x min = 0.240
x max = 0.500
y min = -0.270
y max = 0.270
z min = 0.04
z max = 0.500
'''


# %% 相机测试

def test_displayer_with_cameras():
    """
    测试侧方相机 (index 4) 和 腕部相机 (index 6)
    """
    # 配置摄像机索引
    camera_configs = {
        "side_rgb": 4,  # Intel RealSense 彩色流
        "wrist_rgb": 6  # Integrated Webcam 彩色流
    }
    
    # 1. 初始化摄像头对象
    caps = {}
    for name, index in camera_configs.items():
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"❌ 错误: 无法打开摄像头 {name} (Index {index})")
            continue
        caps[name] = cap
        print(f"✅ 摄像头 {name} (Index {index}) 已就绪")

    if not caps:
        print("没有可用的摄像头，测试终止。")
        return

    # 2. 创建队列和显示线程
    img_queue = queue.Queue(maxsize=10)
    display_thread = ImageDisplayer(img_queue, "Kinova_Camera_Test")
    display_thread.start()

    print("\n[开始采集画面] 按 Ctrl+C 键停止测试...")
    
    try:
        while True:
            current_frames = {}
            for name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    # 将采集到的原始 BGR 图像存入字典
                    current_frames[name] = frame
                else:
                    print(f"⚠️ 警告: 无法从 {name} 获取帧")

            if current_frames:
                # 放入队列供显示线程处理
                try:
                    img_queue.put_nowait(current_frames)
                except queue.Full:
                    pass # 队列满时跳过，保证实时性

            time.sleep(0.03) # 模拟约 30FPS 的采集速度

    except KeyboardInterrupt:
        print("\n正在停止测试...")
    finally:
        # 3. 清理资源
        img_queue.put(None) # 通知线程退出
        display_thread.join(timeout=2)
        for cap in caps.values():
            cap.release()
        print("所有资源已释放。")

# test_displayer_with_cameras()

# %% 测试: Kinova 线性插值运动 (Interpolation Move)
# =========================================================

def test_interpolation_movement():
    print("=== 开始测试: 线性插值运动 ===")
    
    # 1. 实例化环境 (确保 TestConfig 已定义)
    env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
    
    try:
        # 获取当前旋转姿态，确保移动时末端方向不变
        env._update_currpos()
        curr_quat = env.currpos[3:] 
        
        # 2. 定义测试点 (XYZ + Quat)
        # 点 A: [0.25, 0.26, 0.13]
        point_a = np.concatenate([[0.25, 0.26, 0.13], curr_quat])
        # 点 B: [0.39, -0.12, 0.3]
        point_b = np.concatenate([[0.39, -0.12, 0.3], curr_quat])

        print("\n📍 步骤 1: 正在移动到起点 A...")
        # 第一次移动可以稍微慢一点，给 3 秒
        env.interpolate_move(point_a, timeout=3.0)
        time.sleep(1.0)

        print("\n📍 步骤 2: 线性插值移动 A -> B (2.5秒)...")
        start_t = time.time()
        env.interpolate_move(point_b, timeout=2.5)
        print(f"   实际耗时: {time.time() - start_t:.2f}s")
        
        time.sleep(1.0)

        print("\n📍 步骤 3: 线性插值移动 B -> A (快速返回 1.5秒)...")
        start_t = time.time()
        env.interpolate_move(point_a, timeout=1.5)
        print(f"   实际耗时: {time.time() - start_t:.2f}s")

        print("\n✅ 插值运动测试完成！")

    except Exception as e:
        print(f"❌ 测试出错: {e}")
    finally:
        if env:
            print("\n正在清理并断开连接...")
            env.close()

# 运行测试
# test_interpolation_movement()

# %% Kinova 夹爪控制功能测试

import numpy as np
import time

def test_kinova_gripper():
    # 1. 假设环境已经初始化 (env 是你的 APPLEEnv 或 KinovaEnv 实例)
    # 这里我们手动模拟一个简单的测试流程
    print("开始夹爪控制测试...")
    env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
    
    # 确保状态是最新的
    env._update_currpos()
    print(f"初始夹爪位置 (0-1): {env.curr_gripper_pos[0]:.2f}")

    # --- 测试 A: 开启夹爪 ---
    print("\n[测试 A] 发送open信号 (pos=1.0)")
    env._send_gripper_command(1.0, mode="binary")
    env._update_currpos()
    print(f"当前位置: {env.curr_gripper_pos[0]:.2f}")

    time.sleep(10)

    # --- 测试 B: 闭合夹爪 ---
    print("\n[测试 B] 发送close信号 (pos=-1.0)")
    env._send_gripper_command(-1.0, mode="binary")
    env._update_currpos()
    print(f"当前位置: {env.curr_gripper_pos[0]:.2f}")

    time.sleep(10)

    # --- 测试 C: 连续控制 (可选) ---
    print("\n[测试 C] 发送连续信号 (pos=0.0 -> 50% 位置)")
    # env._send_gripper_command(0.0, mode="continuous")
    time.sleep(10)
    env._update_currpos()
    print(f"半开状态位置: {env.curr_gripper_pos[0]:.2f}")

    print("\n✅ 夹爪测试完成")

# test_kinova_gripper()
# %% state observation 测试

import time
import numpy as np

def test_observation_accuracy():
    print("=== 测试: Kinova 状态转换精度与实时性 ===")
    env = None
    try:
        # 1. 初始化环境 (必须 fake_env=False 才能读取真实数据)
        env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        print("🤖 机械臂已连接，准备读取实时状态...")
        print("提示: 请尝试手动微调机械臂末端或操作夹爪，观察数值变化。")
        print("-" * 50)

        # 循环测试 20 次，方便观察数值跳动
        for i in range(1):
            # env.step 或 env.reset 内部都会调用 _update_currpos
            # 我们这里直接调用获取最新观测值
            obs, _ = env.reset(joint_reset=False) 

            action = np.zeros(env.action_space.shape)
            action[2] = 1
            env.step(action)

            # 提取转换后的数据
            pose = obs['state']['tcp_pose']      # [x, y, z, qx, qy, qz, qw]
            vel = obs['state']['tcp_vel']        # [vx, vy, vz, wx, wy, wz]
            force = obs['state']['tcp_force']    # [fx, fy, fz]
            torque = obs['state']['tcp_torque']  # [tx, ty, tz]
            gripper = obs['state']['gripper_pose'] # [0-1]

            # 格式化输出
            print(f"\n[样本 {i+1}]")
            print(f"📍 位置 (XYZ m): {pose[:3].round(4)}")
            print(f"🔄 姿态 (四元数): {pose[3:].round(4)}")
            print(f"🤏 夹爪位置 (0-1): {gripper[0]:.4f} (对应原值: {gripper[0]*100:.1f}%)")
            print(f"💨 末端速度 (Twist m/s & rad/s): {vel.round(4)}")
            print(f"🦾 末端外力 (N): {force.round(4)}")
            print(f"🔩 末端力矩 (Nm): {torque.round(4)}")

            time.sleep(0.5) # 停顿一下方便肉眼观察

        print("-" * 50)
        print("✅ 状态读取测试完成")

        print(f"\n用传统方法读取机械臂状态")
        status = env.robot.get_status()
        # print(status)
        # exit()
        env.robot.print_status(status)
        print()

    except Exception as e:
        print(f"❌ 测试过程中发生崩溃: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            env.close()
            print("👋 环境已关闭")

# 运行测试
# test_observation_accuracy()