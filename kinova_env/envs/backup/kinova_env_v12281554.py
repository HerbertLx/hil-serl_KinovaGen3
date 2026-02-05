# %%
import sys
import os

# 定义你的包路径
target_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl"

# 检查路径是否存在（可选，但建议）
if os.path.exists(target_path):
    # 如果路径不在 sys.path 中，则添加它
    if target_path not in sys.path:
        # 使用 insert(0, ...) 确保该路径被优先搜索，防止被同名包覆盖
        sys.path.insert(0, target_path)
        print(f"✅ 成功添加路径: {target_path}")
    else:
        print("ℹ️ 路径已在环境中")
else:
    print("❌ 错误：找不到该路径，请检查路径是否正确")

# %%
"""Gym Interface for Kinova"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from kinova_env.camera.video_capture import VideoCapture
from kinova_env.camera.rs_capture import RSCapture
from kinova_env.utils.rotations import euler_2_quat, quat_2_euler
# %%

# 图像显示器类：继承自 Thread，在后台窗口更新画面
class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue          # 存放待显示图像的队列
        self.daemon = True          # 设置为守护线程，主程序退出时自动关闭
        self.name = name            # 窗口名称

    def run(self):
        while True:
            # 从队列中获取图像字典（包含多个相机的画面）
            img_array = self.queue.get() 
            if img_array is None:   # 如果收到 None，表示停止信号
                break

            # 处理图像：将各个相机的画面缩放为 128x128 并水平拼接 (concatenate)
            # 排除带有 "full" 标识的高分辨率原图，只显示缩略图
            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            # 调用 OpenCV 窗口显示拼接后的图像
            cv2.imshow(self.name, frame)
            cv2.waitKey(1) # 刷新窗口


##############################################################################
# %% case test for ImageDisplayer with real cameras
def test_displayer_with_real_cameras():
    # 初始化队列
    img_queue = queue.Queue()
    
    # 启动显示线程
    displayer = ImageDisplayer(img_queue, "Real-time Camera Test")
    displayer.start()

    # 打开两个真实摄像头
    # 根据你的信息：Side 是 /dev/video4, Wrist 是 /dev/video6
    cap_side = cv2.VideoCapture(4)
    cap_wrist = cv2.VideoCapture(6)

    # 设置分辨率为 640x480 (减小带宽压力)
    for cap in [cap_side, cap_wrist]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ 测试启动！正在读取 /dev/video4 和 /dev/video6...")
    print("按下 Ctrl+C 或在弹出窗口按下任意键停止")

    try:
        while True:
            ret_s, frame_s = cap_side.read()
            ret_w, frame_w = cap_wrist.read()

            if not ret_s or not ret_w:
                print("⚠️ 警告：无法获取摄像头画面，请检查连接")
                break

            # --- 模拟 KinovaEnv.get_im() 的数据结构 ---
            # 1. 包含普通的缩略图项
            # 2. 包含 "_full" 项（验证 ImageDisplayer 是否会过滤它）
            img_payload = {
                "side_camera": frame_s,
                "wrist_camera": frame_w,
                "side_camera_full": frame_s.copy() # 这个不应该显示在拼接图中
            }

            # 将字典放入队列
            img_queue.put(img_payload)

            # 控制循环频率（如 10Hz，模拟强化学习环境的频率）
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n检测到用户中断，正在关闭...")

    finally:
        # 释放资源
        cap_side.release()
        cap_wrist.release()
        # 发送 None 信号让显示线程退出
        img_queue.put(None)
        displayer.join()
        cv2.destroyAllWindows()
        print("👋 测试结束。")

# test_displayer_with_real_cameras()
# %%
class DefaultEnvConfig:
    """KinovaEnv 的默认配置类。请根据实际硬件和任务需求填充以下值。"""

    # --- 网络与硬件连接 ---
    SERVER_URL: str = "http://127.0.0.1:5000/"  # 机器人控制服务器的地址（通常是运行底层驱动的电脑 IP）
    REALSENSE_CAMERAS: Dict = {                  # 摄像头配置：键为自定义名称，值为 RealSense 摄像头的硬件序列号
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    
    # --- 图像处理 ---
    IMAGE_CROP: dict[str, callable] = {}         # 图像裁剪函数字典。例如：{"wrist_1": crop_func}，用于截取 ROI 区域

    # --- 任务目标与奖励 ---
    TARGET_POSE: np.ndarray = np.zeros((6,))     # 任务的目标位姿 [x, y, z, roll, pitch, yaw]
    GRASP_POSE: np.ndarray = np.zeros((6,))      # 抓取动作发生的特定位姿（用于某些特定脚本任务）
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,)) # 奖励阈值。当当前位姿与目标的差距小于此值时，判定为成功
    
    # --- 动作空间缩放 ---
    # 通常 RL 模型输出的是 [-1, 1]，通过此参数映射到物理单位
    # 例如：[平移步长, 旋转步长, 夹爪步长]
    ACTION_SCALE = np.zeros((3,)) 

    # --- 重置（Reset）逻辑 ---
    RESET_POSE: np.ndarray = np.zeros((6,))      # 环境重置时，机械臂回到的初始位姿 [x, y, z, r, p, y]
    RANDOM_RESET = False                         # 是否启用随机重置（增加环境泛化能力）
    RANDOM_XY_RANGE = (0.0,)                     # 随机重置时，X 和 Y 坐标允许的偏移范围
    RANDOM_RZ_RANGE = (0.0,)                     # 随机重置时，绕 Z 轴旋转的角度允许偏移范围

    # --- 安全边界（Bounding Box） ---
    # 限制机械臂末端 TCP 只能在某个立方体空间内运动，防止撞击桌面或围栏
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))         # 笛卡尔坐标及姿态的上限 [x, y, z, r, p, y]
    ABS_POSE_LIMIT_LOW = np.zeros((6,))          # 笛卡尔坐标及姿态的下限

    # --- 控制模式参数 ---
    # 这些字典通常包含 Kp, Kd 增益或阻抗系数，发送给服务端执行
    COMPLIANCE_PARAM: Dict[str, float] = {}      # 顺应性（柔性）模式参数，用于 RL 交互
    RESET_PARAM: Dict[str, float] = {}           # 重置过程中使用的控制参数
    PRECISION_PARAM: Dict[str, float] = {}       # 高精度位置模式参数
    
    # --- 负载补偿参数 ---
    # 定义末端工具（如夹爪）的质量及物理特性，用于重力补偿
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,                             # 负载质量 (kg)
        "F_x_center_load": [0.0, 0.0, 0.0],      # 负载重心位置
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0] # 负载惯性矩阵
    }

    # --- 运行控制 ---
    DISPLAY_IMAGE: bool = True                   # 是否在本地弹出 OpenCV 窗口显示摄像头画面
    GRIPPER_SLEEP: float = 0.6                   # 夹爪动作后的等待时间（秒），确保动作完成
    MAX_EPISODE_LENGTH: int = 100                # 每个 Episode（回合）的最大步数
    JOINT_RESET_PERIOD: int = 0                  # 关节重置周期。每隔多少个 Cycle 进行一次完整的关节角度重置（防止奇异点）


##############################################################################

class KinovaEnv(gym.Env):
    def __init__(
            self,
            hz=10,                  # 控制频率，即每秒钟执行多少个动作步（Step）
            fake_env=False,         # 是否启用虚拟模式。如果为True，则不连接真实硬件
            save_video=False,       # 是否在运行过程中录制并保存视频
            config: DefaultEnvConfig = None, # 传入的配置对象，包含机器人的物理和网络参数
            set_load=False,         # 是否在初始化时设置机械臂的负载参数（Mass/Inertia）
        ):
            # --- 1. 参数同步：将配置类中的参数赋值给成员变量 ---
            self.action_scale = config.ACTION_SCALE        # 动作缩放比例（归一化动作转物理量）
            self._TARGET_POSE = config.TARGET_POSE        # 训练的目标位姿
            self._RESET_POSE = config.RESET_POSE          # 重置时的初始位姿
            self._REWARD_THRESHOLD = config.REWARD_THRESHOLD # 判定成功的奖励阈值
            self.url = config.SERVER_URL                  # 机器人控制端服务器地址
            self.config = config                          # 保留完整的配置引用
            self.max_episode_length = config.MAX_EPISODE_LENGTH # 每个回合的最大步数
            self.display_image = config.DISPLAY_IMAGE     # 是否显示预览窗口
            self.gripper_sleep = config.GRIPPER_SLEEP     # 夹爪动作后的缓冲时间

            # --- 2. 位姿处理：将欧拉角转换为四元数 ---
            # 机器人底层通常使用四元数 (xyzw) 表示旋转，大小从 (6,) 变为 (7,) [x,y,z, qx,qy,qz,qw]
            self.resetpos = np.concatenate(
                [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
            )
            
            # --- 3. 状态初始化 ---
            self._update_currpos()             # 立即向服务端请求一次，获取机器人当前的实时位置
            self.last_gripper_act = time.time() # 记录上次夹爪动作的时间，用于频率控制
            self.lastsent = time.time()         # 记录上次发送指令的时间
            self.randomreset = config.RANDOM_RESET
            self.random_xy_range = config.RANDOM_XY_RANGE
            self.random_rz_range = config.RANDOM_RZ_RANGE
            self.hz = hz
            self.joint_reset_cycle = config.JOINT_RESET_PERIOD # 每隔多少个回合执行一次关节重置

            # --- 4. 视频保存初始化 ---
            self.save_video = save_video
            if self.save_video:
                print("Saving videos!")
                self.recording_frames = []      # 用于存储视频帧的缓冲区

            # --- 5. 定义安全边界 (Boundary Box) ---
            # 使用 gymnasium 的 Box 空间定义 TCP 末端的移动范围
            self.xyz_bounding_box = gym.spaces.Box(
                config.ABS_POSE_LIMIT_LOW[:3],   # XYZ 最小值
                config.ABS_POSE_LIMIT_HIGH[:3],  # XYZ 最大值
                dtype=np.float64,
            )
            self.rpy_bounding_box = gym.spaces.Box(
                config.ABS_POSE_LIMIT_LOW[3:],   # 旋转 (Roll/Pitch/Yaw) 最小值
                config.ABS_POSE_LIMIT_HIGH[3:],  # 旋转最大值
                dtype=np.float64,
            )

            # --- 6. 定义强化学习的动作空间 (Action Space) ---
            # 标准化为 [-1, 1] 的 7 维向量：前 6 维是 TCP 位姿增量，第 7 维是夹爪开合
            self.action_space = gym.spaces.Box(
                np.ones((7,), dtype=np.float32) * -1,
                np.ones((7,), dtype=np.float32),
            )

            # --- 7. 定义强化学习的观测空间 (Observation Space) ---
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(    # 机械臂的状态信息
                        {
                            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)), # 当前位姿 (xyz+quat)
                            "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),  # 当前速度 (linear+angular)
                            "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),       # 夹爪当前位置
                            "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),# 末端传感器受力
                            "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),# 末端传感器扭矩
                        }
                    ),
                    "images": gym.spaces.Dict(   # 图像观测
                        {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS} # 遍历配置中的所有相机
                    ),
                }
            )
            self.cycle_count = 0  # 计数器，用于决定何时触发 joint_reset

            if fake_env:
                return  # 如果是虚拟环境，到此结束，不初始化硬件

            # --- 8. 硬件初始化：相机与显示器 ---
            self.cap = None
            self.init_cameras(config.REALSENSE_CAMERAS) # 初始化 RealSense 驱动
            if self.display_image:
                self.img_queue = queue.Queue()          # 图像线程队列
                self.displayer = ImageDisplayer(self.img_queue, self.url)
                self.displayer.start()                  # 启动独立的图像显示线程

            # --- 9. 硬件初始化：负载设置 (Set Load) ---
            if set_load:
                # 这是一个交互式过程，需要人工将机器人切换模式
                input("Put arm into programing mode and press enter.")
                requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
                input("Put arm into execution mode and press enter.")
                for _ in range(2):
                    self._recover()                     # 清除可能产生的错误
                    time.sleep(1)

            # --- 10. 安全机制：键盘紧急退出 ---
            if not fake_env:
                from pynput import keyboard
                self.terminate = False
                def on_press(key):
                    # 监听 ESC 键，按下时将终止标志设为 True
                    if key == keyboard.Key.esc:
                        self.terminate = True
                        print("Emergency Termination Triggered!")
                self.listener = keyboard.Listener(on_press=on_press)
                self.listener.start()                   # 后台启动键盘监听

            print("Initialized Kinova")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

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

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                requests.post(self.url + "close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                requests.post(self.url + "open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
