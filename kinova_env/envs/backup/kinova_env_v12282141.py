"""
Gym Interface for Kinova Gen3 (Direct Kortex API Version)
Adapted from Franka HTTP interface.
"""

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
# %% imports
import os
import sys
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import threading
import queue
from datetime import datetime
from collections import OrderedDict
from typing import Dict

# --- 导入 Kortex API 相关库 ---
# 请确保你的环境变量路径是正确的
sys.path.insert(0, "/home/cuhk/Documents/visionpro-kinova-rl/Kinova-kortex2_Gen3_G3L/api_python/examples")
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2

# --- 导入原本的辅助类 (假设路径不变) ---
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
# 2. Kinova 底层控制器封装
# 用于替代原本代码中的 requests.post
# =========================================================
class KinovaController:
    def __init__(self, ip="192.168.8.10", port=10000):
        self.ip = ip
        self.port = port
        self.username = "admin" # 默认用户名
        self.password = "admin" # 默认密码
        
        self.connection = None
        self.base = None
        self.base_cyclic = None
        self.connected = False

        # 模拟 args 对象用于 utilities
        class Args:
            def __init__(self, ip, port, user, pwd):
                self.ip = ip
                self.port = port
                self.username = user
                self.password = pwd
        self.args = Args(ip, port, self.username, self.password)

    def connect(self):
        """建立连接"""
        try:
            self.connection = utilities.DeviceConnection.createTcpConnection(self.args)
            self.router = self.connection.__enter__()
            self.base = BaseClient(self.router)
            self.base_cyclic = BaseCyclicClient(self.router)
            self.connected = True
            print(f"✅ Kinova Connected at {self.ip}")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            self.connected = False

    def disconnect(self):
        if self.connection:
            self.connection.__exit__(None, None, None)
            self.connected = False
            print("❌ Kinova Disconnected")

    def get_feedback(self):
        """获取低延迟反馈 (BaseCyclic)"""
        if not self.connected: return None
        return self.base_cyclic.RefreshFeedback()

    def clear_faults(self):
        """清除错误 (Recover)"""
        if not self.connected: return
        try:
            self.base.ClearFaults()
        except Exception as e:
            print(f"Clear faults failed: {e}")

    def send_pose(self, pos_concat, duration=0.1):
        """
        发送笛卡尔位姿控制指令
        pos_concat: [x, y, z, qx, qy, qz, qw] (m, quaternion)
        注意：Kinova API 需要 Euler Angles (Degrees)
        """
        if not self.connected: return

        # 1. 坐标转换: Quaternion (scipy) -> Euler Degrees (Kinova)
        xyz = pos_concat[:3]
        quat = pos_concat[3:] # qx, qy, qz, qw
        
        try:
            # Scipy 默认 quat 是 [x, y, z, w]
            r = Rotation.from_quat(quat)
            # Kinova 使用 Intrinsic XYZ 顺序 (通常)
            euler_deg = r.as_euler('xyz', degrees=True)
        except Exception as e:
            print(f"Rotation conversion error: {e}")
            return

        # 2. 构建 Action
        action = Base_pb2.Action()
        action.name = "Gym_Step_Action"
        action.application_data = ""

        # 笛卡尔姿态
        pose = action.reach_pose.target_pose
        pose.x = xyz[0]
        pose.y = xyz[1]
        pose.z = xyz[2]
        pose.theta_x = euler_deg[0]
        pose.theta_y = euler_deg[1]
        pose.theta_z = euler_deg[2]

        # TODO: ⚠️ 这里的核心问题：
        # BaseClient.ExecuteAction 是阻塞的，且耗时较长，不适合 10Hz 的高频伺服。
        # 完美的 10Hz 实现需要使用 BaseCyclic 发送 RealTime Frame。
        # 为了兼容你现在的代码结构，我暂时使用 ExecuteAction，
        # 但在真实训练中，这可能会导致动作卡顿。
        
        # 尝试非阻塞发送（如果需要等待完成，逻辑会变得很慢）
        # 这里我们不等待 e.wait()，因为 Gym 需要按频率 loop
        try:
            self.base.ExecuteAction(action)
        except Exception as e:
            pass # 忽略高频发送时的冲突报错

    def send_gripper(self, value_0_to_1):
        """发送夹爪指令 (0.0 打开 - 1.0 关闭) -> Kinova (0-100)"""
        if not self.connected: return
        
        # 限制范围
        val = max(0.0, min(1.0, value_0_to_1))
        
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = val
        print(f"finger.value sent to Kinova API: {finger.value}") # Debug 输出
        print(f"cmd.mode: {cmd.mode}")
        print(f"cmd = {cmd}")
        print()
        
        try:
            self.base.SendGripperCommand(cmd)
        except Exception as e:
            print(f"Gripper error: {e}")

    # TODO: 负载设置功能 (Set Load)
    def set_load(self, mass, center_of_mass, inertia):
        """
        TODO: 使用 self.base.SetPayloadInformation(payload) 实现
        目前留空，待查阅具体 Protobuf 定义后填充
        """
        # payload = Base_pb2.PayloadInformation()
        # payload.mass = mass
        # ...
        # self.base.SetPayloadInformation(payload)
        pass

    def joint_reset(self):
        """
        根据指定的关节角度执行机械臂重置
        目标角度: [355.22, 4.94, 190.63, 241.29, 181.31, 51.82, 277.83]
        """
        if not self.connected:
            print("❌ 未连接机械臂，无法执行重置")
            return

        print("🔄 正在启动关节重置 (Joint Reset)...")

        # 1. 定义目标关节角度（根据你提供的数据）
        target_angles = [355.22, 4.94, 190.63, 241.29, 181.31, 51.82, 277.83]

        # 2. 创建 Action 对象
        action = Base_pb2.Action()
        action.name = "Gym_Joint_Reset_Action"
        action.application_data = ""

        # 3. 填充关节角度
        # ReachJointAngles 包含 JointAngles 列表
        joint_angles = action.reach_joint_angles.joint_angles
        for i, angle in enumerate(target_angles):
            temp_angle = joint_angles.joint_angles.add()
            temp_angle.joint_identifier = i  # 关节 ID (0-6)
            temp_angle.value = angle         # 角度值 (度)

        # 4. 监听动作完成状态 (可选但建议)
        # 这里使用简单的同步等待逻辑，或者直接发送。
        # 由于是重置动作，建议等待其完成再进行后续 RL 训练。
        try:
            # 订阅动作通知以检查是否结束
            e = threading.Event()
            def check_for_end_or_abort(notification):
                if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                    e.set()
            
            # 暂时订阅通知
            handle = self.base.OnNotificationActionTopic(check_for_end_or_abort, Base_pb2.NotificationOptions())
            
            # 执行动作
            self.base.ExecuteAction(action)
            
            # 等待动作完成（设置 30 秒超时防止卡死）
            finished = e.wait(timeout=30)
            self.base.Unsubscribe(handle)
            
            if finished:
                print("✅ 关节重置成功到位")
            else:
                print("⚠️ 重置动作超时，请检查机械臂状态")

        except Exception as e:
            print(f"❌ 执行关节重置时发生错误: {e}")

# %% 测试夹爪控制 (Gripper Test)
# 测试夹爪控制 (Gripper Test)
import time

def test_gripper_function():
    print("=== 开始测试: 夹爪控制 ===")
    
    # 1. 初始化控制器 (假设您已经在之前的单元格定义了 KinovaController)
    # 如果是在独立脚本运行，请确保导入了必要的 Base_pb2 等库
    arm = KinovaController(ip="192.168.8.10")
    
    try:
        # 2. 建立连接
        arm.connect()
        if not arm.connected:
            return

        # 3. 清除可能存在的错误
        arm.clear_faults()
        time.sleep(1)

        # 4. 测试全开 (0.0)
        print("动作: 正在全开夹爪 (0.0)...")
        arm.send_gripper(0.0)
        time.sleep(2) # 给夹爪物理移动的时间

        # 5. 测试半开 (0.5)
        # print("动作: 正在移动到半开位置 (0.5)...")
        # arm.send_gripper(0.5)
        # time.sleep(2)

        # 6. 测试全关 (1.0)
        print("动作: 正在全关夹爪 (1.0)...")
        arm.send_gripper(1.0)
        time.sleep(2)

        # 7. 最后回到全开状态，方便后续实验
        print("动作: 测试完毕，恢复全开状态...")
        arm.send_gripper(0.0)
        time.sleep(1)

    except Exception as e:
        print(f"❌ 测试过程中出现异常: {e}")
    finally:
        # 8. 断开连接
        arm.disconnect()
        print("=== 夹爪测试结束 ===")

# 执行测试
# test_gripper_function()

# %%=========================================================
# 3. 环境配置类
# =========================================================
class DefaultEnvConfig:
    # 修改为实际 IP
    SERVER_IP: str = "192.168.8.10"
        
    REALSENSE_CAMERAS = {
        "side_1": {
            "index": 4  # 对应 /dev/video4 (RealSense 的 RGB 流)
        },
        "wrist_1": {
            "index": 6  # 对应 /dev/video6 (手腕普通摄像头)
        }
    }
    
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    
    # 各种参数
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    
    # TODO: 这里的参数结构可能需要根据 Kinova API 调整
    COMPLIANCE_PARAM: Dict[str, float] = {} 
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0]*9
    }
    
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))

class USBCaptureAdapter:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.cap = cv2.VideoCapture(index)
        # 设置缓冲区大小为 1，减少延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print(f"⚠️ 警告: 无法打开摄像头 {name} (index {index})")

    def read(self):
        # 显式确保只返回两个值：ret 和 frame
        ret, frame = self.cap.read()
        return ret, frame

    def close(self):
        if self.cap:
            self.cap.release()
# %%=========================================================
# 4. 重构后的 KinovaEnv
# =========================================================
class KinovaEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.config = config
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        # 转换重置位姿 (Euler -> Quat)
        # 假设 config.RESET_POSE 后三位是 Euler Angles (Radians or Degrees? usually Rad in Config)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        
        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # ---------------------------------------------------
        # 初始化机器人连接 (替换 HTTP)
        # ---------------------------------------------------
        self.robot = KinovaController(ip=config.SERVER_IP)
        if not fake_env:
            self.robot.connect()
        
        # 初始化状态变量
        self.currpos = np.zeros(7)
        self.currvel = np.zeros(6)
        self.currforce = np.zeros(3)
        self.currtorque = np.zeros(3)
        self.curr_gripper_pos = np.zeros(1)
        
        # 第一次更新状态
        self._update_currpos()
        
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        
        # Random Reset params
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE

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
                "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            }),
            "images": gym.spaces.Dict({
                key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                for key in config.REALSENSE_CAMERAS
            }),
        })
        self.cycle_count = 0
        self.curr_path_length = 0 


        if fake_env: return

        # 相机初始化
        self.cap = None
        print( "Initializing cameras..." )
        print(f"Camera config: {config.REALSENSE_CAMERAS}" )
        print()
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "Kinova View")
            self.displayer.start()

        # ---------------------------------------------------
        # 负载设置 (Set Load) - 待实现区域
        # ---------------------------------------------------
        if set_load:
            print("⚠️ Load setting logic is currently a placeholder.")
            # input("Put arm into programing mode...")
            # self.robot.set_load(...) 
            # input("Put arm into execution mode...")
            for _ in range(2):
                self._recover()
                time.sleep(1)

        # 键盘监听
        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
                    print("🛑 Emergency Stop Triggered!")
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("✅ Initialized KinovaEnv (Direct API Mode)")

    # (保留原有的 clip_safety_box, compute_reward, get_im, save_video_recording, init/close cameras)
    # ... 此处省略未修改的辅助函数以节省篇幅，实际使用时请将原代码的这些函数复制过来 ...
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

    def init_cameras(self, name_config_dict=None):
        """
        针对 index 格式优化的初始化函数
        参数格式: {"side_1": {"index": 4}, "wrist_1": {"index": 6}}
        """
        print("正在初始化摄像头...")
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        
        for cam_name, cfg in name_config_dict.items():
            if "index" in cfg:
                print(f"正在初始化摄像头: {cam_name} (Index: {cfg['index']})...")
                
                # 使用适配器确保 read() 返回值严格为 2 个
                capture_handle = USBCaptureAdapter(cam_name, cfg["index"])
                
                # 包装进你的多线程 VideoCapture 类
                # 注意：请确保你的 VideoCapture 类内部调用的是 self.cap.read()
                self.cap[cam_name] = VideoCapture(capture_handle)
                print(f"✅ {cam_name} 初始化完成")
            else:
                print(f"❌ 错误: 摄像头 {cam_name} 配置中缺少 'index'")

        # 等待摄像头启动稳定
        time.sleep(1.0)

    def get_im(self) -> Dict[str, np.ndarray]:
        """从各个摄像头获取最新帧并进行预处理。"""
        images = {}          
        display_images = {}  
        full_res_images = {} 

        for key, cap in self.cap.items():
            try:
                # 从 VideoCapture 线程队列中取出最新帧
                rgb = cap.read() 
                if rgb is None:
                    continue

                # 1. 裁剪处理 (ROI)
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb

                # 2. 缩放至强化学习模型输入尺寸 (128x128)
                target_size = self.observation_space["images"][key].shape[:2][::-1] # (128, 128)
                resized = cv2.resize(cropped_rgb, target_size)

                # 3. 颜色空间转换 (BGR -> RGB) 用于模型输入
                images[key] = resized[..., ::-1].copy()

                # 4. 准备显示数据
                display_images[key] = resized.copy()
                display_images[key + "_full"] = cropped_rgb.copy()

                # 5. 准备视频保存数据
                if self.save_video:
                    full_res_images[key] = cropped_rgb.copy()

            except Exception as e:
                print(f"⚠️ {key} 摄像头读取错误: {e}")
                # 如果卡死，尝试重启
                # self.init_cameras(self.config.REALSENSE_CAMERAS)
                # return self.get_im()

        # 视频录制逻辑
        if self.save_video and full_res_images:
            self.recording_frames.append(full_res_images)

        # 推送到多线程显示窗口
        if self.display_image and display_images:
            try:
                self.img_queue.put_nowait(display_images)
            except queue.Full:
                pass # 如果显示线程太慢，跳过这一帧，防止阻塞主循环

        return images

    def close_cameras(self):
        """释放所有摄像头资源。"""
        print("正在关闭所有摄像头...")
        if self.cap:
            for name, cap in self.cap.items():
                try:
                    cap.close()
                    print(f"已释放: {name}")
                except Exception as e:
                    print(f"释放 {name} 失败: {e}")
            self.cap = None

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

    # =========================================================
    # 核心 Step 函数
    # =========================================================
    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. 计算下一时刻的 TCP 位姿 (Next Pose)
        xyz_delta = action[:3]
        self.nextpos = self.currpos.copy()
        
        # 位置更新
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # 姿态更新 (Rotation Vector -> Quaternion multiplication)
        # 注意：这里假设 action[3:6] 是旋转向量增量
        self.nextpos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        # 2. 安全限幅
        # self.nextpos = self.clip_safety_box(self.nextpos) # 请取消注释使用

        print(f"Step Action: Delta Pos {xyz_delta}, Delta Rot {action[3:6]}")
        print(f"Action[6] (Gripper Command): {action[6]}")
        print(f"self.action_scale: {self.action_scale}")
        # 3. 发送指令 (Send Command)
        gripper_action = action[6] * self.action_scale[2] # 假设映射到 [0, 1] 或 [-1, 1]
        print(f"Gripper {gripper_action}")
        print()
        # 调用 KinovaController 发送指令
        self._send_gripper_command(gripper_action) # 内部逻辑需适配范围
        self._send_pos_command(self.nextpos)       # 发送笛卡尔位姿

        self.curr_path_length += 1
        
        # 4. 频率控制
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        # 5. 获取观测
        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        
        return ob, int(reward), done, False, {"succeed": reward}

    # =========================================================
    # 辅助与底层交互函数 (核心修改区域)
    # =========================================================
    def _recover(self):
        """调用 controller 清除错误"""
        self.robot.clear_faults()

    def _send_pos_command(self, pos: np.ndarray):
        """发送笛卡尔位姿 (XYZ + Quat)"""
        self._recover()
        # pos 是 [x, y, z, qx, qy, qz, qw]
        self.robot.send_pose(pos)

    def _send_gripper_command(self, raw_val: float):
        """
        将强化学习模型输出的动作值转换为物理夹爪指令并发送。
        
        作用：
        1. 动作空间映射：将 RL 算法常用的 [-1, 1] 连续空间映射到硬件要求的 [0, 1] 空间。
        2. 频率控制（Throttling）：防止由于控制频率过高导致夹爪电机控制板过载或通信阻塞。
        
        输入参数 (raw_val): 
            - 类型: float
            - 含义: 原始动作值，通常由算法输出。
            - 范围: 假设为 [-1, 1]。其中 -1 通常代表“期望完全打开”，1 代表“期望完全关闭”。
            
        输出:
            - 无直接返回值，但会通过 self.robot 向实体机械臂发送控制包。
        """
        
        # --- 1. 动作空间线性映射 ---
        # 公式解析：
        # 当 raw_val = -1.0 -> normalized_val = (-1 + 1) / 2 = 0.0 (全开)
        # 当 raw_val = 1.0  -> normalized_val = (1 + 1) / 2 = 1.0 (全关)
        # 这种处理确保了神经网络的输出对称性，有利于模型收敛。
        normalized_val = (raw_val + 1) / 2.0 
        
        # --- 2. 状态机逻辑与频率限制 ---
        # 作用：由于底层机械臂夹爪（如 Robotiq 或 Kinova 原装夹爪）通常有内部闭环控制，
        # 不需要像机械臂关节那样以 100Hz 甚至更高频率刷新指令。
        # 这里通过对比当前时间与上次发送时间，确保两次指令之间的间隔不小于 self.gripper_sleep（如 0.5s）。
        
        # 检查当前时间与“上一次夹爪动作时间”的差值是否超过了预设的冷却时间（gripper_sleep）
        # print(f"Time since last gripper action: {time.time() - self.last_gripper_act:.2f}s")
        # if (time.time() - self.last_gripper_act > self.gripper_sleep):
        #     # 调用底层控制器发送物理指令（范围已修正为 0.0 - 1.0）
        #     print(f"Sending gripper command: {normalized_val} (raw: {raw_val})")
        #     self.robot.send_gripper(normalized_val)
            
        #     # 发送成功后，更新“最后一次动作时间”，进入下一轮冷却周期
        #     self.last_gripper_act = time.time()
        self.robot.send_gripper(normalized_val)


    def _update_currpos(self):
        """
        核心：从 Kortex Feedback 解析数据到 Gym Numpy 格式
        """
        feedback = self.robot.get_feedback()
        if feedback is None:
            return # 连接失败时保持旧值
            
        base = feedback.base
        
        # 1. 解析 TCP Pose (XYZ + EulerDeg -> XYZ + Quat)
        xyz = np.array([base.tool_pose_x, base.tool_pose_y, base.tool_pose_z])
        euler_deg = np.array([base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z])
        
        try:
            # Kinova Euler -> Scipy Quaternion
            r = Rotation.from_euler('xyz', euler_deg, degrees=True)
            quat = r.as_quat() # [x, y, z, w]
            self.currpos = np.concatenate([xyz, quat])
        except Exception as e:
            print(f"State update math error: {e}")

        # 2. 解析 Velocity (Twist)
        self.currvel = np.array([
            base.tool_twist_linear_x, base.tool_twist_linear_y, base.tool_twist_linear_z,
            base.tool_twist_angular_x, base.tool_twist_angular_y, base.tool_twist_angular_z
        ])

        # 3. 解析 Force/Torque
        self.currforce = np.array([
            base.tool_external_wrench_force_x,
            base.tool_external_wrench_force_y,
            base.tool_external_wrench_force_z
        ])
        self.currtorque = np.array([
            base.tool_external_wrench_torque_x,
            base.tool_external_wrench_torque_y,
            base.tool_external_wrench_torque_z
        ])
        
        # 4. 解析 Gripper
        # Kinova feedback 是 0-100，这里归一化到 0-1 (或按需 -1 到 1)
        grip_motor = feedback.interconnect.gripper_feedback.motor
        if len(grip_motor) > 0:
            self.curr_gripper_pos = np.array([grip_motor[0].position / 100.0])
        else:
            self.curr_gripper_pos = np.array([0.0])

    def _get_obs(self) -> dict:
        # 保持不变
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def reset(self, joint_reset=False, **kwargs):
        """
        环境重置函数。用于在一个训练回合结束或失败后，将机械臂归位并初始化状态。
        
        输入参数 (Inputs):
            - joint_reset (bool): 是否执行“关节回零”。
                                 False: 仅进行笛卡尔空间移动到初始点（快）。
                                 True: 强制所有关节旋转到预设角度，消除奇异点或电缆缠绕（慢）。
            - **kwargs: 兼容 Gym/Gymnasium 接口的其他参数。

        输出参数 (Outputs):
            - obs (dict): 重置后的初始观测值（包含图像和机器人状态）。
            - info (dict): 包含元数据的字典，例如 {"succeed": False}。
        """

        # ---------------------------------------------------
        # 1. 参数更新 (Update Param) - 待实现
        # ---------------------------------------------------
        # TODO: 原 Franka 代码中此处通过 HTTP 请求更新了阻抗控制参数（如 Kp, Kd）。
        # 在 Kinova 中，如果需要从“顺应模式（Compliance）”切换回“精确模式（Precision）”，需在此处通过 API 调用。
        pass 

        # 2. 视频处理：如果开启了录制，在重置时将上一个回合的帧保存为视频文件
        if self.save_video:
            self.save_video_recording()
        

        # 3. 关节重置计数逻辑
        self.cycle_count += 1
        # 如果设置了 joint_reset_cycle（如每 200 次 reset 强制进行一次关节重置）
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True  # 触发耗时较长但更彻底的关节归位

        # 4. 故障恢复：调用 _recover (Base.ClearFaults)，确保机械臂没有处于急停或报错锁定状态
        self._recover()

        # 5. 执行物理归位动作
        # 该函数会驱动机械臂移动到 self.resetpos 定义的位姿
        self.go_to_reset(joint_reset=joint_reset)

        # 6. 二次故障检查：确保归位运动过程中没有发生碰撞或报错
        self._recover()
        
        # 7. 逻辑状态初始化
        self.curr_path_length = 0  # 当前回合步数归零
        self._update_currpos()     # 从 API 获取机器人当前的最新位姿数据
        self.terminate = False     # 清除由键盘 ESC 触发的终止标志

        # 8. 获取重置后的第一帧观测并返回
        obs = self._get_obs()
        return obs, {"succeed": False}

    def go_to_reset(self, joint_reset=False):
        """
        功能：将机械臂重置到预设的初始状态。
        
        输入参数 (Inputs):
            joint_reset (bool): 是否执行关节空间重置。
                - 如果为 True：机械臂会执行一次完整的关节角度回正（通常是回到 Home 位姿），
                  用于消除笛卡尔控制累积的旋转奇异点或线缆缠绕。
                - 如果为 False：仅进行笛卡尔空间（直线）移动到初始点。
        
        输出 (Outputs):
            None: 该函数直接改变硬件状态，不返回特定值。
        """
        # 1. 刷新当前状态：在执行任何移动前，先获取机器人最新的实时坐标
        self._update_currpos()
        
        # 2. 控制模式切换 (TODO):
        # 在重置过程中，通常需要将机器人从“柔顺模式(Compliance)”切换到“高精度模式(Precision)”。
        # 高精度模式拥有更高的 PID 增益，能确保重置位置的绝对准确。
        time.sleep(0.5)

        # 3. 处理关节空间重置逻辑
        if joint_reset:
            print("JOINT RESET (TODO)")
            # 这里的 TODO 需要调用 Kinova API 的 BaseClient.ExecuteAction(Home_Action)
            # 或者使用之前定义的 self.robot.joint_reset()
            self.robot.joint_reset()
            time.sleep(0.5)

        # 4. 笛卡尔空间重置 (移动到 Reset Pose)
        # 根据配置决定是回到固定点，还是在一个范围内随机重置（用于增加环境泛化性）
        if self.randomreset:
             # --- 随机化重置逻辑 (待实现) ---
             # 逻辑通常是：在 self.resetpos 的基础上，对 XY 坐标和 Z 轴旋转角加上随机偏移
             # 这能防止模型只学会从一个绝对死板的起点开始任务
             pass
        else:
             # --- 固定点重置逻辑 ---
             # 拷贝预设的重置位姿 [x, y, z, qx, qy, qz, qw]
             target = self.resetpos.copy()
             
             # 调用底层控制接口发送位姿指令
             # 注意：由于此处没有使用平滑插值(interpolation)，机械臂会根据驱动器内部规划直接冲向目标
             self._send_pos_command(target)
             
             # 设置一个硬延迟 (Hard Sleep)，预留足够的时间让物理机械臂完成这段路程
             # 2.0秒是一个保守估计值，确保移动完全停止后再进行下一步
             time.sleep(2.0) 

        # 5. 恢复运行模式 (TODO):
        # 重置完成后，必须切换回“柔顺模式(Compliance Mode)”或“力控模式”。
        # 这样在模型开始进行随机探索时，如果发生轻微碰撞，机械臂会表现出弹性，保护硬件不损坏。
        pass

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
        
        # 断开机器人连接
        if hasattr(self, 'robot'):
            self.robot.disconnect()

# %% [markdown]
# ### 基础配置类定义
# 所有的测试函数都会引用这个 TestConfig

import numpy as np
import matplotlib.pyplot as plt
import time

class TestConfig(DefaultEnvConfig):
    SERVER_IP = "192.168.8.10"
    # 摄像头配置：请确保索引与你的硬件一致
    REALSENSE_CAMERAS = {
        "side_1": {"index": 4},
        "wrist_1": {"index": 6}
    }
    # 填入之前的重置位姿 [x, y, z, roll, pitch, yaw]
    RESET_POSE = np.array([
        0.325, -0.042, 0.245,           
        np.deg2rad(175.55),             
        np.deg2rad(3.03),               
        np.deg2rad(92.66)               
    ])
    TARGET_POSE = RESET_POSE.copy()
    # 安全边界：限制在一个 10cm x 10cm x 10cm 的小盒子里进行测试
    ABS_POSE_LIMIT_LOW = RESET_POSE[:6] - np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])
    ABS_POSE_LIMIT_HIGH = RESET_POSE[:6] + np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])
    # 动作缩放：[位置缩放, 旋转缩放, 夹爪缩放]
    # 如果 ACTION_SCALE[0] = 0.01，step(action=1.0) 实际移动 1cm
    ACTION_SCALE = np.array([0.01, 0.05, 1.0]) 
    MAX_EPISODE_LENGTH = 50
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    GRIPPER_SLEEP = 0.5

# %% [markdown]
# ### 测试 1: 基础重置与图像获取 (Visual Check)
# 验证图像是否正常显示，状态是否初始化

def test_reset_and_visual():
    print("=== 开始测试: Reset & Visual ===")
    env = None
    try:
        env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        obs, _ = env.reset(joint_reset=True)
        print("✅ 环境重置成功")

        if 'images' in obs and obs['images']:
            num_cams = len(obs['images'])
            fig, axes = plt.subplots(1, num_cams, figsize=(5 * num_cams, 5))
            if num_cams == 1: axes = [axes]
            for ax, (name, img) in zip(axes, obs['images'].items()):
                ax.imshow(img)
                ax.set_title(f"{name}\n{img.shape}")
                ax.axis('off')
            plt.show()
        
        print(f"当前位姿 (XYZ): {obs['state']['tcp_pose'][:3]}")
        print(f"夹爪位置: {obs['state']['gripper_pose']}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        if env: env.close()
    print("=== 测试结束 ===\n")

# test_reset_and_visual()

# %% [markdown]
# ### 测试 2: Step 动作控制 (Delta Movement)
# 验证发送相对动作（例如向上移动）机械臂是否有物理反馈

def test_step_movement():
    print("=== 开始测试: Step Movement (向上微移) ===")
    env = None
    try:
        env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        env.reset()
        
        # 获取初始位置
        start_z = env.currpos[2]
        print(f"初始 Z 轴高度: {start_z:.4f}")

        # 构造动作：[dx, dy, dz, dr, dp, dy, gripper]
        # 让 dz = 1.0，配合 ACTION_SCALE[0]=0.01，预期向上移动 1cm
        action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        
        print("正在执行 5 步向上动作...")
        for i in range(5):
            print()
            obs, reward, done, _, info = env.step(action)
            current_z = obs['state']['tcp_pose'][2]
            print(f"Step {i+1}: 当前 Z = {current_z:.4f} (增量: {(current_z - start_z)*100:.2f} cm)")
            time.sleep(0.1)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        if env: env.close()
    print("=== 测试结束 ===\n")

# test_step_movement()

# %% [markdown]
# ### 测试 3: 夹爪二值化控制 (Gripper Binary Control)
# 验证夹爪的开合逻辑

def test_gripper_control():
    print("=== 开始测试: Gripper Binary Control ===")
    env = None
    try:
        env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        env.reset()

        print("正在尝试 [close] 夹爪...")
        # 第 7 维 >= 0.5 触发关闭
        close_action = np.array([0, 0, 0, 0, 0, 0, 1.0])
        for _ in range(3):
            print("----------------------------------start----------------------------------")
            obs, _, _, _, _ = env.step(close_action)
            print(f"start waiting 5s")
            print(f"夹爪位置反馈: {obs['state']['gripper_pose']}")
            print("----------------------------------end----------------------------------")
            print()
            time.sleep(5)
        print(f"close后夹爪位置反馈: {obs['state']['gripper_pose']}")


        print("正在尝试 [open] 夹爪...")
        # 第 7 维 <= -0.5 触发open
        open_action = np.array([0, 0, 0, 0, 0, 0, -1.0])
        for _ in range(3): 
            print("----------------------------------start----------------------------------")
            obs, _, _, _, _ = env.step(open_action)
            print("----------------------------------end----------------------------------")
            print()
            time.sleep(5)
        print(f"open后夹爪位置反馈: {obs['state']['gripper_pose']}")

        time.sleep(5)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        if env: env.close()
    print("=== 测试结束 ===\n")

test_gripper_control()

# %% [markdown]
# ### 测试 4: 安全边界截断 (Safety Box Clipping)
# 验证当动作尝试超出限位时，`clip_safety_box` 是否生效

def test_safety_boundary():
    print("=== 开始测试: Safety Boundary Clipping ===")
    env = None
    try:
        env = KinovaEnv(hz=10, config=TestConfig(), fake_env=False)
        env.reset()
        
        # 尝试一个非常大的 X 方向动作，预期会被截断在 RESET_POSE_X + 0.05
        print(f"安全上限 X: {TestConfig.ABS_POSE_LIMIT_HIGH[0]:.4f}")
        
        huge_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print("连续执行大跨度动作...")
        for i in range(10):
            obs, _, _, _, _ = env.step(huge_action)
            print(f"Step {i}: 实际 X 位置 = {obs['state']['tcp_pose'][0]:.4f}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        if env: env.close()
    print("=== 测试结束 ===\n")

test_safety_boundary()