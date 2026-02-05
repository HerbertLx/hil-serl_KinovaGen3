from typing import OrderedDict
from kinova_env.camera.rs_capture import RSCapture
from kinova_env.camera.video_capture import VideoCapture
from kinova_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
from kinova_env.envs.kinova_env import KinovaEnv
import cv2

class USBCaptureAdapter:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.cap = cv2.VideoCapture(index)
        # 关键设置：将缓冲区设为1，保证拿到的图像是实时的，而不是几帧前的缓存
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print(f"⚠️ 警告: 无法打开摄像头 {name} (index {index})")

    def read(self):

        return self.cap.read()

    def close(self):
        if self.cap:
            self.cap.release()

class USBEnv(KinovaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_config_dict=None):
            """初始化普通 USB 摄像头."""
            print(f"Initializing USB cameras...")
            
            # 1. 如果已有相机正在运行，先关闭它们
            if hasattr(self, 'cap') and self.cap is not None:
                self.close_cameras()

            self.cap = OrderedDict()

            # 2. 遍历配置字典 (side_1, wrist_1 等)
            for cam_name, cfg in name_config_dict.items():
                # 特殊逻辑：如果是 side_classifier，通常复用 side_policy 的流，避免重复开启硬件
                if cam_name == "side_classifier" and "side_policy" in self.cap:
                    self.cap["side_classifier"] = self.cap["side_policy"]
                else:
                    # 检查配置中是否有 index
                    if "index" in cfg:
                        index = cfg["index"]
                        print(f"  - Opening {cam_name} at index {index}")
                        
                        # 使用 VideoCapture 包装 USBCaptureAdapter
                        # VideoCapture 通常是项目里用来开启独立线程读取图像的装饰器
                        cap = VideoCapture(
                            USBCaptureAdapter(cam_name, index)
                        )
                        self.cap[cam_name] = cap
                    else:
                        print(f"  - Skip {cam_name}: No index provided in config.")

    def reset(self, **kwargs):

        if self.save_video:
            self.save_video_recording() # 如果开启了录制，保存上一回合视频

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        # time.sleep(0.1)
        # requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        self._send_gripper_command(1.0)
        time.sleep(1)
        
        # Move above the target pose
        # target = copy.deepcopy(self.currpos)
        # target[2] = self.config.TARGET_POSE[2] + 0.05
        # self.interpolate_move(target, timeout=0.5)
        # time.sleep(0.5)
        # self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)
        # time.sleep(0.5)

        # self._update_currpos()
        # reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        # reset_pose[1] += 0.04
        # self.interpolate_move(reset_pose, timeout=0.5)

        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04  # 在 Z 轴方向比目标复位点再高 4cm
        self.interpolate_move(reset_pose, timeout=1) # 插值平滑移动

        obs, info = super().reset(**kwargs)
        self._send_gripper_command(1.0)
        time.sleep(1)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, info
    
    # def interpolate_move(self, goal: np.ndarray, timeout: float):
    #     """Move the robot to the goal position with linear interpolation."""
    #     if goal.shape == (6,):
    #         goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
    #     self._send_pos_command(goal)
    #     time.sleep(timeout)
    #     self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            print("\n执行关节空间回归...")

            home_joints = [355.22, 4.94, 190.63, 241.29, 181.31, 51.82, 277.83, 100.0]
            self.robot.move_angular(home_joints, dual_grip=False)
            time.sleep(2)
            print(f"关节回归完成。\n")

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
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
