# %% environment path imported by me
import os
import sys
target_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl"
if os.path.exists(target_path) and target_path not in sys.path:
    sys.path.insert(0, target_path)
    print(f"✅ 成功添加 hil-serl 路径: {target_path}")

# %%
import copy
import time
from kinova_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from pynput import keyboard

from kinova_env.envs.kinova_env import DefaultEnvConfig, KinovaEnv

class APPLEEnv(KinovaEnv):
    """
    APPLEEnv 类：专为 APPLE 放置任务定制的任务环境。
    主要增加了 F1 键触发重新抓取 (Regrasp) 的功能，并优化了复位流程。
    """
    def __init__(self, **kwargs):
        """
        初始化 RAM 任务环境。
        :param kwargs: 传递给父类 KinovaEnv 的配置参数。
        """
        super().__init__(**kwargs)
        self.should_regrasp = False  # 标志位：是否需要在下次 reset 时执行重新抓取逻辑
        
        # 定义键盘按下事件的处理函数
        def on_press(key):
            # 如果按下 F1 键，将标志位设为 True
            if str(key) == "Key.f1":
                print("触发重新抓取 (Regrasp) 任务...")
                self.should_regrasp = True

        # 启动后台键盘监听器（非阻塞）
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def go_to_reset(self, joint_reset=False):
        """
        功能：将机械臂移动到预设的复位位置 (Reset Position)。
        
        过程：
        1. 切换到“精准模式 (PRECISION)”。
        2. 先垂直向上抬起 (避免碰撞)。
        3. 如果需要，执行关节空间复位 (Joint Reset)。
        4. 移动到目标位姿 (笛卡尔空间)。
        5. 切换回“顺应模式 (COMPLIANCE)”。

        输入格式:
        :param joint_reset: bool，是否强制执行关节回归初始角度（通常用于解决奇异位姿）。
        输出格式: None
        """        
        # 1. 更新当前位置并锁定当前位姿，防止意外滑动
        self._update_currpos()
        self._send_pos_command(self.currpos)
        # 通过 HTTP API 更新控制器参数为精准控制模式
        # requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # 2. 向上抬起：为了防止复位过程中撞到桌面的物体
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04  # 在 Z 轴方向比目标复位点再高 4cm
        self.interpolate_move(reset_pose, timeout=1) # 插值平滑移动

        # 3. 关节复位
        if joint_reset:
            print("\n执行关节空间回归...")

            home_joints = [355.22, 4.94, 190.63, 241.29, 181.31, 51.82, 277.83, 100.0]
            self.robot.move_angular(home_joints, dual_grip=False)
            time.sleep(2)
            print(f"关节回归完成。\n")


        # 4. 执行笛卡尔复位
        if self.randomreset:  # 如果配置了随机复位（用于增强算法鲁棒性）
            reset_pose = self.resetpos.copy()
            # 在 XY 平面上添加随机偏移
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            # 在偏航角 (Yaw) 上添加随机旋转
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random) # 欧拉角转四元数
            self._send_pos_command(reset_pose)
        else:
            # 否则移动到固定的 resetpos
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)
        
        # time.sleep(0.5)

        # 5. 切换到顺应模式，准备开始 RL 任务
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def regrasp(self):
        """
        功能：执行完整的“重新抓取”动作序列（通常用于 RAM 掉落或位置偏离）。
        
        过程：
        1. 抬起机械臂。
        2. 提示用户放好 RAM。
        3. 移动到抓取位姿。
        4. 缓慢闭合夹爪并抬起。
        
        输入格式: 无
        输出格式: None
        """
        # 步骤 A: 切换到精准模式并抬起
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        # requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(reset_pose, timeout=1)

        # 步骤 B: 释放夹爪
        input("按下回车 [Enter] 以打开夹爪...")
        self._send_gripper_command(1.0) # 1.0 通常代表完全打开
        
        # 步骤 C: 移动到抓取点上方
        input("请将 RAM 放入夹具中，然后按下回车以抓取...")
        top_pose = self.config.GRASP_POSE.copy()
        top_pose[2] += 0.05  # 抓取点上方 5cm
        top_pose[0] += np.random.uniform(-0.005, 0.005) # X 轴加点随机微调
        self.interpolate_move(top_pose, timeout=1)
        time.sleep(0.5)

        # 步骤 D: 下降并抓取
        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.05
        self.interpolate_move(grasp_pose, timeout=0.5)


        # 调用后端 API 执行缓慢闭合动作（保证抓取稳固）
        # requests.post(self.url + "close_gripper_slow")
        self._send_gripper_command(-1.0, mode="binary") # close the gripper
        self.last_gripper_act = time.time()

        # 步骤 E: 回到复位位置
        self.interpolate_move(top_pose, timeout=0.5)
        time.sleep(0.2)
        self.interpolate_move(self.config.RESET_POSE, timeout=1)
        time.sleep(0.5)

    def reset(self, joint_reset=False, **kwargs):
        """
        功能：环境重置主函数，这是标准的 Gymnasium 接口。
        
        输入格式:
        :param joint_reset: bool，是否执行关节复位。
        :param kwargs: 其他可能传递给 reset 的参数。
        
        输出格式:
        :return obs: dict，当前的观测值（图像 + 机器人状态）。
        :return info: dict，附加信息（如重置原因等）。
        """
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording() # 如果开启了录制，保存上一回合视频

        # 核心逻辑：检查是否需要执行特殊的“重新抓取”
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False # 执行完后重置标志位

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True
        
        # 执行常规复位
        self._recover() # 内部辅助函数，通常用于清除错误状态
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0 # 回合步数清零

        # 获取重置后的第一帧观测数据
        self._update_currpos()
        obs = self._get_obs()
        
        # 任务开始前切换回顺应模式
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

        self.terminate = False
        return obs, {}
    
# %% [markdown]
# ### APPLE 放置任务环境测试脚本 (APPLEEnv Test)
# =========================================================

import numpy as np
import time
import cv2


# --- 1. 补全配置参数 ---
# 如果 DefaultEnvConfig 中缺少以下项，请根据实际情况补充
class TestConfig(DefaultEnvConfig):
    SERVER_IP = "192.168.8.10"
    REALSENSE_CAMERAS = {
        "side_1": {"index": 4},
        "wrist_1": {"index": 6}
    }
    
    # 初始位姿 [x, y, z, r, p, y] (单位: m, rad)
    RESET_POSE = np.array([
        0.300, -0.000, 0.250,           
        np.deg2rad(180.00), np.deg2rad(0.00), np.deg2rad(90.00)               
    ])
    
    TARGET_POSE = RESET_POSE.copy()

    # 安全限位
    ABS_POSE_LIMIT_LOW = np.array([0.240, -0.270, 0.040, RESET_POSE[3]-1.0, RESET_POSE[4]-1.0, RESET_POSE[5]-1.0])
    ABS_POSE_LIMIT_HIGH = np.array([0.500, 0.270, 0.500, RESET_POSE[3]+1.0, RESET_POSE[4]+1.0, RESET_POSE[5]+1.0])

    # 动作缩放
    ACTION_SCALE = np.array([0.02, 0.05, 1.0])

    # --- 补充 APPLEEnv 专用参数 ---
    # 抓取位姿 (用于 regrasp 函数，请根据苹果托盘的实际物理坐标修改)
    GRASP_POSE = np.array([0.400, 0.200, 0.197, np.deg2rad(180), 0, np.deg2rad(90)])
    
    # 控制器参数 (如果不需要 HTTP 更新可留空，或在 APPLEEnv 中注释掉相关行)
    PRECISION_PARAM = {"stiffness": [600, 600, 600, 50, 50, 50]}
    COMPLIANCE_PARAM = {"stiffness": [200, 200, 200, 20, 20, 20]}

    # 其他必要参数
    MAX_EPISODE_LENGTH = 100
    DISPLAY_IMAGE = True
    RANDOM_RESET = False # 初始测试建议先关掉随机复位
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    JOINT_RESET_PERIOD = 0

# %% 2. 核心测试逻辑 ---

def test_apple_env():
    print("=== 初始化 APPLE 任务环境 ===")
    env = None
    try:
        # 实例化环境
        env = APPLEEnv(hz=10, config=TestConfig(), fake_env=False)
        
        # --- 测试 1: 常规 Reset ---
        print("\n[测试 1] 执行常规 Reset...")
        obs, info = env.reset()
        print(f"✅ Reset 完成。当前 TCP 位置: {obs['state']['tcp_pose'][:3]}")

        # --- 测试 2: 键盘触发 Regrasp ---
        print("\n[测试 2] 键盘交互测试 (Regrasp)")
        print(">>> 请在 5 秒内按下键盘上的 [F1] 键来触发重新抓取逻辑...")
        
        countdown = 5
        while countdown > 0:
            if env.should_regrasp:
                print("🔥 检测到 F1 按下，标志位已更新！")
                break
            print(f"倒计时: {countdown}...")
            time.sleep(1)
            countdown -= 1
        
        if env.should_regrasp:
            print(">>> 正在调用 reset()，预期会先进入 regrasp() 流程...")
            # 此时 reset 内部会检查 should_regrasp 并执行 regrasp()
            obs, info = env.reset()
            print("✅ 重新抓取流程结束。")
        else:
            print("⌛ 未检测到按键，跳过本次抓取测试。")

        time.sleep(20)

        # --- 测试 3: 动作步进 (Step) 与安全边界 ---
        # print("\n[测试 3] 执行 10 步 Step (向下移动)...")
        # for i in range(10):
        #     # 动作格式: [dx, dy, dz, dr, dp, dy, gripper]
        #     # 这里设置 dz = -1.0 (最大速度向下)
        #     action = np.array([0.0, 0.0, -1.1, 0.0, 0.0, 0.0, 1.0])
        #     obs, reward, done, truncated, info = env.step(action)
        #     z_val = obs['state']['tcp_pose'][2]
        #     print(f"Step {i+1}: 当前 Z 轴 = {z_val:.4f}")
            
        #     if done:
        #         print("任务终止或触碰边界。")
        #         break
        #     time.sleep(0.1)

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if env:
            print("\n关闭环境中...")
            env.close()
    print("=== 测试结束 ===")

# test_apple_env()