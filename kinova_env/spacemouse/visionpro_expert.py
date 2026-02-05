import multiprocessing
import numpy as np
import time
import sys
from scipy.spatial.transform import Rotation as R

# 尝试引入 AVP 流
try:
    from avp_stream import VisionProStreamer
except ImportError:
    print("❌ 未检测到 avp_stream 库，请确保环境配置正确")

class VisionProExpert:
    """
    VisionProExpert 类 (仿照 GamepadExpert 设计)
    
    控制逻辑:
    - 激活机制: 左手捏合 (Left Pinch) 为离合器/死人开关
    - 右手位移: 控制 XYZ 平移 (已映射到机器人坐标系)
    - 右手姿态: 控制 Pitch (俯仰) 和 Yaw (偏航)
    - 左手姿态: Left Roll 控制 Roll (顺逆时针旋转)
    - 按钮位: 右手捏合 -> buttons[0]=1 (闭合), buttons[0]=0 (打开)
    """

    def __init__(self, avp_ip="192.168.1.223"):
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.avp_ip = avp_ip
        
        # 初始化数据结构 [x, y, z, roll, pitch, yaw]
        self.latest_data["action"] = [0.0] * 6  
        self.latest_data["buttons"] = [0] * 4 # buttons[0] 为夹爪
        self.latest_data["is_active"] = False

        # 配置参数
        self.config = {
            "SCALE_FACTOR": 0.5,
            "PINCH_THRESHOLD": 0.02,
            "NO_OBS_THRESHOLD": 0.0005,
            "LEFT_ROLL_CENTER": 3.0,
            "LEFT_ROLL_DEADZONE": 0.4,
            "LEFT_ROLL_SENSITIVITY": 30.0,
            "KP_POS": 2.5,
            "KP_ORI": 2.0
        }

        self.process = multiprocessing.Process(target=self._read_visionpro)
        self.process.daemon = True
        self.process.start()

    def _read_visionpro(self):
        """子进程：负责数据采集与速度向量计算"""
        streamer = VisionProStreamer(ip=self.avp_ip)
        streamer.start_webrtc()

        # 状态记忆
        clutch_engaged = False
        start_hand_pos = None
        start_hand_rot = None
        last_gripper_state = 0  # 0: Open, 1: Closed

        while True:
            r = streamer.get_latest()
            if not r:
                time.sleep(0.01)
                continue

            # --- 1. 离合器与激活判断 ---
            left_pinch = r['left_pinch_distance']
            is_active = False
            if left_pinch >= self.config["NO_OBS_THRESHOLD"]:
                is_active = left_pinch < self.config["PINCH_THRESHOLD"]

            # 提取右手实时位姿
            matrix = r['right_wrist'][0]
            curr_hand_pos = matrix[:3, 3]
            curr_hand_rot = R.from_matrix(matrix[:3, :3])

            action = [0.0] * 6
            buttons = [0] * 4

            if is_active:
                # 记录起始锚点
                if not clutch_engaged:
                    clutch_engaged = True
                    start_hand_pos = curr_hand_pos
                    start_hand_rot = curr_hand_rot
                    # 注意：由于此 Expert 模式不直接持有机器人状态，
                    # 速度计算将基于手部相对于锚点的相对偏移
                
                # --- 2. 位置平移 (右手) ---
                d_h = curr_hand_pos - start_hand_pos
                # 映射 AVP -> Robot: [dy, -dx, dz]
                d_robot = np.array([d_h[1], -d_h[0], d_h[2]]) * self.config["SCALE_FACTOR"]
                # P控制生成速度指令
                action[0:3] = d_robot * self.config["KP_POS"]

                # --- 3. 姿态旋转 (右手 Pitch/Yaw) ---
                delta_rot_hand = curr_hand_rot * start_hand_rot.inv()
                euler = delta_rot_hand.as_euler('xyz', degrees=True)
                # 映射到机器人 RPY: [Pitch, -Roll, 0] -> 这里根据你逻辑忽略手部自身Roll
                robot_euler = [euler[1], -euler[0], 0]
                error_rot = R.from_euler('xyz', robot_euler, degrees=True)
                ang_vel_raw = error_rot.as_rotvec(degrees=True) * self.config["KP_ORI"]
                action[3] = ang_vel_raw[0] # Pitch
                action[4] = ang_vel_raw[1] # Yaw

                # --- 4. 顺逆时针 (左手 Roll) ---
                left_roll = r['left_wrist_roll']
                if left_roll < 0: left_roll += 2 * np.pi
                roll_diff = left_roll - self.config["LEFT_ROLL_CENTER"]
                
                if abs(roll_diff) > self.config["LEFT_ROLL_DEADZONE"]:
                    # 向上为正 (顺时针)
                    action[5] = (self.config["LEFT_ROLL_CENTER"] - left_roll) * self.config["LEFT_ROLL_SENSITIVITY"]

                # --- 5. 夹爪逻辑 (右手捏合) ---
                right_pinch = r['right_pinch_distance']
                # 迟滞判断防止抖动
                if last_gripper_state == 1:
                    current_gripper = 1 if right_pinch <= self.config["PINCH_THRESHOLD"] + 0.01 else 0
                else:
                    current_gripper = 1 if right_pinch <= self.config["PINCH_THRESHOLD"] - 0.01 else 0
                
                buttons[0] = current_gripper
                last_gripper_state = current_gripper

            else:
                clutch_engaged = False
                start_hand_pos = None
                start_hand_rot = None

            # 更新数据
            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons
            self.latest_data["is_active"] = is_active
            time.sleep(0.02) # 50Hz

    def get_action(self):
        """获取当前 6-DOF 动作向量和按钮状态"""
        return np.array(self.latest_data["action"], dtype=np.float32), \
               self.latest_data["buttons"], \
               self.latest_data["is_active"]

    def close(self):
        self.process.terminate()

# ================= 测试代码 =================
def test_visionpro_expert():
    vp = VisionProExpert(avp_ip="192.168.1.223")
    print("🚀 Vision Pro Expert 专家模式测试启动")
    print("-" * 60)
    print("激活方式: 左手捏合")
    print("右手: 移动=平移, 翻转=Pitch/Yaw")
    print("左手: Roll=顺逆时针")
    print("-" * 60)

    try:
        while True:
            action, btns, active = vp.get_action()
            
            status_str = "🟢 [ACTIVE]" if active else "🔴 [IDLE]  "
            
            if active or np.max(np.abs(action)) > 0:
                out = [
                    f"X:{action[0]:>5.2f}", f"Y:{action[1]:>5.2f}", f"Z:{action[2]:>5.2f}",
                    f"P:{action[3]:>5.2f}", f"Yw:{action[4]:>5.2f}", f"Rz:{action[5]:>5.2f}",
                    f"Grip:{btns[0]}"
                ]
                print(f"\r{status_str} 指令: {' | '.join(out)}", end="", flush=True)
            else:
                print(f"\r{status_str} 等待左手激活...", end="", flush=True)
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n测试结束")
    finally:
        vp.close()

# test_visionpro_expert()