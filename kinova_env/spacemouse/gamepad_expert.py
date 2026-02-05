import multiprocessing
import numpy as np
import time
import os

# 屏蔽 pygame 的欢迎信息
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class GamepadExpert:
    """
    GamepadExpert 类
    
    映射逻辑 (基于 A0-A5 映射):
    - 左摇杆 (A0, A1): 左右->Y轴, 上下->X轴 (平移)
    - 扳机键 (A2, A5): LT下压->Z轴下降, RT下压->Z轴上升 (平移)
    - 右摇杆 (A3, A4): 左右->Roll翻滚, 上下->Pitch俯仰 (旋转)
    - 十字键 (Hat 0): 左右控制 Yaw 偏航 (绕 Z 轴旋转)
    - 按钮 (B0, B1): A键闭合, B键张开 (夹爪)
    """

    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        
        # 6维动作 [x, y, z, roll, pitch, yaw] + 按钮位
        self.latest_data["action"] = [0.0] * 6  
        self.latest_data["buttons"] = [0, 0, 0, 0]

        self.process = multiprocessing.Process(target=self._read_gamepad)
        self.process.daemon = True
        self.process.start()

    def _read_gamepad(self):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            print("❌ 子进程未检测到手柄")
            return

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        DEADZONE = 0.12

        while True:
            pygame.event.pump()
            
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            # --- 1. 位置平移 (X, Y, Z) ---
            # A1: 左摇杆上下 -> X
            # A0: 左摇杆左右 -> Y
            raw_ly = -joystick.get_axis(1) 
            raw_lx = -joystick.get_axis(0)
            action[0] = (raw_ly**3) if abs(raw_ly) > DEADZONE else 0.0
            action[1] = (raw_lx**3) if abs(raw_lx) > DEADZONE else 0.0

            # A2 (LT) 下降, A5 (RT) 上升 -> Z
            lt = (joystick.get_axis(2) + 1.0) / 2.0
            rt = (joystick.get_axis(5) + 1.0) / 2.0
            action[2] = rt - lt

            # --- 2. 姿态旋转 (Roll, Pitch, Yaw) ---
            # A4: 右摇杆上下 -> Pitch (俯仰)
            raw_ry = -joystick.get_axis(4)
            action[4] = (raw_ry**3) if abs(raw_ry) > DEADZONE else 0.0

            # --- 修改部分 ---
            # A3: 右摇杆左右 -> 现在控制 Roll (翻滚)
            raw_rx = joystick.get_axis(3)
            action[3] = (raw_rx**3) if abs(raw_rx) > DEADZONE else 0.0

            # Hat 0: 十字键左右 -> 现在控制 Yaw (绕 Z 轴旋转)
            hat = joystick.get_hat(0)
            action[5] = -float(hat[0]) 
            # ----------------

            # --- 3. 按钮映射 ---
            if joystick.get_button(0): buttons[0] = 1 # A键
            if joystick.get_button(1): buttons[1] = 1 # B键

            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons
            time.sleep(0.01)

    def get_action(self):
        return np.array(self.latest_data["action"], dtype=np.float32), self.latest_data["buttons"]
    
    def close(self):
        self.process.terminate()



def test_new_gamepad_mapping():
    gp = GamepadExpert()
    print("🚀 Xbox 手柄新映射测试启动 (按 Ctrl+C 退出)")
    print("-" * 60)
    print("控制检查单:")
    print("  左摇杆推上 -> X+ (前) | 左摇杆推左 -> Y+ (左)")
    print("  右扳机 (RT) -> Z+ (升) | 左扳机 (LT) -> Z- (降)")
    print("  右摇杆上下 -> Pitch | 右摇杆左右 -> Yaw")
    print("  十字键左右 -> Roll  | A/B键 -> Gripper")
    print("-" * 60)

    try:
        while True:
            action, btns = gp.get_action()
            
            # 只有当产生有效动作时才输出
            if np.max(np.abs(action)) > 0.05 or any(btns):
                out = [
                    f"X:{action[0]:>5.2f}", f"Y:{action[1]:>5.2f}", f"Z:{action[2]:>5.2f}",
                    f"R:{action[3]:>5.2f}", f"P:{action[4]:>5.2f}", f"Yw:{action[5]:>5.2f}",
                    f"A:{btns[0]} B:{btns[1]}"
                ]
                print(f"\r指令输出: {' | '.join(out)}", end="", flush=True)
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n测试结束")
    finally:
        gp.close()

# test_new_gamepad_mapping()