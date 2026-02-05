# %%
import multiprocessing
import numpy as np
from pynput import keyboard
from typing import Tuple

class KeyBoardExpert:
    """
    KeyboardExpert 类:
    通过键盘模拟 SpaceMouse 的 6 自由度控制接口。
    
    职能:
    - 异步监听键盘按键，将离散的按键转换为平滑的动作向量。
    - 保持与 SpaceMouseExpert 完全一致的 get_action 接口。
    """

    def __init__(self):
        """
        初始化键盘监听器与共享内存。
        """
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        
        # 初始化状态: 6维动作 [x, y, z, roll, pitch, yaw] + 4个按键位
        self.latest_data["action"] = [0.0] * 6  
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # 启动后台读取进程
        self.process = multiprocessing.Process(target=self._read_keyboard)
        self.process.daemon = True
        self.process.start()

    def _read_keyboard(self):
        """
        后台私有函数: 持续监听键盘按键并映射为机器人动作。
        
        职能:
        - 维护一个当前按下键的集合。
        - 根据定义的映射表生成 [-1, 0, 1] 的动作向量。
        """
        pressed_keys = set()

        def on_press(key):
            try:
                if hasattr(key, 'char'):
                    pressed_keys.add(key.char.lower())
                else:
                    pressed_keys.add(key.name)
            except: pass

        def on_release(key):
            try:
                if hasattr(key, 'char'):
                    pressed_keys.discard(key.char.lower())
                else:
                    pressed_keys.discard(key.name)
            except: pass

        # 启动非阻塞监听器
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while True:
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            # --- 位置映射 ---
            if 'w' in pressed_keys: action[0] = 1.0   # x+
            if 's' in pressed_keys: action[0] = -1.0  # x-
            if 'a' in pressed_keys: action[1] = 1.0   # y+
            if 'd' in pressed_keys: action[1] = -1.0  # y-
            if 'q' in pressed_keys: action[2] = 1.0   # z+
            if 'e' in pressed_keys: action[2] = -1.0  # z-

            # --- 角度映射 ---
            if 'l' in pressed_keys: action[3] = 1.0   # roll+
            if 'j' in pressed_keys: action[3] = -1.0  # roll-
            if 'k' in pressed_keys: action[4] = 1.0   # pitch+
            if 'i' in pressed_keys: action[4] = -1.0  # pitch-
            if 'o' in pressed_keys: action[5] = 1.0   # yaw+
            if 'u' in pressed_keys: action[5] = -1.0  # yaw-

            # --- 模拟按钮 (例如: F1/F2 对应左/右键) ---
            if 'n' in pressed_keys: buttons[0] = 1
            if 'm' in pressed_keys: buttons[1] = 1

            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons
            import time
            time.sleep(0.01) # 约 100Hz 的刷新率

    def get_action(self) -> Tuple[np.ndarray, list]:
        """
        获取键盘模拟的最新动作增量。
        :return: (action_array, buttons_list)
        """
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action, dtype=np.float32), buttons
    
    def close(self):
        """释放资源"""
        self.process.terminate()

# %% KeyboardExpert 实时测试工具
# 运行此单元格后，按下 A/D/W/S/Q/E 或 J/K/L/I/U/O 键查看输出。
# 按下 Ctrl+C 或点击停止按钮结束。

# %%mnmndndm
import time

def test_keyboard_expert():
    # 1. 实例化
    kb_expert = KeyBoardExpert()
    print("Keyboard Expert 已启动！")
    print("控制提示: WASD (XY), QE (Z), IJKL (Roll/Pitch), UO (Yaw)")
    
    try:
        while True:
            # 2. 调用标准接口
            action, buttons = kb_expert.get_action()
            
            # 3. 只有当有按键按下时才打印，保持界面干净
            if np.any(action != 0) or any(buttons):
                # 格式化输出: 前三位位移，后三位旋转
                pos_str = f"Pos: [X:{action[0]:>2}, Y:{action[1]:>2}, Z:{action[2]:>2}]"
                rot_str = f"Rot: [R:{action[3]:>2}, P:{action[4]:>2}, Y:{action[5]:>2}]"
                print(f"\r{pos_str} | {rot_str} | Buttons: {buttons}", end="")
            
            time.sleep(0.1) # 10Hz 显示频率
            
    except KeyboardInterrupt:
        print("\n测试停止")
    finally:
        kb_expert.close()

# 运行测试
# test_keyboard_expert()