import multiprocessing
import numpy as np
from kinova_env.spacemouse import pyspacemouse
from typing import Tuple

class SpaceMouseExpert:
    """
    SpaceMouseExpert 类:
    负责管理 SpaceMouse 硬件连接并提供实时动作查询接口。
    
    核心逻辑:
    1. 在初始化时开启一个独立的守护进程（Daemon Process）。
    2. 该进程持续循环读取 SpaceMouse 的位移和旋转传感器数据。
    3. 利用 multiprocessing.Manager 在主进程和读取进程间共享最新的动作数据。
    4. 自动处理单只或双只 SpaceMouse 的数据适配。
    """

    def __init__(self):
        """
        初始化硬件连接与共享内存。
        
        职能:
        - 打开 SpaceMouse 设备。
        - 建立基于 Manager 的进程间共享字典。
        - 启动后台读取进程。
        """
        pyspacemouse.open()

        # 使用 Manager.dict() 建立多进程安全的数据共享容器
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        
        # 初始化默认数据：6自由度位移/旋转，以及按键状态
        self.latest_data["action"] = [0.0] * 6  
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # 启动后台守护进程，确保硬件读取不会阻塞算法主逻辑
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True # 随主进程一同销毁
        self.process.start()

    def _read_spacemouse(self):
        """
        后台私有函数：持续从硬件循环读取原始数据。
        
        职能:
        - 无限循环读取传感器的 6 自由度 (X, Y, Z, Roll, Pitch, Yaw)。
        - 对坐标系进行修正（如 Y 轴取反）以适配机器人笛卡尔空间。
        - 将结果实时同步至共享字典 self.latest_data。
        
        输入: 无 (从硬件设备读取)
        输出: None (直接写入共享变量)
        """
        while True:
            # 读取当前所有已连接设备的瞬时状态
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            # 逻辑: 处理连接了两只 SpaceMouse 的情况 (通常用于双臂控制)
            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
                
            # 逻辑: 处理连接单只 SpaceMouse 的情况 (标准配置)
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            # 将解析后的物理动作更新至共享进程空间
            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """
        公有接口：获取 SpaceMouse 的最新状态。
        
        职能:
        - 供外部主逻辑调用，返回当前最新的传感器读数和按键点击情况。

        输入: 无
        输出: 
            - action: np.ndarray, 包含 [dx, dy, dz, droll, dpitch, dyaw] 的动作增量。
            - buttons: list, 代表鼠标按键状态的布尔值列表（如 [左键, 右键]）。
        """
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        """
        资源释放职能:
        - 强行终止后台读取进程，防止产生僵尸进程。
        """
        self.process.terminate()

'''
import multiprocessing
import numpy as np
from kinova_env.spacemouse import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            # Update the shared state
            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        # pyspacemouse.close()
        self.process.terminate()
'''