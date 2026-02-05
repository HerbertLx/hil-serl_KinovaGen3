import numpy as np
from scipy.spatial.transform import Rotation
from avp_stream import VisionProStreamer

class AVPAgent:
    def __init__(self, ip, scale_pos=1.5, scale_rot=1.0):
        print(f"Connecting to AVP at {ip}...")
        self.streamer = VisionProStreamer(ip=ip)
        self.last_matrix = None
        self.scale_pos = scale_pos
        self.scale_rot = scale_rot
        
        # ⚠️ 坐标系对齐矩阵 (需要你在实验室实测微调)
        # 假设: AVP Y朝上, -Z朝前; Robot Z朝上, X朝前
        self.align_matrix = np.array([
            [ 0,  0, -1], 
            [-1,  0,  0], 
            [ 0,  1,  0]
        ])

    def get_action(self):
        data = self.streamer.latest
        wrist_matrix = data['right_wrist'][0] # (4,4)
        
        # 初始化
        if self.last_matrix is None:
            self.last_matrix = wrist_matrix
            return np.zeros(7)

        # 1. 计算位置增量
        curr_pos = wrist_matrix[:3, 3]
        last_pos = self.last_matrix[:3, 3]
        delta_pos = self.align_matrix @ (curr_pos - last_pos)
        
        # 2. 计算旋转增量
        curr_rot = Rotation.from_matrix(wrist_matrix[:3, :3])
        last_rot = Rotation.from_matrix(self.last_matrix[:3, :3])
        diff_rot = curr_rot * last_rot.inv()
        delta_rot = self.align_matrix @ diff_rot.as_rotvec()

        # 3. 夹爪 (距离小于2cm算闭合)
        pinch = data['right_pinch_distance']
        gripper = 1.0 if pinch < 0.02 else -1.0

        self.last_matrix = wrist_matrix
        
        return np.concatenate([
            delta_pos * self.scale_pos, 
            delta_rot * self.scale_rot, 
            [gripper]
        ])

    def reset(self):
        self.last_matrix = None