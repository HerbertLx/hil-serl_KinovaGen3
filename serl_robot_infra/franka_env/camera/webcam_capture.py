import cv2
import numpy as np
import time

class WebcamCapture:
    def __init__(self, name, device_id, dim=(640, 480), fps=30, exposure=None, focus=None, **kwargs):
        """
        device_id: 整数，例如 0, 2, 4 (对应 /dev/video0 等)
        dim: 分辨率 (width, height)
        exposure: 手动曝光值 (0-100 或具体数值，取决于相机驱动)
        focus: 手动焦距 (0-255，取决于相机驱动)
        """
        self.name = name
        self.device_id = device_id
        
        # 打开相机
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开 Webcam 设备 ID: {device_id}")

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, dim[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dim[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # === 关键：针对运动中的腕部相机，必须禁用“自动”功能 ===
        # 1. 禁用自动对焦 (通常 0 是关, 1 是开，但也取决于驱动)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
        if focus is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, focus)
            
        # 2. 禁用自动曝光 (1: Manual, 3: Auto 常见于 v4l2)
        # 注意：不同相机的标志位定义极其混乱，如果报错或无效，建议在终端用 v4l2-ctl 设置
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        # 3. 极小的缓冲区，防止延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 预热：丢弃前几帧，让相机光圈稳定
        for _ in range(5):
            self.cap.read()
            
        print(f"Webcam {name} (ID: {device_id}) initialized.")

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            # Realsense 输出的是 RGB，但 OpenCV 默认是 BGR
            # 为了数据一致性，这里必须转成 RGB！
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame
        else:
            return False, None

    def close(self):
        if self.cap.isOpened():
            self.cap.release()