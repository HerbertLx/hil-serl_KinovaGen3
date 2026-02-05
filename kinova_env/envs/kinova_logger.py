import cv2
import numpy as np
import time
import os

default_log_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/examples/experiments/apple_put/log"

class KinovaLogger:
    def __init__(self, video_idx=0, save_path=default_log_path, fps=10):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # 视频参数
        self.fps = fps
        self.frame_size = (1280, 720)
        self.video_name = os.path.join(save_path, f"trial_{int(time.time())}.mp4")
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(video_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        
        # 视频写入器 (使用 XVID 或 mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.video_name, fourcc, self.fps, self.frame_size)
        
        print(f"🎬 Logger initialized. Recording to: {self.video_name}")

    def log_step(self, frame_raw, data_dict):
        """
        frame_raw: 如果你想用环境原本获取的图像，直接传进来；否则传 None 自动抓取
        data_dict: 包含需要打印的所有信息
        """
        if frame_raw is None:
            ret, frame = self.cap.read()
        else:
            frame = frame_raw.copy()

        if frame is None: return

        # 在画面上绘制背景半透明层
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 打印信息逻辑
        y_offset = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for key, value in data_dict.items():
            text = f"{key}: {value}"
            # 如果触发边界，文字变红
            color = (0, 0, 255) if "⚠️" in str(value) or "Safety" in key else (255, 255, 255)
            cv2.putText(frame, text, (20, y_offset), font, 0.6, color, 1, cv2.LINE_AA)
            y_offset += 25

        # 写入视频
        self.out.write(frame)
        return frame

    def release(self):
        self.cap.release()
        self.out.release()
        print("💾 Video saved and logger released.")