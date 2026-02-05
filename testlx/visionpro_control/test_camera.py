from avp_stream import VisionProStreamer
import time

# 1. 填入你 Vision Pro App 界面上显示的 IP 地址
avp_ip = "192.168.1.163" 
s = VisionProStreamer(ip=avp_ip)

# 2. 配置视频流
# device="/dev/video2" 指向你的摄像头
# size 建议根据你的摄像头实际分辨率调整，fps 设为 30 或 60
s.configure_video(device="/dev/video2", format="v4l2", size="1280x720", fps=30)

# 3. 启动 WebRTC 传输
s.start_webrtc()

print("🚀 视频流已启动，请查收 Vision Pro 画面...")

try:
    while True:
        # 获取追踪数据（即便你只想看画面，也需要调用 get_latest 来维持连接）
        r = s.get_latest()
        
        if r:
            # 你可以在这里打印一点东西确认连接正常
            print(f"\r正在接收追踪数据... 头部位置: {r['head'][0, :3, 3]}", end="")
            
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n停止传输。")