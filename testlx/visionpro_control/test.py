import avp_stream
from avp_stream import VisionProStreamer
import numpy as np
import time
import os

# 配置 Vision Pro IP 地址 (请确保手机 App 上显示的 IP 与此一致)
# Configure Vision Pro IP (Ensure this matches the IP shown in the AVP app)
avp_ip = "192.168.1.223" 
s = VisionProStreamer(ip=avp_ip)

# 配置从机器人摄像头到 Vision Pro 的视频流
# Configure video streaming from robot camera to Vision Pro
s.configure_video(device="/dev/video2", format="v4l2", size="1280x720", fps=30)
s.start_webrtc()

def format_matrix(matrix):
    """将矩阵格式化为位置和旋转矩阵 / Format matrix into position and rotation"""
    pos = matrix[0, :3, 3]
    rot = matrix[0, :3, :3]
    return pos, rot

print("\n" + "="*50)
print("🚀 Vision Pro 数据流已启动 / Data Stream Started")
print("="*50)

try:
    while True:
        r = s.get_latest()
        # for key in r:
        #     print()
        #     print(f"r['{key}'].shape = {r[key].shape}")
        #     print(f"r['{key}'] = ")
        #     print(r[key])
        # exit(0)
        if not r:
            continue

        # 清屏，使动态输出更整洁 (可选)
        # Clear screen to make output cleaner (Optional)
        # os.system('cls' if os.name == 'nt' else 'clear')

        print(f"\n🔔 [更新时间 / Update Time]: {time.strftime('%H:%M:%S')}")
        print("-" * 40)

        # 1. 头部姿态 / Head Pose
        head_pos, _ = format_matrix(r['head'])
        print(f"👤 【头部姿态 / Head Pose】")
        print(f"   位置 / Position (XYZ): {head_pos}")

        # 2. 手腕姿态 / Wrist Pose
        r_wrist_pos, _ = format_matrix(r['right_wrist'])
        l_wrist_pos, _ = format_matrix(r['left_wrist'])
        print(f"👋 【手腕位置 / Wrist Position】")
        print(f"   右腕 / Right (XYZ): {r_wrist_pos}")
        print(f"   左腕 / Left  (XYZ): {l_wrist_pos}")

        # 3. 捏合与旋转 / Pinch & Roll
        print(f"👌 【交互状态 / Interaction State】")
        print(f"   右手指尖距离 / Right Pinch Dist: {r['right_pinch_distance']:.4f}")
        print(f"   左手指尖距离 / Left Pinch Dist:  {r['left_pinch_distance']:.4f}")
        print(f"   右腕翻转角度 / Right Wrist Roll: {r['right_wrist_roll']:.4f} rad") # 0意味着虎口朝左，-1.5意味着虎口朝下，-3意味着虎口朝右
        print(f"   左腕翻转角度 / Left Wrist Roll:  {r['left_wrist_roll']:.4f} rad")  # 0意味着虎口朝左，1.5意味着虎口朝下，3意味着虎口朝右

        # 4. 手臂数据摘要 / Arm Data Summary
        # 这里打印矩阵的形状作为完整性检查
        print(f"💪 【骨骼数据摘要 / Skeleton Summary】")
        print(f"   右臂矩阵 / Right Arm Shape: {r['right_arm'].shape}")
        print(f"   右手关节 / Right Fingers: {len(r['right_fingers'])} joints")
        print(f"   左臂矩阵 / Left Arm Shape: {r['left_arm'].shape}")
        print(f"   左手关节 / Left Fingers: {len(r['left_fingers'])} joints")

        print("-" * 40)
        
        # 频率控制，避免打印过快
        # Frequency control to prevent excessive scrolling
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 停止流传输 / Stopping Stream...")

'''
r['left_wrist'].shape = (1, 4, 4)
r['left_wrist'] = 
[[[ 0.15685628  0.77094048 -0.6172899  -0.10801385]
  [ 0.85154372  0.21102872  0.47993749  0.30220568]
  [ 0.50026923 -0.60093057 -0.62338847  0.85352474]
  [ 0.          0.          0.          1.        ]]]

r['right_wrist'].shape = (1, 4, 4)
r['right_wrist'] = 
[[[ 0.18927515  0.7007187  -0.68787211  0.30336526]
  [-0.92839503 -0.10043176 -0.35776493  0.33650339]
  [-0.31977689  0.70633316  0.63153464  0.85600209]
  [ 0.          0.          0.          1.        ]]]

r['left_fingers'].shape = (25, 4, 4)
r['left_fingers'] = 
[[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
......
 [[-4.88824695e-01 -8.28857124e-01 -2.72110850e-01  9.15177912e-02]
  [ 8.56937885e-01 -5.14645457e-01  2.82059014e-02  4.78715859e-02]
  [-1.63419411e-01 -2.19394550e-01  9.61851537e-01  3.98406461e-02]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]]

r['right_fingers'].shape = (25, 4, 4)
r['right_fingers'] = 
[[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
......
 [[-9.51477513e-02 -9.75526989e-01 -1.98222548e-01 -1.21154726e-01]
  [ 9.92764056e-01 -1.07640974e-01  5.32102585e-02 -4.70620431e-02]
  [-7.32449964e-02 -1.91725537e-01  9.78710711e-01 -3.16581428e-02]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]]

r['left_arm'].shape = (27, 4, 4)
r['left_arm'] = 
[[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
.....
 [[-9.14927900e-01 -1.35122776e-01 -3.80327016e-01 -2.20164761e-01]
  [-1.24771900e-01  9.90828812e-01 -5.18665202e-02 -3.00245751e-02]
  [ 3.83847326e-01 -2.38889797e-10 -9.23396468e-01  9.23675895e-02]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]]

r['right_arm'].shape = (27, 4, 4)
r['right_arm'] = 
[[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
......
 [[-9.70012724e-01 -1.14400305e-01 -2.14446500e-01  2.24893540e-01]
  [-1.11703150e-01  9.93434489e-01 -2.46948805e-02  3.10638566e-02]
  [ 2.15863690e-01 -2.61089983e-10 -9.76423204e-01 -3.20068933e-02]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]]

r['head'].shape = (1, 4, 4)
r['head'] = 
[[[ 0.99795854  0.00573968  0.06360561  0.02439034]
  [ 0.01758338  0.93276709 -0.36005035  0.04233643]
  [-0.0613958   0.36043376  0.93076211  1.16997719]
  [ 0.          0.          0.          1.        ]]]

r['left_pinch_distance'].shape = ()
r['left_pinch_distance'] = 
0.01833002297214995

r['right_pinch_distance'].shape = ()
r['right_pinch_distance'] = 
0.03299899190707854

r['right_wrist_roll'].shape = ()
r['right_wrist_roll'] = 
-0.8412487930570391

r['left_wrist_roll'].shape = ()
r['left_wrist_roll'] = 
2.374535526430786
'''