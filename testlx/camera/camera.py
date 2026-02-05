import cv2
import os
import time

def run_pip_recorder():
    save_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/camera/output"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap0 = cv2.VideoCapture(0)
    cap3 = cv2.VideoCapture(3)

    recording = False
    out = None

    print("--- 控制说明 ---")
    print("按下 's' 键：开始/停止录制")
    print("按下 'q' 键：退出程序")

    while True:
        ret0, frame0 = cap0.read()
        ret3, frame3 = cap3.read()

        if not ret0 or not ret3:
            print("警告：无法从摄像头获取画面")
            break

        # 1. 调整 video0 的亮度和对比度
        # alpha: 对比度 (1.2), beta: 亮度 (30)
        proc_frame0 = cv2.convertScaleAbs(frame0, alpha=1.2, beta=30)

        # 2. 缩小 video3 并叠加到右下角
        small_frame3 = cv2.resize(frame3, (240, 180))
        h0, w0, _ = proc_frame0.shape
        h3, w3, _ = small_frame3.shape
        
        x_offset, y_offset = w0 - w3 - 10, h0 - h3 - 10
        proc_frame0[y_offset:y_offset+h3, x_offset:x_offset+w3] = small_frame3

        # 3. 录制逻辑
        if recording:
            if out is not None:
                out.write(proc_frame0)
            # 在预览画面上打个红点，表示正在录制
            cv2.circle(proc_frame0, (30, 30), 10, (0, 0, 255), -1)

        # 4. 显示画面
        cv2.imshow('PIP Recorder', proc_frame0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if not recording:
                # 开始录制：初始化 VideoWriter
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_name = os.path.join(save_path, f"output_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用 mp4 编码
                fps = 20.0  # 建议根据实际运行帧率调整
                out = cv2.VideoWriter(file_name, fourcc, fps, (w0, h0))
                recording = True
                print(f"开始录制: {file_name}")
            else:
                # 停止录制
                recording = False
                if out:
                    out.release()
                print("录制已停止并保存。")

    cap0.release()
    cap3.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pip_recorder()