import avp_stream
from avp_stream import VisionProStreamer

avp_ip = "192.168.1.163"  # Vision Pro IP (shown in the app)
s = VisionProStreamer(ip=avp_ip)

# Configure video streaming from robot camera
s.configure_video(device="/dev/video2", format="v4l2", size="1280x720", fps=30)
s.start_webrtc()

while True:
    r = s.get_latest()
    # Use tracking data to control your robot
    head_pose = r['head']
    right_wrist = r['right_wrist']
    right_fingers = r['right_fingers']
    print()
    for key in r:
        print(key)
    print(f"\n{r['head']}\n")
    exit()