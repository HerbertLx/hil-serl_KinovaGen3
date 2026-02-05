import os
import pickle as pkl
import numpy as np
import cv2  # 需要安装 opencv-python

# 1. 定义路径

pkl_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/examples/experiments/apple_put/classifier_data/apple_put_failure_images_2026-01-14_13-46-20.pkl"
save_root = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/examples/experiments/apple_put/classifier_data/failure_imgs"

# 2. 创建文件夹
side_dir = os.path.join(save_root, "side_1")
wrist_dir = os.path.join(save_root, "wrist_1")
os.makedirs(side_dir, exist_ok=True)
os.makedirs(wrist_dir, exist_ok=True)

# 3. 加载数据
with open(pkl_path, "rb") as f:
    transitions = pkl.load(f)

print(f"Loaded {len(transitions)} transitions. Starting extraction...")

# 4. 遍历并保存图像
for i, trans in enumerate(transitions):
    obs = trans['observations']
    
    # 获取 side_1 图像并去除 Batch 维度 (1, 128, 128, 3) -> (128, 128, 3)
    img_side = np.squeeze(obs['side_1'])
    # 获取 wrist_1 图像
    img_wrist = np.squeeze(obs['wrist_1'])

    # 注意：如果图像是 RGB 格式，OpenCV 保存需要转为 BGR
    img_side_bgr = cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    # 保存文件，命名格式：index_视角.jpg
    cv2.imwrite(os.path.join(side_dir, f"img_{i:04d}_side.jpg"), img_side_bgr)
    cv2.imwrite(os.path.join(wrist_dir, f"img_{i:04d}_wrist.jpg"), img_wrist_bgr)
    if i % 5 == 0:
        print(f"Processed {i}/{len(transitions)}...")

print(f"✅ Extraction complete! Images saved to: {save_root}")