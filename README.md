# HIL-SERL

## 运行脚本

### 训练脚本 apple_put

```bash
# 1. 记录成功/失败数据
cd /home/cuhk/Documents/visionpro-kinova-rl/hil-serl/examples
python record_success_fail.py --exp_name apple_put --successes_needed 30
```

```bash
# 2. 将 classifier_data 文件夹从 examples 路径移动到 apple_put 路径下
# (手动操作)
```


```bash
# 3. 训练奖励分类器
cd /home/cuhk/Documents/visionpro-kinova-rl/hil-serl/examples/experiments/apple_put
python ../../train_reward_classifier.py --exp_name apple_put
```

```bash
# 4. 记录演示数据
cd /home/cuhk/Documents/visionpro-kinova-rl/hil-serl/examples/experiments/apple_put
python ../../record_demos.py --exp_name apple_put --successes_needed 30
```

```bash
# 5. 运行学习器 (注意: run_learner.sh 中 demo_path 参数需要修改)
cd ~/Documents/visionpro-kinova-rl/hil-serl/examples/experiments/apple_put
bash run_learner.sh
```

```bash
# 6. 运行执行器
cd ~/Documents/visionpro-kinova-rl/hil-serl/examples/experiments/apple_put
bash run_actor.sh
```

### 训练脚本 usb_pickup_insertion

(待补充)

---

## 复原脚本


### 手柄遥操机械臂
```bash
python /home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control/gamepad_control_obs.py
```

### 原始机械臂测试脚本 (运行完记得注释掉测试代码)
```bash
python /home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control/kinova_manage.py
```

### 存储当前 side_camera 照片到指定位置
```bash
ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video4 -frames:v 1 /home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/camera_position/test_image.jpg
```

---

## 依赖版本管理

### 更换 protobuf 版本

```bash
# 适配 kinova_env.py
pip install "protobuf==3.20.3"

# 适配 train_rlpd.py
pip install "protobuf==6.33.2"
```
