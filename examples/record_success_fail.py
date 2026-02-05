#! /usr/bin/env python3

import os
import sys
import time

# --- 路径设置 ---

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 1. 设置 hil-serl 包路径
target_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl"
if os.path.exists(target_path) and target_path not in sys.path:
    sys.path.insert(0, target_path)
    print(f"✅ 成功添加 hil-serl 路径: {target_path}")

# 2. 设置 kinova_manage.py 所在路径 (根据你的描述)
manager_path = "/home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control"
if os.path.exists(manager_path) and manager_path not in sys.path:
    sys.path.insert(0, manager_path)
    print(f"✅ 成功添加 KinovaManager 路径: {manager_path}")

from kinova_manage import KinovaManager

import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
from inputs import get_gamepad  # 需要安装 inputs 库: pip install inputs
import threading
import collections

# 从实验映射表中导入配置（如环境初始化参数）
from experiments.mappings import CONFIG_MAPPING

# --- 定义命令行参数 ---
FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "实验名称，需匹配 CONFIG_MAPPING 中的键")
flags.DEFINE_integer("successes_needed", 200, "需要收集的成功样本数量阈值")

# --- 全局状态：记录空格键是否被按下 ---
success_key = False
def on_press(key):
    global success_key
    try:
        # 如果按下空格键，将当前步骤标记为成功（True）
        if str(key) == 'Key.space':
            success_key = True
    except AttributeError:
        pass

# --- 新增：独立手柄监听逻辑 ---
def gamepad_listener():
    global success_key
    while True:
        try:
            events = get_gamepad()
            print(f"Gamepad events: {events}")  # Debug 输出手柄事件
            for event in events:
                # 'BTN_NORTH' 对应 Xbox 的 Y 按钮 (在 inputs 库中通常是这个代码)
                # event.state == 1 表示按下
                if event.code == 'BTN_NORTH' and event.state == 1:
                    print("\n🎮 Gamepad Y Pressed: Marking Success!")
                    success_key = True
        except Exception as e:
            # 这里的异常处理防止手柄断开连接导致主程序崩溃
            pass

def main(_):
    global success_key
    
    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # 启动手柄监听线程 (Daemon 模式确保随主进程退出)
    # gp_thread = threading.Thread(target=gamepad_listener, daemon=True)
    # gp_thread.start()
    
    assert FLAGS.exp_name in CONFIG_MAPPING, '找不到该实验对应的配置。'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    successes = []
    failures = []
    
    # 使用 deque 作为先进先出的滑窗，容量为 10
    # 它会自动丢弃第 11 个元素，我们手动捕获这个丢弃动作
    buffer_window = collections.deque(maxlen=20)
    
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)

        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )

        
        # --- 核心改进：滑窗过滤逻辑 ---
        if not success_key:
            # 如果窗口已满，最旧的数据正式确认为“失败数据”
            # print(f"\nRecording failure transition. Buffer size: {len(buffer_window)}")
            if len(buffer_window) == buffer_window.maxlen:
                old_transition = buffer_window.popleft()
                failures.append(old_transition)
            
            # 将当前数据放入滑窗暂存（观察期）
            buffer_window.append(transition)
        else:
            if len(buffer_window) < buffer_window.maxlen:
                print("\n⚠️ Warning: Success key pressed but buffer window is not full.")
                success_key = False
            else:
            # 触发成功！
                print(f"\n🎉 Success detected! Recording {len(buffer_window)} buffered failures as ignored.")
                # 1. 存入当前这一帧作为成功范本
                successes.append(transition)
                # 2. 【关键】直接清空滑窗。滑窗内的 10 帧数据既不给 success 也不给 failure
                buffer_window.clear()
                
                pbar.update(1)
                success_key = False
                obs, _ = env.reset() # 成功后立即重置
                continue # 跳过下面的赋值，开启新回合

        obs = next_obs

        if done or truncated:
            # 回合非正常结束时（如超时），清空滑窗，不记录为失败（可选，也可以记录为失败）
            buffer_window.clear()
            obs, _ = env.reset()

    # --- 数据持久化 ---
    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
        
    # 生成时间戳，确保文件名唯一
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 保存成功数据
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    # 保存失败数据
    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)

'''
import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


success_key = False
def on_press(key):
    global success_key
    try:
        if str(key) == 'Key.space':
            success_key = True
    except AttributeError:
        pass

def main(_):
    global success_key
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
        else:
            failures.append(transition)

        if done or truncated:
            obs, _ = env.reset()

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)
'''