import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

# ### NEW: 引入依赖 ###
from pynput import keyboard 
from avp_agent import AVPAgent
# ####################

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
# ### NEW: 增加 AVP IP 参数 ###
flags.DEFINE_string("avp_ip", "10.31.181.201", "IP address of Apple Vision Pro.") 
# ###########################

# ### NEW: 全局变量控制离合器 ###
is_clutched = False

def on_press(key):
    global is_clutched
    if key == keyboard.Key.space:
        is_clutched = True

def on_release(key):
    global is_clutched
    if key == keyboard.Key.space:
        is_clutched = False
# #############################

def main(_):
    # ### NEW: 启动键盘监听 ###
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # 初始化 AVP Agent
    agent = AVPAgent(ip=FLAGS.avp_ip)
    print("========================================")
    print("  遥操作模式已就绪")
    print("  按住 [空格键] -> 激活机器人运动")
    print("  松开 [空格键] -> 机器人停止 (可调整手部姿态)")
    print("========================================")
    # #######################

    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 这里的 Env 应该是你的 KinovaEnv
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    while success_count < success_needed:
        # ### NEW: 核心控制逻辑修改 ###
        
        # 1. 从 AVP 获取动作
        avp_action = agent.get_action()
        
        # 2. 离合器逻辑
        if is_clutched:
            # 按下空格：发送 AVP 的真实动作
            actions = avp_action
            # 可选：打印动作数值用于调试
            # print(f"Act: {actions[:3]}") 
        else:
            # 松开空格：发送 0 (保持不动)，并重置 Agent 状态防止跳变
            actions = np.zeros(env.action_space.sample().shape)
            agent.reset() 
            
        # 3. 执行动作 (Active Control)
        # 注意：这里不再是从 info 里面获取 intervene_action 了
        # 因为是你主动控制，你发的 actions 就是 Ground Truth
        next_obs, rew, done, truncated, info = env.step(actions)
        
        # ###########################

        returns += rew
        
        # ### OLD CODE REMOVED ###
        # if "intervene_action" in info:
        #     actions = info["intervene_action"]
        # ########################

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions, # 这里存的就是我们刚才发的 active action
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            # 如果这轮示教结束/重置，记得重置 agent 状态
            agent.reset() 
            
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)