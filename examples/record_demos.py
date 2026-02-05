import os
import sys
import time
import glob
import pickle as pkl
import numpy as np
import tqdm
from absl import app, flags

# --- 路径与环境配置 ---
# 添加项目核心包路径，确保 serl_launcher 可被导入
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
target_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl"
if os.path.exists(target_path) and target_path not in sys.path:
    sys.path.insert(0, target_path)

# 添加机器人控制 API 路径
manager_path = "/home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control"
if os.path.exists(manager_path) and manager_path not in sys.path:
    sys.path.insert(0, manager_path)

# 设置环境变量：解决 Protobuf 版本冲突，强制使用 Python 纯脚本实现
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 导入依赖库：系统操作、进度条、数值计算、深拷贝、数据序列化、时间处理、命令行参数解析
import os
from tqdm import tqdm  # 进度条显示
import numpy as np     # 数值计算（主要用于动作空间初始化）
import copy            # 深拷贝（避免数据引用冲突）
import pickle as pkl   # 数据序列化（保存演示数据为.pkl文件）
import datetime        # 时间戳（用于生成唯一文件名）
from absl import app, flags  # Google的命令行参数解析工具
import time

# 从自定义模块导入配置映射：不同实验任务的环境配置（对应文章中的11类任务）
from experiments.mappings import CONFIG_MAPPING

# 初始化命令行参数解析器
FLAGS = flags.FLAGS
# 定义必填参数：实验名称（需与CONFIG_MAPPING中的任务名对应，如"ram_insertion"）
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# 定义可选参数：需要采集的成功演示次数（默认20次，对应文章中"20-30条离线演示"的设置）
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    # 校验实验名称是否有效（必须在预定义的配置映射中）
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    # 根据实验名称获取对应配置，创建真实环境（非仿真）
    # fake_env=False：使用真实机器人硬件（而非仿真环境）
    # save_video=False：不保存采集过程视频（减少存储开销）
    # classifier=True：启用奖励分类器相关的状态判断（用于标记任务是否成功）
    print(f"Loading environment for experiment: {FLAGS.exp_name}")
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    print(f"Finish reading config from command")
    print()

    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    # 重置环境，获取初始观测值obs和环境信息info（如初始姿态、传感器数据）
    obs, info = env.reset() # info is an empty dictionary here
    print("Reset done")  # 提示环境重置完成
    
    # 初始化存储变量
    transitions = []  # 存储所有成功演示的轨迹数据（每条轨迹包含多个时间步的transition）
    success_count = 0  # 已采集的成功演示次数
    success_needed = FLAGS.successes_needed  # 目标采集的成功演示次数
    pbar = tqdm(total=success_needed)  # 创建进度条（可视化采集进度）
    trajectory = []  # 存储单条轨迹的所有时间步数据
    returns = 0  # 单条轨迹的累计奖励（用于实时显示采集状态）
    
    # 循环采集，直到达到目标成功演示次数
    while success_count < success_needed:
        # 初始化动作：获取动作空间的维度，生成全0动作（占位用，实际动作由人类遥操作输入）
        actions = np.zeros(env.action_space.sample().shape) 
        # 环境执行动作：获取下一个观测值、即时奖励、任务结束标记、截断标记、环境信息
        # 关键：这里的动作实际由人类通过SpaceMouse遥操作输入（通过info["intervene_action"]传递）
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew  # 累计当前轨迹的奖励
        
        # 如果存在人类遥操作动作（info中包含"intervene_action"），则替换为人类输入的动作
        if "intervene_action" in info:
            actions = info["intervene_action"]
        
        # 保存当前时间步的transition数据（状态转移）
        # 包含：当前观测、执行动作、下一个观测、即时奖励、结束标记、环境信息
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,  # 掩码：1表示未结束，0表示已结束
                dones=done,        # 是否结束当前轨迹
                infos=info,        # 环境附加信息（如是否成功、传感器原始数据）
            )
        )
        trajectory.append(transition)  # 将当前时间步数据加入当前轨迹
        
        # 更新进度条描述：显示当前轨迹的累计奖励（便于实时监控）
        pbar.set_description(f"Return: {returns}")

        # 更新当前观测值（准备下一个时间步）
        obs = next_obs
        
        # 如果当前轨迹结束（done=True，任务完成或失败）
        if done:
            # 检查是否成功完成任务（info["succeed"]由奖励分类器或人工标记）
            print(f"\nDone = {done}, Reward = {rew}, Succeed = {info['succeed']}")  # --- DEBUG ---
            if info["succeed"]:
                # 将成功轨迹的所有时间步数据加入总演示集
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                print(f"Collected {success_count + 1} / {success_needed} successful demos.")
                success_count += 1  # 成功次数+1
                pbar.update(1)      # 进度条更新
            # 重置轨迹存储和累计奖励（准备下一条轨迹采集）
            trajectory = []
            returns = 0
            obs, info = env.reset()  # 重置环境到初始状态
            
    # 检查演示数据保存目录是否存在，不存在则创建
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    # 生成唯一文件名（包含实验名称、成功次数、时间戳）：避免文件覆盖
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    # 保存演示数据（使用pickle序列化，保留Python对象结构）
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")  # 提示保存完成

# 程序入口：解析命令行参数并执行main函数
if __name__ == "__main__":
    app.run(main)

'''
import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
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
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
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
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
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
'''