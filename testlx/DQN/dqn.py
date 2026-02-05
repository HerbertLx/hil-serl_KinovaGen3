# %% [1] 导入库与环境配置
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import gymnasium as gym
import os
import copy
from datetime import datetime # 新增：用于处理日期时间
from collections import deque
from gymnasium.wrappers import RecordVideo

# --- 时间戳工具函数 ---
def get_timestamp():
    """返回格式如 12191343 的字符串"""
    return datetime.now().strftime("%m%d%H%M")

# 锁定当前运行批次的时间戳和序号
CURRENT_TIME = get_timestamp()
FILE_VERSION = "1" # 你可以手动修改这个序号，或者将其设为变量
FILE_PREFIX = f"{CURRENT_TIME}_{FILE_VERSION}"
print(f"当前文件前缀: {FILE_PREFIX}")

# %%
# ==========================================
#                  CONFIG
# ==========================================
CONFIG = {
    # 路径设定
    "output_path": r"/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/DQN/output",
    "video_path": r"/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/DQN/videos",
    
    # 文件名：带时间戳前缀
    "model_name": f"{FILE_PREFIX}_best_dqn.pth",
    "video_prefix": f"{FILE_PREFIX}_eval",
    
    # 环境参数
    "env_name": "CartPole-v1",
    "state_dim": 4,
    "action_dim": 2,
    
    # 训练超参数
    "lr": 1e-4,
    "gamma": 0.999,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "buffer_capacity": 10000,
    "batch_size": 64,
    
    # 早停与监控
    "window_size": 10,
    "target_avg_score": 480,
    "patience": 80,
    "max_episodes": 1000,
    "target_update_freq": 10
}

# 自动创建目录
for path in [CONFIG["output_path"], CONFIG["video_path"]]:
    os.makedirs(path, exist_ok=True)

# %% [2] 模型与智能体类定义
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return torch.FloatTensor(state), torch.LongTensor(action), \
               torch.FloatTensor(reward), torch.FloatTensor(next_state), \
               torch.FloatTensor(done)
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=CONFIG["lr"])
        self.memory = ReplayBuffer(CONFIG["buffer_capacity"])
        self.epsilon = CONFIG["epsilon_start"]

    def choose_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, CONFIG["action_dim"] - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def train_step(self):
        if len(self.memory) < CONFIG["batch_size"]:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(CONFIG["batch_size"])
        curr_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * CONFIG["gamma"] * max_next_q
        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(CONFIG["epsilon_min"], self.epsilon * CONFIG["epsilon_decay"])

# %% [3] 训练函数定义
def run_training():
    env = gym.make(CONFIG["env_name"])
    agent = DQNAgent(CONFIG["state_dim"], CONFIG["action_dim"])
    
    reward_history = deque(maxlen=CONFIG["window_size"]) 
    best_avg_reward = 0 
    best_model_weights = None
    no_avg_improvement_count = 0 

    print(f"\n=== 开始训练 | 存档前缀: {FILE_PREFIX} ===")
    for episode in range(CONFIG["max_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.add(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            if done: break
                
        reward_history.append(total_reward)
        avg_reward = sum(reward_history) / len(reward_history)
        
        if episode % CONFIG["target_update_freq"] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_avg_improvement_count = 0
            best_model_weights = copy.deepcopy(agent.policy_net.state_dict())
            # 保存带日期时间的文件名
            save_path = os.path.join(CONFIG["output_path"], CONFIG["model_name"])
            torch.save(best_model_weights, save_path)
        else:
            no_avg_improvement_count += 1

        if episode % 10 == 0:
            print(f"Ep: {episode:3d} | Reward: {total_reward:5.1f} | Avg: {avg_reward:5.1f} | Best: {best_avg_reward:5.1f}")

        if avg_reward >= CONFIG["target_avg_score"]:
            print(f"\n[任务达成] 均分已达 {avg_reward:.1f}")
            break
        if no_avg_improvement_count >= CONFIG["patience"]:
            print(f"\n[早停触发] 连续 {CONFIG['patience']} 轮未创新高")
            break

    env.close()
    return agent

# %% [4] 视频录制函数定义
def record_results(weight_path, num_episodes=5):
    print(f"\n=== 开始录制演示视频 | 前缀: {CONFIG['video_prefix']} ===")
    
    base_env = gym.make(CONFIG["env_name"], render_mode="rgb_array")
    render_env = RecordVideo(
        base_env, 
        video_folder=CONFIG["video_path"], 
        episode_trigger=lambda x: True,
        name_prefix=CONFIG["video_prefix"] # 使用带日期的视频前缀
    )

    agent = DQNAgent(CONFIG["state_dim"], CONFIG["action_dim"])
    if os.path.exists(weight_path):
        agent.policy_net.load_state_dict(torch.load(weight_path, map_location='cpu'))
        agent.policy_net.eval()
        print(f"权重加载成功: {os.path.basename(weight_path)}")
    else:
        print("未找到权重文件！")
        return

    for ep in range(num_episodes):
        state, _ = render_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = render_env.step(action)
            state = next_state
            episode_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1} | Score: {episode_reward}")
    
    render_env.close()
    print(f"✅ 录制完成！存放于: {CONFIG['video_path']}")

# %% [5] 执行入口
# 1. 运行训练
trained_agent = run_training()
# %%
# 2. 录制视频
best_weight_file = os.path.join(CONFIG["output_path"], CONFIG["model_name"])
record_results(best_weight_file, num_episodes=5)