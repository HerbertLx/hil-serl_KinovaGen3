# %% [1] 导入库与环境配置
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import os
import copy
import random
from datetime import datetime
from collections import deque
from gymnasium.wrappers import RecordVideo

# --- 工具函数：时间戳 ---
def get_timestamp():
    return datetime.now().strftime("%m%d%H%M")

FILE_PREFIX = f"{get_timestamp()}_SAC_Continuous"
print(f"🚀 当前实验前缀: {FILE_PREFIX}")

# %% [2] 配置参数
CONFIG = {
    "output_path": "./output",
    "video_path": "./videos",
    "model_name": f"{FILE_PREFIX}_best_sac.pth",
    "video_prefix": f"{FILE_PREFIX}_eval",
    
    # 任务改为连续控制环境：Pendulum (摆杆上扬)
    # 状态：3维 (cos, sin, angular_vel) | 动作：1维 (扭矩 [-2, 2])
    "env_name": "Pendulum-v1",
    "state_dim": 3,
    "action_dim": 1,
    "max_action": 2.0, 
    
    # SAC 超参数
    "lr": 3e-4,             # 统一学习率
    "gamma": 0.99,          # 折扣因子
    "tau": 0.005,           # 软更新系数 (Soft Update)
    "alpha": 0.2,           # 初始熵系数
    "buffer_capacity": 100000,
    "batch_size": 256,
    
    "max_steps": 40000,     # SAC 通常按 Step 计数
    "start_steps": 1000,    # 预热期：先随机采样一些数据
    "update_after": 1000,   # 何时开始更新网络
    "update_every": 50,     # 更新频率
}

os.makedirs(CONFIG["output_path"], exist_ok=True)
os.makedirs(CONFIG["video_path"], exist_ok=True)

# %% [3] 定义网络结构 (SAC 三剑客)

# 1. Actor 网络 (输出高斯分布参数并重参数化)
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state, deterministic=False, with_logprob=True):
        x = self.fc(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2) # 限制标准差防止数值爆炸
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample() # 重参数化采样 (Trick!)

        # 映射到 [-max_action, max_action]
        log_prob = None
        if with_logprob:
            # 修正 Tanh 激活带来的对数概率偏差
            log_prob = dist.log_prob(action).sum(dim=-1)
            log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1)
        
        action = torch.tanh(action) * self.max_action
        return action, log_prob

# 2. Critic 网络 (双 Q 网络设计)
class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1 网络
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 网络 (缓解价值高估)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

# %% [4] 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.storage = deque(maxlen=capacity)
    def push(self, data):
        self.storage.append(data)
    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        return map(lambda x: torch.FloatTensor(np.array(x)), zip(*batch))

# %% [5] SAC 智能体
class SACAgent:
    def __init__(self):
        self.actor = SACActor(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["max_action"])
        self.critic = SACCritic(CONFIG["state_dim"], CONFIG["action_dim"])
        self.target_critic = copy.deepcopy(self.critic)
        
        # 自动调整熵系数 alpha 的目标
        self.target_entropy = -CONFIG["action_dim"]
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=CONFIG["lr"])
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=CONFIG["lr"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CONFIG["lr"])
        self.memory = ReplayBuffer(CONFIG["buffer_capacity"])

    def update(self):
        s, a, r, ns, d = self.memory.sample(CONFIG["batch_size"])
        r, d = r.unsqueeze(-1), d.unsqueeze(-1)
        
        curr_alpha = self.log_alpha.exp()

        # 1. 更新 Critic
        with torch.no_grad():
            next_a, next_logp = self.actor(ns)
            q1_t, q2_t = self.target_critic(ns, next_a)
            # SAC 核心：目标值 = r + gamma * (min_Q - alpha * log_prob)
            target_q = r + (1 - d) * CONFIG["gamma"] * (torch.min(q1_t, q2_t) - curr_alpha * next_logp.unsqueeze(-1))
        
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. 更新 Actor
        new_a, logp = self.actor(s)
        q1_new, q2_new = self.critic(s, new_a)
        # Actor 目标：最大化 (min_Q - alpha * log_prob)
        actor_loss = (curr_alpha * logp.unsqueeze(-1) - torch.min(q1_new, q2_new)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3. 更新 Alpha (自动熵调整)
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. 软更新 Target Critic
        for p, p_t in zip(self.critic.parameters(), self.target_critic.parameters()):
            p_t.data.copy_(CONFIG["tau"] * p.data + (1 - CONFIG["tau"]) * p_t.data)

# %% [6] 训练逻辑
def train():
    # 1. 初始化环境与智能体
    env = gym.make(CONFIG["env_name"])
    agent = SACAgent()
    
    # 获取初始状态并初始化最高奖励记录
    state, _ = env.reset()
    best_reward = -np.inf

    print(f"\n[INFO] SAC 训练开始 | 环境: {CONFIG['env_name']}")

    # 2. 步数驱动的主循环 (SAC 通常按 Step 而非 Episode 计数，因为它是 Off-policy 算法)
    for t in range(CONFIG["max_steps"]):
        
        # --- 策略选择阶段 ---
        # 预热期 (Warm-up)：在最初的 start_steps 内，完全随机采样动作
        # 目的：填充 Replay Buffer，让初始训练时数据分布更均匀，避免网络过早过拟合旧数据
        if t < CONFIG["start_steps"]:
            action = env.action_space.sample() 
        else:
            # 探索期：使用 Actor 网络根据当前状态 $s$ 采样动作 $a$
            with torch.no_grad():
                # 采样包含随机性 (重参数化采样)，有助于持续探索
                action, _ = agent.actor(torch.FloatTensor(state).unsqueeze(0))
                action = action.numpy()[0] # 转为 numpy 以喂给 gymnasium 环境

        # --- 环境交互阶段 ---
        # 执行动作，获取：次态、奖励、是否终止、是否截断
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # 存储经验 (Experience Replay)：将五元组存入 Buffer
        # 这是异策略 (Off-policy) 的核心：存下来以后反复学
        agent.memory.push((state, action, reward, next_state, done))
        
        # 状态流转
        state = next_state
        # 如果回合结束，立即重置环境，开始新的一轮采样
        if done: 
            state, _ = env.reset()

        # --- 训练更新阶段 ---
        # 只有在过了 update_after 步之后，才开始更新网络参数
        # 且每隔 update_every 步，集中进行一次“高强度更新”
        if t >= CONFIG["update_after"] and t % CONFIG["update_every"] == 0:
            # 每次更新 update_every 次，保证采样比 (Sample Efficiency)
            for _ in range(CONFIG["update_every"]):
                # 每次 update 都会从 Buffer 中随机采样一个 Batch 进行梯度下降
                agent.update()

        # --- 日志记录阶段 ---
        # 每 2000 步打印一次当前的训练进度和 Alpha 值
        # Alpha 反映了当前智能体的探索欲望 (熵的权重)
        if (t + 1) % 2000 == 0:
            print(f"Step: {t+1:5d}/{CONFIG['max_steps']} | Alpha: {agent.log_alpha.exp().item():.4f}")

    # 3. 训练结束，持久化保存 Actor 网络权重
    torch.save(agent.actor.state_dict(), os.path.join(CONFIG["output_path"], CONFIG["model_name"]))
    print("✅ 训练完成")
    return agent

# %% [7] 录制演示
def record_eval():
    print(f"\n[INFO] 录制 SAC 演示视频...")
    base_env = gym.make(CONFIG["env_name"], render_mode="rgb_array")
    eval_env = RecordVideo(base_env, video_folder=CONFIG["video_path"], name_prefix=CONFIG["video_prefix"])
    
    agent = SACActor(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["max_action"])
    agent.load_state_dict(torch.load(os.path.join(CONFIG["output_path"], CONFIG["model_name"])))
    agent.eval()

    state, _ = eval_env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action, _ = agent(torch.FloatTensor(state).unsqueeze(0), deterministic=True)
            state, _, term, trunc, _ = eval_env.step(action.numpy()[0])
            done = term or trunc
    eval_env.close()
    print(f"🎥 视频已存至: {CONFIG['video_path']}")

# %% [8] 主程序入口
# 执行
trained_agent = train()
# %% [9] 录制视频
record_eval()
# %%
import gymnasium as gym

def print_env_info():
    env = gym.make("Pendulum-v1")
    
    print("--- 任务具体信息 ---")
    print(f"任务名称: Pendulum-v1")
    
    # 查看动作空间 (Action Space)
    # SAC 必须知道动作的上限和下限，以便进行 Tanh 映射
    print(f"动作空间类型: {env.action_space}") 
    print(f"动作最大值: {env.action_space.high}")
    print(f"动作最小值: {env.action_space.low}")
    
    # 查看观测空间 (Observation Space)
    print(f"观测空间类型: {env.observation_space}")
    print(f"观测值上限: {env.observation_space.high}")
    print(f"观测值下限: {env.observation_space.low}")
    
    env.close()

print_env_info()

# %%
import gymnasium as gym
import torch

env = gym.make("Pendulum-v1")
state, _ = env.reset()

print(f"初始状态 (cos, sin, 角速度): {state}")

# 案例 1：施加最大的逆时针力矩 (2.0)
action = [2.0]
next_state, reward, term, trunc, info = env.step(action)
print(f"\n施加力矩 {action} 后：")
print(f"新的角速度: {next_state[2]:.4f} (应该变大了)")

# 案例 2：施加最大的顺时针力矩 (-2.0)
action = [-2.0]
next_state, reward, term, trunc, info = env.step(action)
print(f"\n施加力矩 {action} 后：")
print(f"新的角速度: {next_state[2]:.4f} (由于反向用力，角速度变小或变负)")

env.close()