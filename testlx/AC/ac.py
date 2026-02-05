# %% [1] 导入库与环境配置
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
import gymnasium as gym
import os
import copy
from datetime import datetime
from collections import deque
from gymnasium.wrappers import RecordVideo

# --- 工具函数：时间戳 ---
def get_timestamp():
    """返回格式如 12201430 的字符串 (月日时分)"""
    return datetime.now().strftime("%m%d%H%M")

# 锁定当前运行批次的前缀
FILE_PREFIX = f"{get_timestamp()}_AC_V1"
print(f"🚀 当前实验前缀: {FILE_PREFIX}")

# %% [2] 配置参数
CONFIG = {
    "output_path": "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/AC/output",
    "video_path": "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/AC/videos",
    "model_name": f"{FILE_PREFIX}_best_ac.pth",
    "video_prefix": f"{FILE_PREFIX}_eval",
    
    "env_name": "CartPole-v1",
    "state_dim": 4,
    "action_dim": 2,
    
    # AC 超参数
    "lr_actor": 1e-3,       # Actor 通常学习率稍高
    "lr_critic": 3e-3,      # Critic 负责基准，通常需要收敛得更快
    "gamma": 0.99,          # 折扣因子
    "max_episodes": 1000,
    "target_avg_score": 480,
    "window_size": 30,      # 计算滑动平均的窗口
    "patience": 100         # 早停耐受度
}

# 确保目录存在
os.makedirs(CONFIG["output_path"], exist_ok=True)
os.makedirs(CONFIG["video_path"], exist_ok=True)

# %% [3] 定义网络结构

# Actor 网络：输入状态，输出动作的概率分布
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1) # 关键：确保输出为概率分布（和为1）
        )
    
    def forward(self, x):
        return self.fc(x)

# Critic 网络：输入状态，输出一个标量 V(s) 价值预测
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 输出当前处境的“得分”
        )
    
    def forward(self, x):
        return self.fc(x)

# %% [4] 定义 AC 智能体
class ACAgent:
    def __init__(self):
        self.actor = Actor(CONFIG["state_dim"], CONFIG["action_dim"])
        self.critic = Critic(CONFIG["state_dim"])
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=CONFIG["lr_actor"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CONFIG["lr_critic"])
        
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state) # 获取概率分布
        m = Categorical(probs)    # 构建分类分布
        action = m.sample()       # 采样动作
        return action.item(), m.log_prob(action) # 返回动作和其对数概率

    def update(self, state, log_prob, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        # 1. 计算 Critic 目标与误差 (TD Error)
        # 这里的 TD Error 就是优势函数 A(s,a) = r + gamma*V(s') - V(s)
        current_v = self.critic(state)
        with torch.no_grad():
            next_v = self.critic(next_state)
            target_v = reward + (1 - done) * CONFIG["gamma"] * next_v
        
        td_error = target_v - current_v # 这就是我们常说的 Advantage
        
        # 2. 更新 Critic (均方误差损失)
        critic_loss = td_error.pow(2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # 3. 更新 Actor (策略梯度损失)
        # Loss = -log_prob * Advantage
        # detach 的目的是防止梯度流向 Critic 网络
        actor_loss = -(log_prob * td_error.detach())
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        return actor_loss.item(), critic_loss.item()

# %% [5] 训练主循环
def train():
    env = gym.make(CONFIG["env_name"])
    agent = ACAgent()
    
    score_history = deque(maxlen=CONFIG["window_size"])
    best_avg_score = 0
    patience_counter = 0
    
    print(f"\n[INFO] 开始训练 {CONFIG['env_name']}...")
    
    for ep in range(CONFIG["max_episodes"]):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            # 智能体选择动作
            action, log_prob = agent.choose_action(state)
            
            # 环境交互
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # AC 是单步更新 (也可批量更新，这里演示标准的单步逻辑)
            agent.update(state, log_prob, reward, next_state, done)
            
            state = next_state
            ep_reward += reward
            
        score_history.append(ep_reward)
        avg_score = np.mean(score_history)
        
        # 记录最佳模型
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            patience_counter = 0
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, os.path.join(CONFIG["output_path"], CONFIG["model_name"]))
        else:
            patience_counter += 1
            
        # 打印进度
        if ep % 20 == 0:
            print(f"Ep: {ep:4d} | Reward: {ep_reward:5.1f} | Avg: {avg_score:5.1f} | Best: {best_avg_score:5.1f} | Counter: {patience_counter}")
            
        # 终止条件
        if avg_score >= CONFIG["target_avg_score"]:
            print(f"\n🎉 任务达成！最终均分: {avg_score:.1f}")
            break
        if patience_counter >= CONFIG["patience"]:
            print(f"\n⚠️ 触发早停，停止训练。")
            break
            
    env.close()
    return agent

# %% [6] 结果验证与视频录制
def record_eval(num_episodes=5):
    print(f"\n[INFO] 正在加载最佳模型并录制视频...")
    
    base_env = gym.make(CONFIG["env_name"], render_mode="rgb_array")
    eval_env = RecordVideo(
        base_env, 
        video_folder=CONFIG["video_path"],
        episode_trigger=lambda x: True,
        name_prefix=CONFIG["video_prefix"]
    )
    
    agent = ACAgent()
    checkpoint_path = os.path.join(CONFIG["output_path"], CONFIG["model_name"])
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.actor.eval()
        print(f"✅ 模型权重加载成功: {CONFIG['model_name']}")
    else:
        print("❌ 未找到权重文件！")
        return

    for i in range(num_episodes):
        state, _ = eval_env.reset()
        total_r = 0
        done = False
        while not done:
            with torch.no_grad():
                probs = agent.actor(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(probs).item() # 测试时选择概率最大的动作
            state, reward, term, trunc, _ = eval_env.step(action)
            total_r += reward
            done = term or trunc
        print(f"Eval Ep {i+1} Score: {total_r}")
        
    eval_env.close()
    print(f"\n[FINISH] 视频已存放在: {CONFIG['video_path']}")

# %% [7] 执行
# 1. 执行训练
train()

# %% [8] 测试
# 2. 录制视频
record_eval(num_episodes=5)
# %%