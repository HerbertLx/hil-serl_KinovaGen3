# /home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/Game2048/train.py

# %% [1] 导入库与环境配置
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy
import random
from datetime import datetime
from collections import deque

# 导入自定义环境
from game2048_env import Game2048Env

# --- 工具函数：时间戳 ---
def get_timestamp():
    return datetime.now().strftime("%m%d%H%M")

FILE_PREFIX = f"{get_timestamp()}_DQN_2048"
print(f"🚀 当前实验前缀: {FILE_PREFIX}")

# %% [2] 配置参数
CONFIG = {
    "output_path": "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/Game2048/output",
    "model_name": f"{FILE_PREFIX}_best_dqn.pth",
    
    # 2048 任务参数
    "state_dim": 16,        # 4x4 展平
    "action_dim": 4,       # 上下左右
    
    # DQN 超参数
    "lr": 1e-4,
    "gamma": 0.99,
    "buffer_capacity": 50000,
    "batch_size": 64,
    "epsilon_start": 1.0,   # 初始探索率
    "epsilon_end": 0.05,    # 最小探索率
    "epsilon_decay": 20000, # 在多少步内完成探索率衰减
    
    "max_steps": 100000,    # 总训练步数
    "start_steps": 2000,    # 预热期：先随机采样数据
    "update_after": 2000,   # 何时开始更新网络
    "update_every": 4,      # 每隔几步更新一次网络
    "target_update": 1000,  # 目标网络同步频率
}

os.makedirs(CONFIG["output_path"], exist_ok=True)

# %% [3] 定义网络结构 (DQN)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 使用简单的全连接网络，对于4x4棋盘足够
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# %% [4] 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.storage = deque(maxlen=capacity)
    def push(self, data):
        self.storage.append(data)
    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)), 
                torch.LongTensor(np.array(a)), 
                torch.FloatTensor(np.array(r)), 
                torch.FloatTensor(np.array(ns)), 
                torch.FloatTensor(np.array(d)))

# %% [5] 状态预处理
def preprocess(board):
    """将棋盘数值转为 log2 格式并展平"""
    # 0 保持为 0，其余取 log2 (例如 2->1, 4->2, 8->3...)
    board_log = np.where(board > 0, np.log2(board), 0)
    return board_log.flatten().astype(np.float32)

# %% [6] DQN 智能体
class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(CONFIG["state_dim"], CONFIG["action_dim"]).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=CONFIG["lr"])
        self.memory = ReplayBuffer(CONFIG["buffer_capacity"])
        self.steps_done = 0

    def select_action(self, state, epsilon):
        # epsilon-greedy 探索
        if random.random() < epsilon:
            return random.randrange(CONFIG["action_dim"])
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.q_net(state).argmax().item()

    def update(self):
        s, a, r, ns, d = self.memory.sample(CONFIG["batch_size"])
        s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)

        # 当前 Q 值
        curr_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        
        # 目标 Q 值 (DQN 核心公式)
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0]
            target_q = r + (1 - d) * CONFIG["gamma"] * max_next_q

        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# %% [7] 训练逻辑
def train():
    env = Game2048Env()
    agent = DQNAgent()
    
    obs, _ = env.reset()
    state = preprocess(obs)
    best_score = 0
    total_loss = 0
    update_cnt = 0

    print(f"\n[INFO] DQN 训练开始 | 设备: {agent.device}")

    for t in range(CONFIG["max_steps"]):
        # 计算当前的探索率 epsilon (线性衰减)
        epsilon = max(CONFIG["epsilon_end"], 
                      CONFIG["epsilon_start"] - t / CONFIG["epsilon_decay"] * (CONFIG["epsilon_start"] - CONFIG["epsilon_end"]))

        # --- 策略选择 ---
        if t < CONFIG["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, epsilon)

        # --- 环境交互 ---
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = preprocess(next_obs)
        
        # 存入 Buffer
        agent.memory.push((state, action, reward, next_state, done))
        state = next_state

        if done:
            if info["score"] > best_score:
                best_score = info["score"]
                # 保存表现最好的模型
                torch.save(agent.q_net.state_dict(), os.path.join(CONFIG["output_path"], CONFIG["model_name"]))
            obs, _ = env.reset()
            state = preprocess(obs)

        # --- 网络更新 ---
        if t >= CONFIG["update_after"] and t % CONFIG["update_every"] == 0:
            loss = agent.update()
            total_loss += loss
            update_cnt += 1
            
            # 定期同步目标网络
            if t % CONFIG["target_update"] == 0:
                agent.target_net.load_state_dict(agent.q_net.state_dict())

        # --- 打印日志 ---
        if (t + 1) % 5000 == 0:
            avg_loss = total_loss / update_cnt if update_cnt > 0 else 0
            print(f"Step: {t+1:6d} | Eps: {epsilon:.2f} | Best Score: {best_score} | Avg Loss: {avg_loss:.4f}")
            total_loss = 0
            update_cnt = 0

    print(f"✅ 训练完成。最佳得分: {best_score}")
    return agent

# %% [8] 执行训练
trained_agent = train()

# %%
# /home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/Game2048/eval.py

# %% [1] 导入库与环境
import torch
import torch.nn as nn
import numpy as np
import os
import time
from game2048_env import Game2048Env

# --- 配置推理参数 ---
MODEL_PATH = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/Game2048/output/12311247_DQN_2048_best_dqn.pth"
STATE_DIM = 16
ACTION_DIM = 4
ACTION_NAMES = ["UP (上)", "DOWN (下)", "LEFT (左)", "RIGHT (右)"]

# %% [2] 定义网络结构 (必须与训练时一致)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# %% [3] 状态预处理函数
def preprocess(board):
    """将棋盘数值转为 log2 格式并展平，确保与训练输入一致"""
    board_log = np.where(board > 0, np.log2(board), 0)
    return board_log.flatten().astype(np.float32)

# %% [4] 推理函数
def evaluate():
    # 1. 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    # 2. 初始化环境和模型
    env = Game2048Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(STATE_DIM, ACTION_DIM).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # 设置为评估模式
    print(f"✅ 成功加载模型: {MODEL_PATH}")
    print("准备开始测试... (按 Ctrl+C 可随时退出)")
    time.sleep(1)

    # 3. 运行一局游戏
    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        step_count += 1
        state = preprocess(obs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 4. Agent 决策
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        # 5. 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 6. 打印当前步的状态
        # 清屏指令（在 Linux 终端生效），让动画看起来在原地更新
        # os.system('clear') 
        
        print(f"\n--- 第 {step_count} 步 ---")
        print(f"Agent 选择动作: {ACTION_NAMES[action]}")
        print(f"获得奖励: {reward}")
        env.render() # 调用你环境里的渲染函数
        
        obs = next_obs
        done = terminated or truncated

        # 暂停一下，方便观察。如果想快速运行，可以减小数值或注释掉 input()
        # time.sleep(0.2) 
        input("按【回车键】继续下一步...") 

    print("\n" + "="*20)
    print(f"游戏结束！总步数: {step_count}")
    print(f"最终得分: {info['score']}")
    print("="*20)

# %% [5] 执行
if __name__ == "__main__":
    evaluate()
