# /home/cuhk/Documents/visionpro-kinova-rl/hil-serl/testlx/Game2048/game2048_env.py

# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

#%%
class Game2048Env(gym.Env):
    """
    2048 游戏环境类，遵循 Gymnasium 接口规范。
    
    状态空间 (Observation): 4x4 的 numpy 数组，取值范围为 0 (空格) 到 2^16。
    动作空间 (Action): Discrete(4)，即 0:上, 1:下, 2:左, 3:右。
    """
    def __init__(self):
        super(Game2048Env, self).__init__()
        # 定义动作空间
        self.action_space = spaces.Discrete(4)
        # 定义观测空间，low=0, high=65536 (2^16)，shape=(4, 4)
        self.observation_space = spaces.Box(low=0, high=65536, shape=(4, 4), dtype=np.int32)
        
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        输入: 
            seed: 随机种子 (可选)
            options: 额外选项 (可选)
        输出: 
            tuple: (initial_observation, info_dict)
            - initial_observation: 4x4 的 numpy 数组
            - info_dict: 包含额外信息的字典，如初始分数
        """
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self._add_new_tile() # 2048 初始通常生成两个方块
        self._add_new_tile()
        return self.board.copy(), {}

    def _add_new_tile(self):
        """
        在随机的空位置生成一个新的方块（90%概率为2，10%为4）。
        内部逻辑: 寻找 board 中为 0 的坐标，随机选一个填充。
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r, c] = 2 if random.random() < 0.9 else 4

    def step(self, action):
        """
        执行一个动作，更新游戏状态。
        输入: 
            action: 整数 (0:上, 1:下, 2:左, 3:右)
        输出: 
            tuple: (observation, reward, terminated, truncated, info)
            - observation: 动作执行后的 4x4 棋盘
            - reward: 该步合并产生的总分数
            - terminated: bool，表示游戏是否由于无法移动而结束
            - truncated: bool，表示是否由于步数超限而截断 (此处固定为 False)
            - info: dict，包含当前总分 "score"
        """
        old_board = self.board.copy()
        reward = 0
        
        # 旋转策略：统一将各个方向旋转为“向左滑动”进行处理，处理完再转回去
        # np.rot90(m, k) 表示逆时针旋转 k*90 度
        rot_map = {0: 1, 1: -1, 2: 0, 3: 2} # 上, 下, 左, 右 对应旋转次数
        self.board = np.rot90(self.board, rot_map[action])
        
        # 逐行处理滑动和合并逻辑
        for i in range(4):
            new_row, row_reward = self._slide_and_merge(self.board[i])
            self.board[i] = new_row
            reward += row_reward
        
        # 将棋盘旋转回原始方向
        self.board = np.rot90(self.board, -rot_map[action])
        
        # 检查棋盘是否有物理变化（即动作是否有效）
        moved = not np.array_equal(old_board, self.board)
        if moved:
            self._add_new_tile()
        else:
            # 如果动作没产生位移，给一个小的惩罚项，帮助 RL 算法学习有效移动
            reward -= 1
        
        terminated = self._is_game_over()
        self.score += reward
        
        return self.board.copy(), float(reward), terminated, False, {"score": self.score}

    def _slide_and_merge(self, row):
        """
        处理单行（4个元素）的滑动和合并。
        输入: row (长度为4的一维 numpy 数组)
        输出: (new_row, reward)
        逻辑: 
            1. 移除所有零
            2. 检查相邻且相等的数字进行合并
            3. 计算合并得分，并将结果填充回长度为4的数组
        """
        non_zero = row[row != 0]
        new_row = np.zeros(4, dtype=np.int32)
        reward = 0
        
        skip = False
        idx = 0
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            # 发现相邻且相等
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
                new_row[idx] = non_zero[i] * 2
                reward += new_row[idx]
                skip = True # 下一个数字已被合并，跳过
            else:
                new_row[idx] = non_zero[i]
            idx += 1
        return new_row, reward

    def _is_game_over(self):
        """
        判断游戏是否结束。
        逻辑: 
            1. 棋盘还有空位，未结束。
            2. 棋盘没空位，但水平或垂直方向还有能合并的相邻数字，未结束。
            3. 否则，游戏结束。
        """
        if np.any(self.board == 0):
            return False
        # 检查水平和垂直方向是否还能移动
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j+1] or \
                   self.board[j, i] == self.board[j+1, i]:
                    return False
        return True

    def render(self):
        """
        在终端显示当前棋盘状态。
        """
        print("-" * 21)
        for row in self.board:
            # 使用 4 位宽度格式化输出，零值显示为空白
            print("|" + "|".join(f"{val:4}" if val != 0 else "    " for val in row) + "|")
        print("-" * 21)
        print(f"Current Score: {self.score}")

    def close(self):
        pass

#%%
def test_env():
    """
    环境测试函数：使用随机策略运行一次游戏直到结束。
    """
    print(">>> 启动环境测试（随机策略）")
    env = Game2048Env()
    obs, info = env.reset()
    env.render()
    
    done = False
    while not done:
        # 随机选择一个动作 (0, 1, 2, 3)
        action = env.action_space.sample()
        action_name = ["UP", "DOWN", "LEFT", "RIGHT"][action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 如果棋盘发生了变化或有得分才打印，避免刷屏
        if reward != -1: 
            print(f"\n执行动作: {action_name} | 获得奖励: {reward}")
            env.render()
        
        done = terminated
            
    print(f"\n>>> 游戏结束！最终总分: {info['score']}")
    env.close()

if __name__ == "__main__":
    test_env()