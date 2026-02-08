import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import flax.linen as nn
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.networks.actor_critic_nets import Policy, Critic
from serl_launcher.networks.mlp import MLP
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from flax.core import FrozenDict
import flax
from tqdm import tqdm

# ==============================================================================
# 全局配置与路径设置
# ==============================================================================
ENV_NAME = "Pendulum-v1"  # 测试环境：经典倒立摆
OUTPUT_DIR = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl_KinovaGen3/testlx/serl_launcher_sac_test/output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")  # 模型权重保存路径
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")      # 日志保存路径

# 确保输出目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env():
    """
    创建并返回 Gymnasium 环境实例。
    :return: gym.Env 实例
    """
    return gym.make(ENV_NAME)

class SimpleEncoder(nn.Module):
    """
    简易特征编码器：用于处理多模态观测。
    逻辑：如果是单模态则直接返回，多模态则在最后一个维度拼接。
    """
    @nn.compact
    def __call__(self, observations, train=False, stop_gradient=False):
        import jax.tree_util as jtu
        leaves = jtu.tree_leaves(observations)
        if len(leaves) == 1:
            return leaves[0]
        else:
            # 将不同传感器的数据（如图像特征、关节状态）拼接成一个长向量
            return jnp.concatenate(leaves, axis=-1)

def preprocess_obs(obs):
    """
    预处理观测数据：将 Numpy 数组转换为 JAX 数组。
    :param obs: 原始环境观测 (Numpy)
    :return: JAX 处理后的观测 (DeviceArray)
    """
    obs_jax = jnp.array(obs, dtype=jnp.float32)
    return obs_jax

class ReplayBuffer:
    """
    经验回放池：存储智能体与环境交互的历史轨迹，供 SAC 算法进行离线学习。
    """
    def __init__(self, capacity, obs_dim, act_dim):
        """
        :param capacity: 缓冲区最大容量
        :param obs_dim: 观测维度
        :param act_dim: 动作维度
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # 使用 Numpy 数组预分配内存，提高存储效率
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks = np.ones(capacity, dtype=np.float32)  # 用于标识回合是否结束 (1-done)
        self.size = 0  # 当前存储的数据总量
        self.ptr = 0   # 写入指针（循环队列逻辑）

    def add(self, obs, action, reward, next_obs, done):
        """将一步交互存入缓冲区"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.masks[self.ptr] = 0.0 if done else 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        随机采样一个批次的数据，并转换为 JAX 类型。
        :return: 包含 observations, actions, rewards, next_observations, masks 的字典
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = FrozenDict({
            'observations': jnp.array(self.observations[idxs], dtype=jnp.float32),
            'actions': jnp.array(self.actions[idxs], dtype=jnp.float32),
            'rewards': jnp.array(self.rewards[idxs], dtype=jnp.float32),
            'next_observations': jnp.array(self.next_observations[idxs], dtype=jnp.float32),
            'masks': jnp.array(self.masks[idxs], dtype=jnp.float32)
        })
        return batch

def evaluate_agent(agent, env, num_episodes=10):
    """
    评估智能体性能：不进行探索，直接取策略分布的均值（argmax=True）。
    :return: 评估回合的平均累计奖励
    """
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            obs_dict = preprocess_obs(obs)
            # 使用 argmax=True 进行确定性动作选择
            action = agent.sample_actions(obs_dict, seed=jax.random.PRNGKey(0), argmax=True)
            action_np = np.asarray(action)
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def save_model(agent, path):
    """
    将智能体的网络参数序列化并保存到磁盘。
    """
    state_dict = flax.serialization.to_state_dict(agent.state)
    with open(path, 'wb') as f:
        f.write(flax.serialization.msgpack_serialize(state_dict))
    print(f"模型保存到: {path}")

def load_model(agent, path):
    """
    从磁盘加载模型参数并更新智能体状态。
    """
    with open(path, 'rb') as f:
        state_dict = flax.serialization.msgpack_deserialize(f.read())
    # 使用 replace 方法更新智能体的 state 属性
    agent = agent.replace(state=flax.serialization.from_state_dict(agent.state, state_dict))
    print(f"模型从 {path} 加载")
    return agent

# ==============================================================================
# 主执行逻辑
# ==============================================================================
def main():
    print(f"=== 测试环境: {ENV_NAME} ===")
    
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    print(f"观测维度: {obs_dim}, 动作维度: {act_dim}")

    # 1. 构造 SAC 策略网络 (Actor)
    actor_def = Policy(
        encoder=None, # Pendulum 直接使用状态向量，无需额外 Encoder
        network=MLP(hidden_dims=[256, 256], activate_final=True),
        action_dim=act_dim,
        tanh_squash_distribution=True, # 将动作限制在 [-1, 1] 之间
        std_parameterization="exp",    # 使用指数形式参数化标准差，确保正数
        name="actor",
    )
    
    # 2. 构造 评价网络 (Critic)
    critic_def = Critic(
        encoder=None,
        network=MLP(hidden_dims=[256, 256], activate_final=True),
        name="critic",
    )
    
    # 3. 构造 温度系数调节器 (Alpha / Lagrange Multiplier)
    # 用于平衡“最大化奖励”和“最大化熵（探索能力）”
    temperature_def = GeqLagrangeMultiplier(
        init_value=1.0, constraint_shape=(), constraint_type="geq", name="temperature"
    )

    # 初始化随机数种子与虚拟输入以确定模型参数形状
    rng = jax.random.PRNGKey(42)
    obs = env.reset(seed=42)[0]
    obs_jax = preprocess_obs(obs)
    dummy_action = jnp.zeros((act_dim,), dtype=jnp.float32)

    # 创建 SACAgent 实例：封装了更新逻辑、目标网络更新等
    agent = SACAgent.create(
        rng,
        observations=obs_jax,
        actions=dummy_action,
        actor_def=actor_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        image_keys=[], # 本实验不涉及图像输入
        critic_ensemble_size=2,  # 使用两个 Critic (Double Q-learning) 减少过估计
        critic_subsample_size=None,
        discount=0.99,           # 折扣因子
        soft_target_update_rate=0.005, # 指数移动平均 (EMA) 更新率
        target_entropy=-act_dim, # 目标熵，通常设为动作维度的负值
        backup_entropy=True,     # 在贝尔曼方程更新中包含熵
        reward_bias=0.0,
    )

    print("SACAgent 初始化成功！")

    # --- 训练超参数 ---
    total_steps = 100000    # 总步数
    start_steps = 10000    # 初始随机探索步数（填充 buffer）
    batch_size = 256        # 每次训练采样的批大小
    update_freq = 1         # 训练频率（每交互 1 步更新一次）
    eval_freq = 5000        # 评估频率
    save_freq = 10000       # 模型保存频率
    
    replay_buffer = ReplayBuffer(capacity=100000, obs_dim=obs_dim, act_dim=act_dim)

    print(f"开始训练 SAC 智能体...")
    
    obs, _ = env.reset(seed=42)
    episode_reward = 0
    episode_length = 0

    # ==========================================================================
    # 核心训练循环
    # ==========================================================================
    for step in tqdm(range(total_steps), desc="训练进度"):
        
        # --- 动作选择阶段 ---
        if step < start_steps:
            # 随机探索阶段：从环境动作空间中随机采样
            action = env.action_space.sample()
        else:
            # 训练阶段：使用网络策略采样动作
            obs_dict = preprocess_obs(obs)
            rng, sample_rng = jax.random.split(rng) # 分裂随机种子，保证每次采样不同
            action = agent.sample_actions(obs_dict, seed=sample_rng)
            action = np.asarray(action) # 将 JAX DeviceArray 转回 Numpy 进行环境交互

        # --- 环境交互阶段 ---
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # --- 存储阶段 ---
        replay_buffer.add(obs, action, reward, next_obs, done)

        # --- 状态更新 ---
        obs = next_obs
        episode_reward += reward
        episode_length += 1

        # 若一个回合结束（例如倒立摆倒下或达到时间上限）
        if done:
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        # --- 网络训练阶段 ---
        if step >= start_steps and step % update_freq == 0:
            for _ in range(update_freq):
                # 从 Replay Buffer 采样
                batch = replay_buffer.sample(batch_size)
                # 调用 agent.update 进行梯度下降更新，返回更新后的 agent 和 loss 信息
                agent, info = agent.update(batch)

        # --- 评估阶段 ---
        if step % eval_freq == 0 and step > 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=5)
            print(f"测试步数: {step}, 平均奖励: {eval_reward:.2f}")

        # --- 持久化阶段 ---
        if step % save_freq == 0 and step > 0:
            model_path = os.path.join(MODEL_DIR, f"sac_model_{ENV_NAME}_{step}.msgpack")
            save_model(agent, model_path)

    # ==========================================================================
    # 训练后处理
    # ==========================================================================
    print("\n训练完成，进行最终评估...")
    final_reward = evaluate_agent(agent, env, num_episodes=10)
    print(f"最终平均奖励: {final_reward:.2f}")

    final_model_path = os.path.join(MODEL_DIR, f"sac_model_{ENV_NAME}_final.msgpack")
    save_model(agent, final_model_path)

    env.close()
    print(f"\n所有输出文件已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()