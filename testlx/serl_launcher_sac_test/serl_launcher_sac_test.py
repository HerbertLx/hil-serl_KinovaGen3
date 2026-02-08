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

# 设置环境
ENV_NAME = "Pendulum-v1"
OUTPUT_DIR = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl_KinovaGen3/testlx/serl_launcher_sac_test/output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 创建输出目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env():
    return gym.make(ENV_NAME)

class SimpleEncoder(nn.Module):
    @nn.compact
    def __call__(self, observations, train=False, stop_gradient=False):
        import jax.tree_util as jtu
        leaves = jtu.tree_leaves(observations)
        if len(leaves) == 1:
            return leaves[0]
        else:
            return jnp.concatenate(leaves, axis=-1)

def preprocess_obs(obs):
    obs_jax = jnp.array(obs, dtype=jnp.float32)
    return obs_jax

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks = np.ones(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.masks[self.ptr] = 0.0 if done else 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
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
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            obs_dict = preprocess_obs(obs)
            action = agent.sample_actions(obs_dict, seed=jax.random.PRNGKey(0), argmax=True)
            action_np = np.asarray(action)
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def save_model(agent, path):
    state_dict = flax.serialization.to_state_dict(agent.state)
    with open(path, 'wb') as f:
        f.write(flax.serialization.msgpack_serialize(state_dict))
    print(f"模型保存到: {path}")

def load_model(agent, path):
    with open(path, 'rb') as f:
        state_dict = flax.serialization.msgpack_deserialize(f.read())
    agent = agent.replace(state=flax.serialization.from_state_dict(agent.state, state_dict))
    print(f"模型从 {path} 加载")
    return agent

def main():
    print(f"=== 测试环境: {ENV_NAME} ===")
    
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    print(f"观测维度: {obs_dim}, 动作维度: {act_dim}")

    # 构造SACAgent所需的网络
    actor_def = Policy(
        encoder=None,
        network=MLP(hidden_dims=[256, 256], activate_final=True),
        action_dim=act_dim,
        tanh_squash_distribution=True,
        std_parameterization="exp",
        name="actor",
    )
    critic_def = Critic(
        encoder=None,
        network=MLP(hidden_dims=[256, 256], activate_final=True),
        name="critic",
    )
    temperature_def = GeqLagrangeMultiplier(
        init_value=1.0, constraint_shape=(), constraint_type="geq", name="temperature"
    )

    # 初始化观测和动作
    rng = jax.random.PRNGKey(42)
    obs = env.reset(seed=42)[0]
    obs_jax = preprocess_obs(obs)
    dummy_action = jnp.zeros((act_dim,), dtype=jnp.float32)

    agent = SACAgent.create(
        rng,
        observations=obs_jax,
        actions=dummy_action,
        actor_def=actor_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        image_keys=[],
        critic_ensemble_size=2,
        critic_subsample_size=None,
        discount=0.99,
        soft_target_update_rate=0.005,
        target_entropy=-act_dim,
        backup_entropy=True,
        reward_bias=0.0,
    )

    print("SACAgent 初始化成功！")

    # 训练参数
    total_steps = 100000
    start_steps = 10000
    batch_size = 256
    update_freq = 1
    eval_freq = 5000
    save_freq = 10000
    
    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(capacity=100000, obs_dim=obs_dim, act_dim=act_dim)

    print(f"开始训练 SAC 智能体在 {ENV_NAME} 环境...")
    print(f"总训练步数: {total_steps}")
    print(f"开始随机探索步数: {start_steps}")
    print(f"测试频率: {eval_freq} 步")
    print(f"保存频率: {save_freq} 步")

    obs, _ = env.reset(seed=42)
    episode_reward = 0
    episode_length = 0

    # 训练循环
    for step in tqdm(range(total_steps), desc="训练进度"):
        # 随机探索阶段
        if step < start_steps:
            action = env.action_space.sample()
        else:
            obs_dict = preprocess_obs(obs)
            rng, sample_rng = jax.random.split(rng)
            action = agent.sample_actions(obs_dict, seed=sample_rng)
            action = np.asarray(action)

        # 执行动作
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存储经验
        replay_buffer.add(obs, action, reward, next_obs, done)

        # 更新状态
        obs = next_obs
        episode_reward += reward
        episode_length += 1

        # 回合结束，重置环境
        if done:
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        # 训练智能体
        if step >= start_steps and step % update_freq == 0:
            for _ in range(update_freq):
                batch = replay_buffer.sample(batch_size)
                agent, info = agent.update(batch)

        # 评估智能体
        if step % eval_freq == 0 and step > 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=5)
            print(f"测试步数: {step}, 平均奖励: {eval_reward:.2f}")

        # 保存模型
        if step % save_freq == 0 and step > 0:
            model_path = os.path.join(MODEL_DIR, f"sac_model_{ENV_NAME}_{step}.msgpack")
            save_model(agent, model_path)

    # 最终评估
    print("\n训练完成，进行最终评估...")
    final_reward = evaluate_agent(agent, env, num_episodes=10)
    print(f"最终平均奖励: {final_reward:.2f}")

    # 保存最终模型
    final_model_path = os.path.join(MODEL_DIR, f"sac_model_{ENV_NAME}_final.msgpack")
    save_model(agent, final_model_path)

    env.close()
    print(f"\n所有输出文件已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()