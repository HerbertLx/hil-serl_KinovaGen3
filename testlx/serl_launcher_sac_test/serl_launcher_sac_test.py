import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# 添加项目根目录到 Python 路径
sys.path.append('/home/cuhk/Documents/visionpro-kinova-rl/hil-serl_KinovaGen3')

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.networks.actor_critic_nets import Critic, Policy
from serl_launcher.networks.mlp import MLP
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.common.optimizers import make_optimizer

# 输出目录
OUTPUT_DIR = '/home/cuhk/Documents/visionpro-kinova-rl/hil-serl_KinovaGen3/testlx/serl_launcher_sac_test/output'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
VIDEO_DIR = os.path.join(OUTPUT_DIR, 'videos')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

# 创建输出目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

class SACTester:
    def __init__(self, env_name='LunarLanderContinuous-v2', seed=42):
        """
        初始化 SAC 测试器
        
        Args:
            env_name: 环境名称
            seed: 随机种子
        """
        self.env_name = env_name
        self.seed = seed
        
        # 创建训练环境
        self.train_env = gym.make(env_name)
        self.train_env.action_space.seed(seed)
        
        # 创建测试环境
        self.test_env = gym.make(env_name, render_mode='rgb_array')
        self.test_env.action_space.seed(seed)
        
        # 环境参数
        self.observation_space = self.train_env.observation_space
        self.action_space = self.train_env.action_space
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        
        # 训练参数
        self.batch_size = 256
        self.replay_buffer_size = 1000000
        self.start_steps = 10000
        self.total_steps = 100000
        self.test_freq = 5000
        self.test_episodes = 10
        self.save_freq = 10000
        
        # 创建经验回放缓冲区
        self.replay_buffer = {}
        self.replay_buffer['observations'] = np.empty((self.replay_buffer_size, self.obs_dim), dtype=np.float32)
        self.replay_buffer['actions'] = np.empty((self.replay_buffer_size, self.action_dim), dtype=np.float32)
        self.replay_buffer['rewards'] = np.empty((self.replay_buffer_size,), dtype=np.float32)
        self.replay_buffer['next_observations'] = np.empty((self.replay_buffer_size, self.obs_dim), dtype=np.float32)
        self.replay_buffer['masks'] = np.empty((self.replay_buffer_size,), dtype=np.float32)
        self.buffer_ptr = 0
        self.buffer_size = 0
        
        # 初始化智能体
        self.agent = self._init_agent()
        
        # 训练记录
        self.episode_rewards = []
        self.episode_lengths = []
        self.test_rewards = []
        
    def _init_agent(self):
        """
        初始化 SAC 智能体
        """
        # 随机种子
        rng = jax.random.PRNGKey(self.seed)
        
        # 创建观测和动作示例
        obs = self.train_env.reset(seed=self.seed)[0]
        action = self.train_env.action_space.sample()
        
        # 转换为 JAX 数组
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        action_jax = jnp.array(action, dtype=jnp.float32)
        
        # 构造符合 SAC 智能体期望的观测数据结构
        # 注意：SAC 智能体期望 observations 是字典
        observations_dict = {'observation': obs_jax}
        
        # 创建一个简单的编码器，从字典中提取观测值
        # 注意：JAX 不支持字符串索引，需要使用其他方式处理字典
        # 我们使用 JAX 的树结构来处理字典
        class SimpleEncoder(nn.Module):
            @nn.compact
            def __call__(self, observations, train=False, stop_gradient=False):
                # 使用 JAX 的树结构来处理字典
                # observations 是一个字典或 FrozenDict，我们需要提取 'observation' 键的值
                # 由于 JAX 不支持字符串索引，我们需要使用 jax.tree_util 来处理
                
                # 使用 jax.tree_util.tree_leaves 来提取所有叶子节点
                # 对于简单的字典结构，这会返回所有的值
                import jax.tree_util as jtu
                
                # 获取所有叶子节点（字典的值）
                leaves = jtu.tree_leaves(observations)
                
                if len(leaves) == 1:
                    return leaves[0]
                else:
                    # 如果有多个值，将它们拼接起来
                    return jnp.concatenate(leaves, axis=-1)
        
        # 创建网络定义
        critic_backbone = MLP(hidden_dims=[256, 256], activate_final=True)
        actor_backbone = MLP(hidden_dims=[256, 256], activate_final=True)
        
        # 创建策略网络
        policy_def = Policy(
            encoder=SimpleEncoder(),  # 使用简单的编码器
            network=actor_backbone,
            action_dim=self.action_dim,
            tanh_squash_distribution=True,
            std_parameterization='exp',
            name='actor'
        )
        
        # 创建评论家网络
        critic_def = Critic(
            encoder=SimpleEncoder(),  # 使用简单的编码器
            network=critic_backbone,
            name='critic'
        )
        
        # 创建温度网络
        temperature_def = GeqLagrangeMultiplier(
            init_value=1.0,
            constraint_shape=(),
            constraint_type='geq',
            name='temperature'
        )
        
        # 创建智能体
        agent = SACAgent.create(
            rng=rng,
            observations=observations_dict,  # 使用字典结构
            actions=action_jax,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            actor_optimizer_kwargs={'learning_rate': 3e-4},
            critic_optimizer_kwargs={'learning_rate': 3e-4},
            temperature_optimizer_kwargs={'learning_rate': 3e-4},
            discount=0.99,
            soft_target_update_rate=0.005,
            target_entropy=-self.action_dim / 2,
            backup_entropy=True,
            critic_ensemble_size=2,
            critic_subsample_size=None,
            image_keys=["observation"],  # 使用有效的图像键
            augmentation_function=None,
            reward_bias=0.0
        )
        
        return agent
    
    def _add_to_buffer(self, obs, action, reward, next_obs, done):
        """
        添加经验到回放缓冲区
        """
        # 计算掩码（1 - done）
        mask = 1.0 - done
        
        # 添加到缓冲区
        self.replay_buffer['observations'][self.buffer_ptr] = obs
        self.replay_buffer['actions'][self.buffer_ptr] = action
        self.replay_buffer['rewards'][self.buffer_ptr] = reward
        self.replay_buffer['next_observations'][self.buffer_ptr] = next_obs
        self.replay_buffer['masks'][self.buffer_ptr] = mask
        
        # 更新缓冲区指针和大小
        self.buffer_ptr = (self.buffer_ptr + 1) % self.replay_buffer_size
        self.buffer_size = min(self.buffer_size + 1, self.replay_buffer_size)
    
    def _sample_batch(self):
        """
        从回放缓冲区采样批次
        """
        # 采样索引
        idxs = np.random.randint(0, self.buffer_size, size=self.batch_size)
        
        # 采样数据
        observations = jnp.array(self.replay_buffer['observations'][idxs], dtype=jnp.float32)
        actions = jnp.array(self.replay_buffer['actions'][idxs], dtype=jnp.float32)
        rewards = jnp.array(self.replay_buffer['rewards'][idxs], dtype=jnp.float32)
        next_observations = jnp.array(self.replay_buffer['next_observations'][idxs], dtype=jnp.float32)
        masks = jnp.array(self.replay_buffer['masks'][idxs], dtype=jnp.float32)
        
        # 构造符合 SAC 智能体期望的批次结构
        # 注意：SAC 智能体期望 observations 和 next_observations 是字典
        # 并且 batch 对象需要支持 copy(add_or_replace=...) 方法
        from flax.core import FrozenDict
        
        batch = FrozenDict({
            'observations': FrozenDict({'observation': observations}),
            'actions': actions,
            'rewards': rewards,
            'next_observations': FrozenDict({'observation': next_observations}),
            'masks': masks
        })
        
        return batch
    
    def train(self):
        """
        训练智能体
        """
        print(f"开始训练 SAC 智能体在 {self.env_name} 环境...")
        print(f"总训练步数: {self.total_steps}")
        print(f"开始随机探索步数: {self.start_steps}")
        print(f"测试频率: {self.test_freq} 步")
        print(f"保存频率: {self.save_freq} 步")
        
        # 初始化环境
        obs, _ = self.train_env.reset(seed=self.seed)
        episode_reward = 0
        episode_length = 0
        start_time = time.time()
        
        # 训练循环
        for step in range(self.total_steps):
            # 开始阶段使用随机动作
            if step < self.start_steps:
                action = self.train_env.action_space.sample()
            else:
                # 从策略中采样动作
                action = self.agent.sample_actions(
                    jnp.array(obs, dtype=jnp.float32),
                    seed=jax.random.PRNGKey(step),
                    argmax=False
                )
                action = np.array(action)
            
            # 执行动作
            next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated
            
            # 添加到回放缓冲区
            self._add_to_buffer(obs, action, reward, next_obs, done)
            
            # 更新当前状态
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # 当回合结束时
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # 打印回合信息
                if len(self.episode_rewards) % 10 == 0:
                    print(f"回合: {len(self.episode_rewards)}, 奖励: {episode_reward:.2f}, 长度: {episode_length}, 步数: {step}")
                
                # 重置环境
                obs, _ = self.train_env.reset()
                episode_reward = 0
                episode_length = 0
            
            # 从回放缓冲区采样并更新智能体
            if step >= self.start_steps:
                batch = self._sample_batch()
                self.agent, info = self.agent.update(batch)
            
            # 定期测试智能体
            if (step + 1) % self.test_freq == 0:
                test_reward = self.test()
                self.test_rewards.append(test_reward)
                print(f"测试步数: {step + 1}, 平均奖励: {test_reward:.2f}")
            
            # 定期保存模型
            if (step + 1) % self.save_freq == 0:
                self.save_model(step + 1)
        
        # 训练结束后测试
        final_test_reward = self.test()
        self.test_rewards.append(final_test_reward)
        print(f"最终测试平均奖励: {final_test_reward:.2f}")
        print(f"训练时间: {time.time() - start_time:.2f} 秒")
        
        # 保存最终模型
        self.save_model(self.total_steps)
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def test(self, record_video=False):
        """
        测试智能体
        
        Args:
            record_video: 是否记录视频
            
        Returns:
            float: 平均测试奖励
        """
        total_reward = 0
        
        # 如果需要记录视频
        if record_video:
            video_env = RecordVideo(
                self.test_env,
                video_folder=VIDEO_DIR,
                name_prefix=f"sac_{self.env_name}",
                episode_trigger=lambda x: True
            )
            env = video_env
        else:
            env = self.test_env
        
        for episode in range(self.test_episodes):
            obs, _ = env.reset(seed=self.seed + episode)
            episode_reward = 0
            done = False
            
            while not done:
                # 构造符合 SAC 智能体期望的观测数据结构
                obs_dict = {'observation': jnp.array(obs, dtype=jnp.float32)}
                
                # 使用确定性策略
                action = self.agent.sample_actions(
                    obs_dict,
                    seed=jax.random.PRNGKey(episode),
                    argmax=True
                )
                action = np.array(action)
                
                # 执行动作
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 更新状态和奖励
                obs = next_obs
                episode_reward += reward
            
            total_reward += episode_reward
        
        # 关闭视频环境
        if record_video:
            video_env.close()
        
        # 计算平均奖励
        average_reward = total_reward / self.test_episodes
        return average_reward
    
    def save_model(self, step):
        """
        保存模型
        
        Args:
            step: 训练步数
        """
        import flax.serialization
        model_path = os.path.join(MODEL_DIR, f"sac_model_{self.env_name}_{step}.msgpack")
        
        # 保存模型参数
        state_dict = flax.serialization.to_state_dict(self.agent)
        with open(model_path, 'wb') as f:
            f.write(flax.serialization.msgpack_serialize(state_dict))
        
        print(f"模型保存到: {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        import flax.serialization
        
        # 加载模型参数
        with open(model_path, 'rb') as f:
            state_dict = flax.serialization.msgpack_deserialize(f.read())
        
        # 恢复模型状态
        self.agent = flax.serialization.from_state_dict(self.agent, state_dict)
        
        print(f"模型从: {model_path} 加载成功")
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        # 绘制回合奖励曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title(f"SAC 在 {self.env_name} 上的训练奖励")
        plt.xlabel("回合")
        plt.ylabel("奖励")
        plt.grid(True)
        
        # 绘制测试奖励曲线
        plt.subplot(1, 2, 2)
        test_steps = [i * self.test_freq for i in range(len(self.test_rewards))]
        plt.plot(test_steps, self.test_rewards, 'o-')
        plt.title(f"SAC 在 {self.env_name} 上的测试奖励")
        plt.xlabel("训练步数")
        plt.ylabel("平均奖励")
        plt.grid(True)
        
        # 保存图表
        plot_path = os.path.join(PLOT_DIR, f"sac_training_{self.env_name}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        print(f"训练曲线保存到: {plot_path}")
    
    def record_final_video(self):
        """
        记录最终模型的视频
        """
        print("记录最终模型的视频...")
        self.test(record_video=True)
        print("视频记录完成")

if __name__ == "__main__":
    # 测试不同的环境（只使用 Pendulum-v1 避免 box2d 依赖）
    environments = [
        'Pendulum-v1'
    ]
    
    for env_name in environments:
        print(f"\n=== 测试环境: {env_name} ===")
        
        # 创建测试器
        tester = SACTester(env_name=env_name, seed=42)
        
        # 训练智能体
        tester.train()
        
        # 记录最终视频
        tester.record_final_video()
        
        print(f"=== 环境 {env_name} 测试完成 ===\n")
