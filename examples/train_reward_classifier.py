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
target_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl"
if os.path.exists(target_path) and target_path not in sys.path:
    sys.path.insert(0, target_path)

# 添加机器人控制 API 路径
manager_path = "/home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control"
if os.path.exists(manager_path) and manager_path not in sys.path:
    sys.path.insert(0, manager_path)

# 设置环境变量：解决 Protobuf 版本冲突，强制使用 Python 纯脚本实现
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import optax
from tqdm import tqdm

# SERL 专用组件导入
from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from experiments.mappings import CONFIG_MAPPING

# --- 命令行参数定义 ---
FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "实验名称，对应 experiments/mappings.py 中的配置键")
flags.DEFINE_integer("num_epochs", 150, "训练迭代总轮数")
flags.DEFINE_integer("batch_size", 256, "每步训练的总 Batch Size (正负样本各占一半)")


def main(_):
    """
    奖励分类器训练主函数。
    逻辑：加载数据 -> 预处理 -> 初始化 ResNet 网络 -> 迭代训练 -> 保存模型。
    """
    assert FLAGS.exp_name in CONFIG_MAPPING, '找不到对应的实验配置，请检查 exp_name。'
    
    # 1. 加载实验配置与环境
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # 使用 fake_env=True，因为此时只需获取 observation_space 结构，无需启动物理机器人
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    # 2. JAX 并行计算设置
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices) # 定义数据分片策略
    
    # 3. 创建并填充【成功样本 (Positive)】缓冲区
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    for path in success_paths:
        with open(path, "rb") as f:
            success_data = pkl.load(f)
            for trans in success_data:
                # 过滤掉不含图像的数据
                if "images" in trans['observations'].keys(): continue
                trans["labels"] = 1 # 成功样本标签为 1
                trans['actions'] = env.action_space.sample() # 随机填充动作占位符
                pos_buffer.insert(trans)
            
    # 定义正样本迭代器，每次获取 batch_size 的一半
    pos_iterator = pos_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
        device=sharding.replicate(),
    )
    
    # 4. 创建并填充【失败样本 (Negative)】缓冲区
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    for path in failure_paths:
        with open(path, "rb") as f:
            failure_data = pkl.load(f)
            for trans in failure_data:
                if "images" in trans['observations'].keys(): continue
                trans["labels"] = 0 # 失败样本标签为 0
                trans['actions'] = env.action_space.sample()
                neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
        device=sharding.replicate(),
    )

    print(f"📊 成功加载数据：失败样本 {len(neg_buffer)} 条, 成功样本 {len(pos_buffer)} 条")

    # 5. 初始化模型
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    # 取样以确定网络输入维度
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    # 创建基于 ResNet 的二分类网络
    classifier = create_classifier(
        key, 
        sample["observations"], 
        config.classifier_keys, # 指定用于分类的摄像头键名（如 'wrist_1'）
    )

    def data_augmentation_fn(rng, observations):
        """
        数据增强函数：对观测值中的图像进行随机裁剪。
        Args:
            rng: JAX 随机密钥
            observations: 原始观测字典
        Returns:
            增强后的观测字典
        """
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        """
        JIT 编译的单步训练函数。
        Args:
            state: TrainState 对象（包含 params 和 opt_state）
            batch: 包含 observations 和 labels 的训练批次
            key: 随机密钥（用于 Dropout）
        Returns:
            new_state, loss, train_accuracy
        """
        def loss_fn(params):
            # 前向传播：计算 Logits
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            # 计算交叉熵损失
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        # 计算梯度与损失
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        # 再次传播以计算准确率（评估模式）
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        # 更新权重
        return state.apply_gradients(grads=grads), loss, train_accuracy

    # 6. 训练循环
    print("🚀 开始训练...")
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # 混合正负样本：构造 1:1 的训练批次
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        batch = concat_batches(pos_sample, neg_sample, axis=0)
        
        # 数据增强与维度调整
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None], # 将标签从 (B,) 调整为 (B, 1)
            }
        )
            
        # 执行一步梯度下降
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}, Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}")

    # 7. 保存模型权重
    ckpt_path = os.path.join(os.getcwd(), "classifier_ckpt/")
    checkpoints.save_checkpoint(
        ckpt_path,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    print(f"💾 模型已保存至: {ckpt_path}")
    

if __name__ == "__main__":
    app.run(main)