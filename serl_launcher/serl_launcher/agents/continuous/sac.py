from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class SACAgent(flax.struct.PyTreeNode):
    """
    在线演员-评论家智能体，支持多种不同的算法配置：
     - SAC (默认)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    
    属性:
        state: JaxRLTrainState，包含网络参数、优化器状态等
        config: dict，算法配置参数
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        评论家网络前向传播
        
        参数:
            observations: Data，环境观测数据
            actions: jax.Array，动作数据
            rng: PRNGKey，随机数种子
            grad_params: Optional[Params]，可选的梯度参数
            train: bool，是否为训练模式
            
        返回:
            jax.Array，评论家网络输出的Q值
        """
        if train:
            assert rng is not None, "训练时必须指定rng"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        目标评论家网络前向传播
        
        参数:
            observations: Data，环境观测数据
            actions: jax.Array，动作数据
            rng: PRNGKey，随机数种子
            
        返回:
            jax.Array，目标评论家网络输出的Q值
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    @jax.jit
    def jitted_forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        目标评论家网络前向传播（JIT编译版本）
        
        参数:
            observations: Data，环境观测数据
            actions: jax.Array，动作数据
            rng: PRNGKey，随机数种子
            
        返回:
            jax.Array，目标评论家网络输出的Q值
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        策略网络前向传播
        
        参数:
            observations: Data，环境观测数据
            rng: Optional[PRNGKey]，随机数种子
            grad_params: Optional[Params]，可选的梯度参数
            train: bool，是否为训练模式
            
        返回:
            distrax.Distribution，动作概率分布
        """
        if train:
            assert rng is not None, "训练时必须指定rng"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        温度拉格朗日乘数前向传播
        
        参数:
            grad_params: Optional[Params]，可选的梯度参数
            
        返回:
            distrax.Distribution，温度值
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        温度拉格朗日惩罚项前向传播
        
        参数:
            entropy: jnp.ndarray，熵值
            grad_params: Optional[Params]，可选的梯度参数
            
        返回:
            distrax.Distribution，拉格朗日惩罚项
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """
        损失函数之间共享的计算：计算下一状态的动作和对数概率
        
        参数:
            batch: Batch，训练数据批次
            rng: PRNGKey，随机数种子
            
        返回:
            Tuple[jax.Array, jax.Array]，下一状态的动作和对数概率
        """
        batch_size = batch["rewards"].shape[0]

        # 计算下一状态的动作分布
        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )
        # 采样动作并计算对数概率
        (
            next_actions,
            next_actions_log_probs,
        ) = next_action_distributions.sample_and_log_prob(seed=rng)
        # 验证动作形状
        chex.assert_equal_shape([batch["actions"], next_actions])
        # 验证对数概率形状
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        评论家网络损失函数
        子类可以重写此函数以更改行为
        
        参数:
            batch: Batch，训练数据批次
            params: Params，网络参数
            rng: PRNGKey，随机数种子
            
        返回:
            Tuple[jax.Array, dict]，损失值和信息字典
        """
        batch_size = batch["rewards"].shape[0]
        # 分割随机数种子
        rng, next_action_sample_key = jax.random.split(rng)
        # 计算下一状态的动作和对数概率
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # 评估所有集成成员的下一Q值（只做前向传播，计算成本低）
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # 如果请求，进行子采样（用于REDQ算法）
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # 计算集成成员中的最小Q值
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        # 计算目标Q值
        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        # 如果配置了备份熵，则在目标Q值中考虑熵项
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        # 计算预测Q值
        predicted_qs = self.forward_critic(
            batch["observations"], batch["actions"], rng=rng, grad_params=params
        )

        # 验证预测Q值形状
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        # 为每个集成成员准备目标Q值
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        # 计算均方误差损失
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        # 准备信息字典
        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        策略网络损失函数
        
        参数:
            batch: Batch，训练数据批次
            params: Params，网络参数
            rng: PRNGKey，随机数种子
            
        返回:
            Tuple[jax.Array, dict]，损失值和信息字典
        """
        batch_size = batch["rewards"].shape[0]
        # 获取当前温度值
        temperature = self.forward_temperature()

        # 分割随机数种子
        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        # 计算当前状态的动作分布
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        # 采样动作并计算对数概率
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        # 计算预测Q值
        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        # 计算平均Q值
        predicted_q = predicted_qs.mean(axis=0)
        # 验证形状
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        # 计算演员目标函数和损失
        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)

        # 准备信息字典
        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        温度参数损失函数
        
        参数:
            batch: Batch，训练数据批次
            params: Params，网络参数
            rng: PRNGKey，随机数种子
            
        返回:
            Tuple[jax.Array, dict]，损失值和信息字典
        """
        # 分割随机数种子
        rng, next_action_sample_key = jax.random.split(rng)
        # 计算下一状态的动作和对数概率
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # 计算熵值
        entropy = -next_actions_log_probs.mean()
        # 计算温度拉格朗日惩罚项
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}
    
    def loss_fns(self, batch):
        """
        创建损失函数字典
        
        参数:
            batch: Batch，训练数据批次
            
        返回:
            dict，包含各个网络的损失函数
        """
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
        **kwargs
    ) -> Tuple["SACAgent", dict]:
        """
        对智能体的网络进行一次梯度更新
        
        参数:
            batch: Batch，训练数据批次，应包含以下键：
                "observations", "actions", "next_observations", "rewards", "masks"
            pmap_axis: Optional[str]，用于pmap的轴（如果为None，则不使用pmap）
            networks_to_update: FrozenSet[str]，要更新的网络名称集合（默认：所有网络）
                例如，在高UTD设置中，通常多次更新评论家，而只更新一次演员（和其他网络）
            
        返回:
            Tuple[SACAgent, dict]，更新后的智能体和信息字典
        """
        batch_size = batch["rewards"].shape[0]
        # 验证批次形状
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        # 如果批次中没有图像键，则解包批次
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        # 分割随机数种子
        rng, aug_rng = jax.random.split(self.state.rng)
        # 如果配置了数据增强函数，则应用数据增强
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        # 添加奖励偏差
        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # 计算梯度并更新参数
        loss_fns = self.loss_fns(batch, **kwargs)

        # 只计算指定网络的梯度
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"无效的梯度步骤：{networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        # 应用损失函数并更新状态
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # 如果更新了评论家，则更新目标网络
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # 更新随机数种子
        new_state = new_state.replace(rng=rng)

        # 记录学习率
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        从策略网络采样动作，使用外部RNG（或通过模式近似argmax）
        内部RNG不会更新
        
        参数:
            observations: Data，环境观测数据
            seed: Optional[PRNGKey]，外部随机数种子
            argmax: bool，是否使用argmax（确定性策略）
            
        返回:
            jnp.ndarray，采样的动作
        """

        # 前向传播策略网络
        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            # 使用确定性策略（分布的模式）
            return dist.mode()
        else:
            # 从分布中采样
            return dist.sample(seed=seed)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # 模型定义
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # 优化器参数
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # 算法配置
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        """
        创建一个新的SAC智能体
        
        参数:
            rng: PRNGKey，随机数种子
            observations: Data，环境观测数据示例
            actions: jnp.ndarray，动作数据示例
            actor_def: nn.Module，演员网络定义
            critic_def: nn.Module，评论家网络定义
            temperature_def: nn.Module，温度网络定义
            actor_optimizer_kwargs: dict，演员优化器参数
            critic_optimizer_kwargs: dict，评论家优化器参数
            temperature_optimizer_kwargs: dict，温度优化器参数
            discount: float，折扣因子
            soft_target_update_rate: float，软目标更新率
            target_entropy: Optional[float]，目标熵值
            entropy_per_dim: bool，是否按维度计算熵
            backup_entropy: bool，是否在备份中包含熵
            critic_ensemble_size: int，评论家集成大小
            critic_subsample_size: Optional[int]，评论家子采样大小
            image_keys: Iterable[str]，图像键列表
            augmentation_function: Optional[callable]，数据增强函数
            reward_bias: float，奖励偏差
            
        返回:
            SACAgent，创建的智能体
        """
        # 创建网络字典
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }

        # 创建模块字典
        model_def = ModuleDict(networks)

        # 定义优化器
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        # 初始化参数
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
        )["params"]

        # 创建训练状态
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # 配置
        assert not entropy_per_dim, "未实现"
        if target_entropy is None:
            # 如果未指定目标熵，则使用默认值
            target_entropy = -actions.shape[-1] / 2

        # 创建智能体
        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # 模型架构
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        """
        创建一个新的基于像素的智能体
        
        参数:
            rng: PRNGKey，随机数种子
            observations: Data，环境观测数据示例
            actions: jnp.ndarray，动作数据示例
            encoder_type: str，编码器类型
            use_proprio: bool，是否使用本体感觉信息
            critic_network_kwargs: dict，评论家网络参数
            policy_network_kwargs: dict，策略网络参数
            policy_kwargs: dict，策略参数
            critic_ensemble_size: int，评论家集成大小
            critic_subsample_size: Optional[int]，评论家子采样大小
            temperature_init: float，温度初始值
            image_keys: Iterable[str]，图像键列表
            augmentation_function: Optional[callable]，数据增强函数
            
        返回:
            SACAgent，创建的基于像素的智能体
        """

        # 确保网络的最终层被激活
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        # 根据编码器类型创建编码器
        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            # 创建预训练编码器
            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"未知的编码器类型: {encoder_type}")

        # 创建编码包装器
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        # 为演员和评论家创建编码器
        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # 定义网络
        # 创建评论家集成
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        # 创建评论家网络
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        # 创建策略网络
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        # 创建温度网络
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        # 创建智能体
        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        # 如果使用预训练编码器，则加载预训练权重
        if "pretrained" in encoder_type:  # 为ResNet-10加载预训练权重
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
