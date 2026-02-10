"""
ResNet V1 编码器实现

本文件实现了基于 ResNet V1 架构的视觉编码器，用于从图像观测中提取特征表示。
主要包含以下组件：
- AddSpatialCoordinates: 向特征图添加空间坐标信息
- SpatialSoftmax: 空间softmax池化
- SpatialLearnedEmbeddings: 空间学习嵌入
- MyGroupNorm: 组归一化层
- ResNetBlock: 基础ResNet块
- BottleneckResNetBlock: 瓶颈ResNet块
- ResNetEncoder: ResNet V1编码器主类
- PreTrainedResNetEncoder: 预训练ResNet编码器包装器

作者: SERL团队
"""

import functools as ft
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.vision.film_conditioning_layer import FilmConditioning
from serl_launcher.vision.data_augmentations import resize
ModuleDef = Any


class AddSpatialCoordinates(nn.Module):
    """
    向特征图添加空间坐标信息的模块
    
    功能说明：
    该模块将归一化的空间坐标（x, y）添加到输入特征图的通道维度中。
    这有助于网络理解特征在空间中的位置，增强空间感知能力。
    
    参数:
        dtype: 数据类型，默认为 jnp.float32
    
    输入:
        x: 输入特征图，形状为 (batch, height, width, channels) 或 (height, width, channels)
    
    输出:
        添加了空间坐标的特征图，形状为 (batch, height, width, channels+2) 或 (height, width, channels+2)
        额外的2个通道分别是归一化的x和y坐标
    """
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """
        前向传播，添加空间坐标
        
        逻辑流程：
        1. 创建归一化的空间坐标网格，范围从-1到1
        2. 将坐标网格广播到批次维度（如果存在）
        3. 将坐标拼接到输入特征图的通道维度
        
        参数:
            x: 输入特征图，形状为 (batch, height, width, channels) 或 (height, width, channels)
        
        返回:
            添加了空间坐标的特征图
        """
        # 创建归一化的空间坐标网格
        # np.meshgrid 生成坐标网格，范围从0到1
        # 乘以2再减1将范围归一化到[-1, 1]
        # * [s - 1] 确保坐标覆盖整个特征图
        grid = jnp.array(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        ).transpose(1, 0, 2)

        # 如果输入有批次维度，将坐标网格广播到批次维度
        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, [x.shape[0], *grid.shape])

        # 将坐标网格拼接到输入特征图的通道维度
        # 原始特征图有C个通道，添加2个坐标通道后变为C+2个通道
        return jnp.concatenate([x, grid], axis=-1)


class SpatialSoftmax(nn.Module):
    """
    空间softmax池化模块
    
    功能说明：
    该模块使用空间softmax对特征图进行池化，计算每个特征图的空间加权平均位置。
    与传统的平均池化或最大池化不同，空间softmax可以保留空间位置信息，
    对于需要精确空间定位的任务（如抓取、操作）非常有用。
    
    参数:
        height: 特征图高度
        width: 特征图宽度
        channel: 特征图通道数
        pos_x: x方向的归一化坐标数组，形状为 (height * width,)
        pos_y: y方向的归一化坐标数组，形状为 (height * width,)
        temperature: softmax温度参数，-1表示可学习温度
        log_heatmap: 是否返回对数热图（默认False）
    
    输入:
        features: 输入特征图，形状为 (batch, height, width, channel)
    
    输出:
        expected_xy: 每个特征图的期望位置，形状为 (batch, 2 * channel)
        每个特征图返回2个值（x和y坐标）
    """
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, features):
        """
        前向传播，执行空间softmax池化
        
        逻辑流程：
        1. 确定softmax温度参数
        2. 重塑特征图为 (batch, channel, height*width)
        3. 应用softmax计算空间注意力权重
        4. 计算期望的x和y坐标
        5. 拼接期望坐标并返回
        
        参数:
            features: 输入特征图，形状为 (batch, height, width, channel)
        
        返回:
            expected_xy: 每个特征图的期望位置
        """
        # 确定softmax温度参数
        # temperature=-1表示使用可学习的温度参数
        if self.temperature == -1:
            from jax.nn import initializers

            # 创建可学习的温度参数，初始值为1
            temperature = self.param(
                "softmax_temperature", initializers.ones, (1), jnp.float32
            )
        else:
            # 使用固定的温度参数
            temperature = 1.0

        # 如果输入没有批次维度，添加批次维度
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        # 验证特征图形状
        assert len(features.shape) == 4
        
        # 获取批次大小和特征图数量
        batch_size, num_featuremaps = features.shape[0], features.shape[3]
        
        # 重塑特征图为 (batch, channel, height*width)
        # 先转置为 (batch, channel, height, width)
        # 再重塑为 (batch, channel, height*width)
        features = features.transpose(0, 3, 1, 2).reshape(
            batch_size, num_featuremaps, self.height * self.width
        )

        # 应用softmax计算空间注意力权重
        # 温度参数控制softmax的尖锐程度
        # 温度越高，分布越平滑；温度越低，分布越尖锐
        softmax_attention = nn.softmax(features / temperature)
        
        # 计算期望的x坐标
        # 使用softmax权重对x坐标进行加权平均
        expected_x = jnp.sum(
            self.pos_x * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        
        # 计算期望的y坐标
        # 使用softmax权重对y坐标进行加权平均
        expected_y = jnp.sum(
            self.pos_y * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        
        # 拼接x和y坐标
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        # 重塑为 (batch, 2 * num_featuremaps)
        # 每个特征图对应2个值（x和y）
        expected_xy = jnp.reshape(expected_xy, [batch_size, 2 * num_featuremaps])

        # 如果输入没有批次维度，移除批次维度
        if no_batch_dim:
            expected_xy = expected_xy[0]
        
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    """
    空间学习嵌入模块
    
    功能说明：
    该模块使用可学习的空间嵌入对特征图进行池化。
    与空间softmax不同，这里使用可学习的权重而不是softmax权重，
    可以学习到更适合任务的空间聚合方式。
    
    参数:
        height: 特征图高度
        width: 特征图宽度
        channel: 特征图通道数
        num_features: 输出特征数量，默认为5
        kernel_init: 卷积核初始化器，默认为lecun_normal
        param_dtype: 参数数据类型，默认为jnp.float32
    
    输入:
        features: 输入特征图，形状为 (batch, height, width, channel)
    
    输出:
        features: 学习到的空间嵌入，形状为 (batch, num_features)
    """
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = nn.initializers.lecun_normal()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        """
        前向传播，计算学习到的空间嵌入
        
        逻辑流程：
        1. 创建可学习的空间权重卷积核
        2. 使用逐元素乘法将权重应用于特征图
        3. 在空间维度上求和得到嵌入向量
        
        参数:
            features: 输入特征图，形状为 (batch, height, width, channel)
        
        返回:
            features: 学习到的空间嵌入
        """
        """
        输入特征图形状说明：
        features is B x H x W x C
        B: batch size（批次大小）
        H: height（高度）
        W: width（宽度）
        C: channel（通道数）
        """
        # 创建可学习的空间权重卷积核
        # 形状为 (height, width, channel, num_features)
        # 每个空间位置和通道都有对应的权重
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.height, self.width, self.channel, self.num_features),
            self.param_dtype,
        )

        # 如果输入没有批次维度，添加批次维度
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        # 获取批次大小
        batch_size = features.shape[0]
        
        # 验证特征图形状
        assert len(features.shape) == 4
        
        # 使用逐元素乘法将权重应用于特征图
        # 先扩展维度以支持广播：
        # features: (batch, height, width, channel) -> (batch, height, width, channel, 1)
        # kernel: (height, width, channel, num_features) -> (1, height, width, channel, num_features)
        # 乘法结果: (batch, height, width, channel, num_features)
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2)
        )
        
        # 在高度、宽度和通道维度上求和，得到嵌入向量
        # 形状从 (batch, height, width, channel, num_features) 变为 (batch, num_features)
        features = jnp.reshape(features, [batch_size, -1])

        # 如果输入没有批次维度，移除批次维度
        if no_batch_dim:
            features = features[0]

        return features


class MyGroupNorm(nn.GroupNorm):
    """
    自定义组归一化层
    
    功能说明：
    继承自Flax的GroupNorm，添加了对3D输入的支持。
    标准的GroupNorm要求4D输入（batch, height, width, channels），
    该类允许3D输入（height, width, channels），自动添加批次维度。
    
    使用场景：
    - 当输入没有批次维度时，自动添加批次维度进行归一化
    - 归一化后移除添加的批次维度，保持输入输出形状一致
    
    参数:
        继承自nn.GroupNorm的所有参数
    """
    def __call__(self, x):
        """
        前向传播，执行组归一化
        
        逻辑流程：
        1. 检查输入维度
        2. 如果是3D输入，添加批次维度
        3. 执行组归一化
        4. 如果添加了批次维度，移除它
        
        参数:
            x: 输入张量，形状为 (batch, height, width, channels) 或 (height, width, channels)
        
        返回:
            x: 归一化后的张量，形状与输入相同
        """
        # 如果输入是3D（没有批次维度），添加批次维度
        if x.ndim == 3:
            x = x[jnp.newaxis]
            # 执行组归一化
            x = super().__call__(x)
            # 移除添加的批次维度
            return x[0]
        else:
            # 输入已经是4D，直接执行组归一化
            return super().__call__(x)


class ResNetBlock(nn.Module):
    """
    基础ResNet块
    
    功能说明：
    实现标准的ResNet残差块，包含两个3x3卷积层和跳跃连接。
    残差连接允许梯度更容易地流过网络，缓解深层网络的梯度消失问题。
    
    参数:
        filters: 输出通道数
        conv: 卷积层定义（如nn.Conv）
        norm: 归一化层定义（如MyGroupNorm）
        act: 激活函数（如nn.relu）
        strides: 卷积步长，默认为(1, 1)
    
    输入:
        x: 输入特征图，形状为 (batch, height, width, in_channels)
    
    输出:
        output: 输出特征图，形状为 (batch, height', width', filters)
        如果strides=(2,2)，则高度和宽度减半
    """
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        """
        前向传播，执行ResNet块计算
        
        逻辑流程：
        1. 保存输入作为残差
        2. 第一个3x3卷积 -> 归一化 -> 激活
        3. 第二个3x3卷积 -> 归一化
        4. 如果输入输出形状不同，对残差进行投影
        5. 激活(残差 + 输出)
        
        参数:
            x: 输入特征图
        
        返回:
            output: 残差块的输出
        """
        # 保存输入作为残差连接
        residual = x
        
        # 第一个3x3卷积层
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        # 归一化
        y = self.norm()(y)
        # 激活函数
        y = self.act(y)
        
        # 第二个3x3卷积层（步长为1）
        y = self.conv(self.filters, (3, 3))(y)
        # 归一化
        y = self.norm()(y)

        # 如果输入输出形状不同，对残差进行投影
        # 使用1x1卷积调整通道数和空间尺寸
        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        # 残差连接 + 激活
        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """
    瓶颈ResNet块
    
    功能说明：
    实现ResNet的瓶颈块，用于更深的网络（如ResNet-50）。
    瓶颈结构使用1x1-3x3-1x1卷积序列，减少计算量：
    - 第一个1x1卷积：降低通道数（压缩）
    - 3x3卷积：主要计算
    - 最后一个1x1卷积：恢复通道数（扩展）
    
    参数:
        filters: 基础通道数，输出通道数为filters*4
        conv: 卷积层定义
        norm: 归一化层定义
        act: 激活函数
        strides: 卷积步长，默认为(1, 1)
    
    输入:
        x: 输入特征图
    
    输出:
        output: 输出特征图，通道数为filters*4
    """
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        """
        前向传播，执行瓶颈ResNet块计算
        
        逻辑流程：
        1. 保存输入作为残差
        2. 1x1卷积（压缩） -> 归一化 -> 激活
        3. 3x3卷积（主要计算） -> 归一化 -> 激活
        4. 1x1卷积（扩展） -> 归一化（零初始化）
        5. 如果输入输出形状不同，对残差进行投影
        6. 激活(残差 + 输出)
        
        参数:
            x: 输入特征图
        
        返回:
            output: 瓶颈块的输出
        """
        # 保存输入作为残差连接
        residual = x
        
        # 第一个1x1卷积：降低通道数
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        
        # 3x3卷积：主要计算，可能下采样
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        
        # 最后一个1x1卷积：恢复通道数（4倍扩展）
        y = self.conv(self.filters * 4, (1, 1))(y)
        # 使用零初始化的最后一个归一化层，有助于训练稳定性
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        # 如果输入输出形状不同，对残差进行投影
        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        # 残差连接 + 激活
        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """
    ResNet V1编码器
    
    功能说明：
    实现完整的ResNet V1编码器，用于从图像中提取特征表示。
    支持多种配置选项，包括：
    - 不同的网络深度（10, 18, 34, 50层）
    - 不同的池化方法（平均、最大、空间softmax、学习嵌入）
    - 可选的空间坐标添加
    - 可选的FILM条件化
    - 可选的瓶颈层
    
    参数:
        stage_sizes: 每个阶段的块数，如(2,2,2,2)表示4个阶段各有2个块
        block_cls: 使用的块类型（ResNetBlock或BottleneckResNetBlock）
        num_filters: 初始通道数，默认为64
        dtype: 数据类型，默认为jnp.float32
        act: 激活函数名称，默认为"relu"
        conv: 卷积层类型，默认为nn.Conv
        norm: 归一化类型（"group"、"batch"、"layer"）
        add_spatial_coordinates: 是否添加空间坐标，默认为False
        pooling_method: 池化方法（"avg"、"max"、"spatial_softmax"、"spatial_learned_embeddings"）
        use_spatial_softmax: 是否使用空间softmax（已弃用，使用pooling_method）
        softmax_temperature: 空间softmax温度，默认为1.0
        use_multiplicative_cond: 是否使用乘法条件化，默认为False
        num_spatial_blocks: 空间块数量，默认为8
        use_film: 是否使用FILM条件化，默认为False
        bottleneck_dim: 瓶颈层维度，None表示不使用
        pre_pooling: 是否在池化前返回特征，默认为True
        image_size: 输入图像尺寸，默认为(128, 128)
    
    输入:
        observations: 输入图像，形状为 (batch, height, width, channels)
        train: 是否为训练模式，默认为True
        cond_var: 条件变量（用于FILM或乘法条件化）
        stop_gradient: 是否停止梯度传播，默认为False
    
    输出:
        x: 编码后的特征，形状取决于池化方法和瓶颈层
    """
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"
    add_spatial_coordinates: bool = False
    pooling_method: str = "avg"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    use_multiplicative_cond: bool = False
    num_spatial_blocks: int = 8
    use_film: bool = False
    bottleneck_dim: Optional[int] = None
    pre_pooling: bool = True
    image_size: tuple = (128, 128)

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        train: bool = True,
        cond_var=None,
        stop_gradient=False,
    ):
        """
        前向传播，执行ResNet编码
        
        逻辑流程：
        1. 调整图像大小（如果需要）
        2. 归一化图像（ImageNet统计量）
        3. 可选：添加空间坐标
        4. 初始7x7卷积 -> 归一化 -> 激活 -> 最大池化
        5. 多个阶段的ResNet块
        6. 可选：应用条件化（FILM或乘法）
        7. 可选：停止梯度
        8. 池化（多种方法）
        9. 可选：瓶颈层
        
        参数:
            observations: 输入图像，形状为 (batch, height, width, channels)
            train: 是否为训练模式
            cond_var: 条件变量
            stop_gradient: 是否停止梯度传播
        
        返回:
            x: 编码后的特征
        """
        # 调整图像大小（如果与配置不同）
        if observations.shape[-3:-1] != self.image_size:
            observations = resize(observations, self.image_size)

        # ImageNet数据集的均值和标准差
        # 这些统计量用于预训练模型的归一化
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        
        # 归一化图像：先缩放到[0,1]，然后应用ImageNet归一化
        # 公式：(x/255 - mean) / std
        x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        # 可选：添加空间坐标
        # 将归一化的x,y坐标添加到输入中
        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates(dtype=self.dtype)(x)

        # 创建偏置为False的卷积层（组归一化不需要偏置）
        # 使用Kaiming正态初始化（适合ReLU激活）
        conv = partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        
        # 创建归一化层
        if self.norm == "batch":
            # 批归一化（未实现）
            raise NotImplementedError
        elif self.norm == "group":
            # 组归一化（4个组）
            norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            # 层归一化
            norm = partial(
                nn.LayerNorm,
                epsilon=1e-5,
                dtype=self.dtype,
            )
        else:
            # 未知的归一化类型
            raise ValueError("norm not found")

        # 获取激活函数
        act = getattr(nn, self.act)

        # 初始7x7卷积，步长为2
        # 这是ResNet的标准初始层，用于下采样和特征提取
        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)

        # 归一化
        x = norm(name="norm_init")(x)
        # 激活
        x = act(x)
        # 3x3最大池化，步长为2
        # 进一步下采样，减少空间维度
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        
        # 构建ResNet的各个阶段
        # 每个阶段可能包含多个ResNet块
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # 第一个阶段的第一个块使用步长1
                # 后续阶段的第一个块使用步长2进行下采样
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                
                # 应用ResNet块
                # 通道数随阶段指数增长：num_filters * 2^i
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)
                
                # 可选：应用FILM条件化
                # FILM (Feature-wise Linear Modulation) 允许网络根据条件变量调整特征
                if self.use_film:
                    # 确保提供了条件变量
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    # 应用FILM条件化
                    x = FilmConditioning()(x, cond_var)
                
                # 可选：应用乘法条件化
                # 使用条件变量对特征进行逐元素缩放
                if self.use_multiplicative_cond:
                    # 确保提供了条件变量
                    assert (
                        cond_var is not None
                    ), "Cond var is None, nothing to condition on"
                    # 将条件变量映射到特征维度
                    cond_out = nn.Dense(
                        x.shape[-1], kernel_init=nn.initializers.xavier_normal()
                    )(cond_var)
                    # 扩展维度以支持广播
                    x_mult = jnp.expand_dims(jnp.expand_dims(cond_out, 1), 1)
                    # 逐元素乘法
                    x = x * x_mult
        
        # 如果pre_pooling为True，在池化前返回特征
        # 用于需要空间特征的任务（如注意力机制）
        if self.pre_pooling:
            # 停止梯度传播（如果需要）
            return jax.lax.stop_gradient(x)
            # return x  # 备选方案

        # 根据配置的池化方法进行池化
        if self.pooling_method == "spatial_learned_embeddings":
            # 空间学习嵌入池化
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
            # 训练时应用dropout
            x = nn.Dropout(0.1, deterministic=not train)(x)
            
        elif self.pooling_method == "spatial_softmax":
            # 空间softmax池化
            # 保留空间位置信息，适合需要精确定位的任务
            height, width, channel = x.shape[-3:]
            # 创建归一化的空间坐标网格
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
            )
            # 展平坐标数组
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            # 应用空间softmax
            x = SpatialSoftmax(
                height, width, channel, pos_x, pos_y, self.softmax_temperature
            )(x)
            
        elif self.pooling_method == "avg":
            # 平均池化
            # 对空间维度求平均
            x = jnp.mean(x, axis=(-3, -2))
            
        elif self.pooling_method == "max":
            # 最大池化
            # 对空间维度取最大值
            x = jnp.max(x, axis=(-3, -2))
            
        elif self.pooling_method == "none":
            # 不进行池化
            pass
            
        else:
            # 未知的池化方法
            raise ValueError("pooling method not found")

        # 可选：添加瓶颈层
        # 降低特征维度，减少计算量
        if self.bottleneck_dim is not None:
            # 全连接层降维
            x = nn.Dense(self.bottleneck_dim)(x)
            # 层归一化
            x = nn.LayerNorm()(x)
            # tanh激活，限制输出范围
            x = nn.tanh(x)

        return x


class PreTrainedResNetEncoder(nn.Module):
    """
    预训练ResNet编码器包装器
    
    功能说明：
    包装预训练的ResNet编码器，提供统一的接口。
    允许使用预训练的权重，并支持不同的池化方法。
    
    参数:
        pooling_method: 池化方法，默认为"avg"
        use_spatial_softmax: 是否使用空间softmax（已弃用）
        softmax_temperature: 空间softmax温度，默认为1.0
        num_spatial_blocks: 空间块数量，默认为8
        bottleneck_dim: 瓶颈层维度，None表示不使用
        pretrained_encoder: 预训练的编码器模块
    
    输入:
        observations: 输入图像
        encode: 是否进行编码，默认为True
        train: 是否为训练模式，默认为True
    
    输出:
        x: 编码后的特征
    """
    pooling_method: str = "avg"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    num_spatial_blocks: int = 8
    bottleneck_dim: Optional[int] = None
    pretrained_encoder: nn.module = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        encode: bool = True,
        train: bool = True,
    ):
        """
        前向传播，使用预训练编码器提取特征
        
        逻辑流程：
        1. 如果encode为True，使用预训练编码器
        2. 应用配置的池化方法
        3. 可选：添加瓶颈层
        
        参数:
            observations: 输入图像
            encode: 是否进行编码
            train: 是否为训练模式
        
        返回:
            x: 编码后的特征
        """
        x = observations
        
        # 如果需要编码，使用预训练编码器
        if encode:
            x = self.pretrained_encoder(x, train=train)

        # 根据配置的池化方法进行池化
        if self.pooling_method == "spatial_learned_embeddings":
            # 空间学习嵌入池化
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
            # 训练时应用dropout
            x = nn.Dropout(0.1, deterministic=not train)(x)
            
        elif self.pooling_method == "spatial_softmax":
            # 空间softmax池化
            height, width, channel = x.shape[-3:]
            # 创建归一化的空间坐标网格
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
            )
            # 展平坐标数组
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            # 应用空间softmax
            x = SpatialSoftmax(
                height, width, channel, pos_x, pos_y, self.softmax_temperature
            )(x)
            
        elif self.pooling_method == "avg":
            # 平均池化
            x = jnp.mean(x, axis=(-3, -2))
            
        elif self.pooling_method == "max":
            # 最大池化
            x = jnp.max(x, axis=(-3, -2))
            
        elif self.pooling_method == "none":
            # 不进行池化
            pass
            
        else:
            # 未知的池化方法
            raise ValueError("pooling method not found")

        # 可选：添加瓶颈层
        if self.bottleneck_dim is not None:
            # 全连接层降维
            x = nn.Dense(self.bottleneck_dim)(x)
            # 层归一化
            x = nn.LayerNorm()(x)
            # tanh激活
            x = nn.tanh(x)

        return x


# ResNet V1配置字典
# 包含各种预定义的ResNet配置，可以通过名称直接使用
resnetv1_configs = {
    # ResNet-10: 最小的ResNet，每个阶段1个块
    # 适用于计算资源受限的场景
    "resnetv1-10": ft.partial(
        ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock
    ),
    
    # ResNet-10 Frozen: 在池化前返回特征
    # 保留空间信息，适合需要空间特征的任务
    "resnetv1-10-frozen": ft.partial(
        ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock, pre_pooling=True
    ),
    
    # ResNet-18: 标准的轻量级ResNet
    # 平衡了性能和计算成本
    "resnetv1-18": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock
    ),
    
    # ResNet-18 Frozen: 在池化前返回特征
    "resnetv1-18-frozen": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock, pre_pooling=True
    ),
    
    # ResNet-34: 更深的ResNet，性能更好
    "resnetv1-34": ft.partial(
        ResNetEncoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock
    ),
    
    # ResNet-50: 使用瓶颈块的ResNet
    # 更深的网络，性能更强，但计算量更大
    "resnetv1-50": ft.partial(
        ResNetEncoder, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
    ),
    
    # ResNet-18 Deeper: 每个阶段3个块
    # 比标准ResNet-18更深
    "resnetv1-18-deeper": ft.partial(
        ResNetEncoder, stage_sizes=(3, 3, 3, 3), block_cls=ResNetBlock
    ),
    
    # ResNet-18 Deepest: 每个阶段4个块
    # 最深的ResNet-18变体
    "resnetv1-18-deepest": ft.partial(
        ResNetEncoder, stage_sizes=(4, 4, 4, 4), block_cls=ResNetBlock
    ),
    
    # ResNet-18 Bridge: 使用空间学习嵌入
    # 适合需要空间定位的任务（如抓取）
    "resnetv1-18-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(2, 2, 2, 2),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    
    # ResNet-34 Bridge: 更深的网络，使用空间学习嵌入
    "resnetv1-34-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    
    # ResNet-34 Bridge with FILM: 使用FILM条件化
    # 允许网络根据条件变量调整特征
    "resnetv1-34-bridge-film": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
        use_film=True,
    ),
    
    # ResNet-50 Bridge: 使用瓶颈块和空间学习嵌入
    # 最强大的配置，适合复杂任务
    "resnetv1-50-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=BottleneckResNetBlock,
        num_spatial_blocks=8,
    ),
    
    # ResNet-50 Bridge with FILM: 最强大的配置
    # 结合了瓶颈块、空间学习嵌入和FILM条件化
    "resnetv1-50-bridge-film": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=BottleneckResNetBlock,
        num_spatial_blocks=8,
        use_film=True,
    ),
}


# ============================================================================
# 测试代码
# 说明：以下代码用于测试各个模块的功能
# 使用方法：取消注释相应的测试函数，然后运行脚本
# ============================================================================

# 取消下面的注释以运行完整测试
def test_all_modules():
    import jax
    
    print("=" * 80)
    print("开始测试 ResNet V1 编码器的各个模块")
    print("=" * 80)
    
    # 测试参数
    batch_size = 2
    height = 32
    width = 32
    channels = 3
    rng = jax.random.PRNGKey(42)
    
    # 创建测试输入
    print("\\n【步骤1】创建测试输入")
    print(f"输入形状: (batch_size={batch_size}, height={height}, width={width}, channels={channels})")
    test_input = jax.random.uniform(rng, (batch_size, height, width, channels))
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    # 测试 AddSpatialCoordinates
    print("\\n【步骤2】测试 AddSpatialCoordinates 模块")
    print("-" * 80)
    print("功能：向特征图添加空间坐标信息")
    add_coord_module = AddSpatialCoordinates()
    # 初始化模块（即使是无参数模块也需要初始化）
    rng, init_rng = jax.random.split(rng)
    params = add_coord_module.init(init_rng, test_input)
    with_coord_output = add_coord_module.apply(params, test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {with_coord_output.shape}")
    print(f"输出说明: 原始{channels}通道 + 2个坐标通道 = {with_coord_output.shape[-1]}通道")
    print(f"坐标通道范围: x=[{with_coord_output[..., -2].min():.4f}, {with_coord_output[..., -2].max():.4f}], "
          f"y=[{with_coord_output[..., -1].min():.4f}, {with_coord_output[..., -1].max():.4f}]")
    
    # 测试 SpatialSoftmax
    print("\\n【步骤3】测试 SpatialSoftmax 模块")
    print("-" * 80)
    print("功能：空间softmax池化，计算期望位置")
    # 创建测试特征图
    test_features = jax.random.uniform(rng, (batch_size, height, width, 8))
    # 创建坐标网格
    pos_x, pos_y = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
    )
    pos_x = pos_x.reshape(height * width)
    pos_y = pos_y.reshape(height * width)
    print(f"输入特征图形状: {test_features.shape}")
    print(f"坐标网格形状: pos_x={pos_x.shape}, pos_y={pos_y.shape}")
    spatial_softmax_module = SpatialSoftmax(height, width, 8, pos_x, pos_y, temperature=1.0)
    # 初始化模块参数
    rng, init_rng = jax.random.split(rng)
    params = spatial_softmax_module.init(init_rng, test_features)
    expected_xy = spatial_softmax_module.apply(params, test_features)
    print(f"输出形状: {expected_xy.shape}")
    print(f"输出说明: 每个特征图返回2个值（x和y坐标），共{8}个特征图")
    print(f"期望位置范围: x=[{expected_xy[:, 0::2].min():.4f}, {expected_xy[:, 0::2].max():.4f}], "
          f"y=[{expected_xy[:, 1::2].min():.4f}, {expected_xy[:, 1::2].max():.4f}]")
    
    # 测试 SpatialLearnedEmbeddings
    print("\\n【步骤4】测试 SpatialLearnedEmbeddings 模块")
    print("-" * 80)
    print("功能：使用可学习的空间嵌入进行池化")
    spatial_emb_module = SpatialLearnedEmbeddings(height=height, width=width, channel=8, num_features=5)
    # 初始化模块参数
    rng, init_rng = jax.random.split(rng)
    params = spatial_emb_module.init(init_rng, test_features)
    spatial_emb_output = spatial_emb_module.apply(params, test_features)
    print(f"输入特征图形状: {test_features.shape}")
    print(f"输出形状: {spatial_emb_output.shape}")
    print(f"输出说明: 从空间维度池化到{5}个特征向量")
    print(f"输出范围: [{spatial_emb_output.min():.4f}, {spatial_emb_output.max():.4f}]")
    
    # 测试 MyGroupNorm
    print("\\n【步骤5】测试 MyGroupNorm 模块")
    print("-" * 80)
    print("功能：组归一化，支持3D和4D输入")
    # 测试4D输入（使用8个通道，可以被4个组整除）
    test_input_4d = jax.random.uniform(rng, (batch_size, height, width, 8))
    group_norm_module = MyGroupNorm(num_groups=4, epsilon=1e-5)
    # 初始化模块参数
    rng, init_rng = jax.random.split(rng)
    params = group_norm_module.init(init_rng, test_input_4d)
    norm_output_4d = group_norm_module.apply(params, test_input_4d)
    print(f"4D输入形状: {test_input_4d.shape}")
    print(f"4D输出形状: {norm_output_4d.shape}")
    print(f"4D输出均值: {norm_output_4d.mean():.6f}, 标准差: {norm_output_4d.std():.6f}")
    
    # 测试3D输入
    test_input_3d = jax.random.uniform(rng, (height, width, 8))
    norm_output_3d = group_norm_module.apply(params, test_input_3d)
    print(f"3D输入形状: {test_input_3d.shape}")
    print(f"3D输出形状: {norm_output_3d.shape}")
    print(f"3D输出均值: {norm_output_3d.mean():.6f}, 标准差: {norm_output_3d.std():.6f}")
    
    # 测试 ResNetBlock
    print("\\n【步骤6】测试 ResNetBlock 模块")
    print("-" * 80)
    print("功能：基础ResNet块，包含残差连接")
    # 创建测试输入
    test_block_input = jax.random.uniform(rng, (batch_size, height, width, 64))
    resnet_block = ResNetBlock(filters=128, conv=nn.Conv, norm=MyGroupNorm, act=nn.relu)
    # 初始化模块参数
    rng, init_rng = jax.random.split(rng)
    params = resnet_block.init(init_rng, test_block_input)
    block_output = resnet_block.apply(params, test_block_input)
    print(f"输入形状: {test_block_input.shape}")
    print(f"输出形状: {block_output.shape}")
    print(f"输出通道数: {block_output.shape[-1]} (与filters参数一致)")
    print(f"输出均值: {block_output.mean():.6f}, 标准差: {block_output.std():.6f}")
    
    # 测试 BottleneckResNetBlock
    print("\\n【步骤7】测试 BottleneckResNetBlock 模块")
    print("-" * 80)
    print("功能：瓶颈ResNet块，使用1x1-3x3-1x1卷积序列")
    bottleneck_block = BottleneckResNetBlock(filters=64, conv=nn.Conv, norm=MyGroupNorm, act=nn.relu)
    # 初始化模块参数
    rng, init_rng = jax.random.split(rng)
    params = bottleneck_block.init(init_rng, test_block_input)
    bottleneck_output = bottleneck_block.apply(params, test_block_input)
    print(f"输入形状: {test_block_input.shape}")
    print(f"输出形状: {bottleneck_output.shape}")
    print(f"输出通道数: {bottleneck_output.shape[-1]} (filters*4 = 64*4 = 256)")
    print(f"输出均值: {bottleneck_output.mean():.6f}, 标准差: {bottleneck_output.std():.6f}")
    
    # 测试 ResNetEncoder
    print("\\n【步骤8】测试 ResNetEncoder 模块")
    print("-" * 80)
    print("功能：完整的ResNet V1编码器")
    # 创建测试图像输入
    test_image = jax.random.uniform(rng, (batch_size, 128, 128, 3))
    print(f"输入图像形状: {test_image.shape}")
    
    # 测试不同的池化方法
    pooling_methods = ["avg", "max", "spatial_softmax", "spatial_learned_embeddings"]
    
    for method in pooling_methods:
        print(f"\\n  测试池化方法: {method}")
        print("  " + "-" * 76)
        
        # 创建编码器
        encoder_config = resnetv1_configs["resnetv1-10"]
        encoder = encoder_config(pooling_method=method)
        
        # 初始化模块参数
        rng, init_rng = jax.random.split(rng)
        params = encoder.init(init_rng, test_image)
        
        # 前向传播
        encoder_output = encoder.apply(params, test_image, train=False)
        
        print(f"  输入形状: {test_image.shape}")
        print(f"  输出形状: {encoder_output.shape}")
        print(f"  输出维度: {encoder_output.shape[-1]}")
        print(f"  输出范围: [{encoder_output.min():.4f}, {encoder_output.max():.4f}]")
    
    # 测试带空间坐标的编码器
    print("\\n  测试带空间坐标的编码器")
    print("  " + "-" * 76)
    encoder_with_coord = encoder_config(add_spatial_coordinates=True)
    rng, init_rng = jax.random.split(rng)
    params = encoder_with_coord.init(init_rng, test_image)
    output_with_coord = encoder_with_coord.apply(params, test_image, train=False)
    print(f"  输入形状: {test_image.shape}")
    print(f"  输出形状: {output_with_coord.shape}")
    print(f"  说明: 添加了空间坐标，输入通道从3变为5")
    
    # 测试带瓶颈层的编码器
    print("\\n  测试带瓶颈层的编码器")
    print("  " + "-" * 76)
    encoder_with_bottleneck = encoder_config(bottleneck_dim=256)
    rng, init_rng = jax.random.split(rng)
    params = encoder_with_bottleneck.init(init_rng, test_image)
    output_with_bottleneck = encoder_with_bottleneck.apply(params, test_image, train=False)
    print(f"  输入形状: {test_image.shape}")
    print(f"  输出形状: {output_with_bottleneck.shape}")
    print(f"  瓶颈维度: {output_with_bottleneck.shape[-1]}")
    print(f"  输出范围: [{output_with_bottleneck.min():.4f}, {output_with_bottleneck.max():.4f}]")
    
    print("\\n" + "=" * 80)
    print("所有模块测试完成！")
    print("=" * 80)


# 取消下面的注释以运行快速测试（只测试ResNetEncoder）
def test_resnet_encoder_quick():
    import jax
    
    print("=" * 80)
    print("快速测试 ResNet V1 编码器")
    print("=" * 80)
    
    # 测试参数
    batch_size = 4
    image_size = 128
    rng = jax.random.PRNGKey(42)
    
    # 创建测试图像
    print("\\n【步骤1】创建测试图像")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {image_size}x{image_size}")
    print(f"通道数: 3 (RGB)")
    test_image = jax.random.uniform(rng, (batch_size, image_size, image_size, 3))
    print(f"测试图像形状: {test_image.shape}")
    print(f"像素值范围: [{test_image.min():.4f}, {test_image.max():.4f}]")
    
    # 测试不同的ResNet配置
    configs_to_test = [
        ("resnetv1-10", "ResNet-10 (最小)"),
        ("resnetv1-18", "ResNet-18 (标准)"),
        ("resnetv1-34", "ResNet-34 (更深)"),
    ]
    
    for config_name, config_desc in configs_to_test:
        print(f"\\n【步骤{configs_to_test.index((config_name, config_desc)) + 2}】测试 {config_desc}")
        print("-" * 80)
        
        # 获取配置
        encoder_config = resnetv1_configs[config_name]
        encoder = encoder_config()
        
        # 初始化模块参数
        rng, init_rng = jax.random.split(rng)
        params = encoder.init(init_rng, test_image)
        
        # 前向传播
        encoder_output = encoder.apply(params, test_image, train=False)
        
        print(f"配置名称: {config_name}")
        print(f"输入形状: {test_image.shape}")
        print(f"输出形状: {encoder_output.shape}")
        print(f"输出维度: {encoder_output.shape[-1]}")
        print(f"输出均值: {encoder_output.mean():.6f}")
        print(f"输出标准差: {encoder_output.std():.6f}")
        print(f"输出范围: [{encoder_output.min():.4f}, {encoder_output.max():.4f}]")
    
    # 测试不同的池化方法
    print("\\n【步骤5】测试不同的池化方法")
    print("-" * 80)
    encoder_config = resnetv1_configs["resnetv1-18"]
    
    pooling_methods = {
        "avg": "平均池化",
        "max": "最大池化",
        "spatial_softmax": "空间softmax池化",
        "spatial_learned_embeddings": "空间学习嵌入池化",
    }
    
    for method, desc in pooling_methods.items():
        print(f"\\n  测试: {desc}")
        print("  " + "-" * 76)
        
        encoder = encoder_config(pooling_method=method)
        rng, init_rng = jax.random.split(rng)
        params = encoder.init(init_rng, test_image)
        output = encoder.apply(params, test_image, train=False)
        
        print(f"  池化方法: {method}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出维度: {output.shape[-1]}")
    
    print("\\n" + "=" * 80)
    print("快速测试完成！")
    print("=" * 80)


# 取消下面的注释以测试单个模块
def test_single_module():
    import jax
    
    print("=" * 80)
    print("测试单个模块")
    print("=" * 80)
    
    # 测试参数
    batch_size = 2
    height = 16
    width = 16
    channels = 64
    rng = jax.random.PRNGKey(42)
    
    # 创建测试输入
    test_input = jax.random.uniform(rng, (batch_size, height, width, channels))
    print(f"\\n【步骤1】创建测试输入")
    print(f"输入形状: {test_input.shape}")
    
    # 选择要测试的模块（取消注释以测试）
    
    # 测试 ResNetBlock
    # print("\\n【步骤2】测试 ResNetBlock")
    # print("-" * 80)
    # resnet_block = ResNetBlock(filters=128, conv=nn.Conv, norm=MyGroupNorm, act=nn.relu)
    # rng, init_rng = jax.random.split(rng)
    # params = resnet_block.init(init_rng, test_input)
    # output = resnet_block.apply(params, test_input)
    # print(f"输入形状: {test_input.shape}")
    # print(f"输出形状: {output.shape}")
    # print(f"输出通道数: {output.shape[-1]}")
    
    # 测试 BottleneckResNetBlock
    # print("\\n【步骤3】测试 BottleneckResNetBlock")
    # print("-" * 80)
    # bottleneck_block = BottleneckResNetBlock(filters=64, conv=nn.Conv, norm=MyGroupNorm, act=nn.relu)
    # rng, init_rng = jax.random.split(rng)
    # params = bottleneck_block.init(init_rng, test_input)
    # output = bottleneck_block.apply(params, test_input)
    # print(f"输入形状: {test_input.shape}")
    # print(f"输出形状: {output.shape}")
    # print(f"输出通道数: {output.shape[-1]}")
    
    # 测试 AddSpatialCoordinates
    # print("\\n【步骤4】测试 AddSpatialCoordinates")
    # print("-" * 80)
    # add_coord_module = AddSpatialCoordinates()
    # output = add_coord_module(test_input)
    # print(f"输入形状: {test_input.shape}")
    # print(f"输出形状: {output.shape}")
    # print(f"输出通道数: {output.shape[-1]}")
    
    # 测试 SpatialSoftmax
    # print("\\n【步骤5】测试 SpatialSoftmax")
    # print("-" * 80)
    # test_features = jax.random.uniform(rng, (batch_size, height, width, 8))
    # pos_x, pos_y = jnp.meshgrid(
    #     jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
    # )
    # pos_x = pos_x.reshape(height * width)
    # pos_y = pos_y.reshape(height * width)
    # spatial_softmax_module = SpatialSoftmax(height, width, 8, pos_x, pos_y, temperature=1.0)
    # output = spatial_softmax_module(test_features)
    # print(f"输入形状: {test_features.shape}")
    # print(f"输出形状: {output.shape}")
    
    # 测试 SpatialLearnedEmbeddings
    # print("\\n【步骤6】测试 SpatialLearnedEmbeddings")
    # print("-" * 80)
    # spatial_emb_module = SpatialLearnedEmbeddings(height=height, width=width, channel=8, num_features=5)
    # rng, init_rng = jax.random.split(rng)
    # params = spatial_emb_module.init(init_rng, test_features)
    # output = spatial_emb_module.apply(params, test_features)
    # print(f"输入形状: {test_features.shape}")
    # print(f"输出形状: {output.shape}")
    
    print("\\n" + "=" * 80)
    print("请取消注释要测试的模块，然后重新运行脚本")
    print("=" * 80)


# 取消下面的注释以运行测试
if __name__ == "__main__":
    # 选择要运行的测试（取消注释）
    test_all_modules()
    # test_resnet_encoder_quick()
    # test_single_module()