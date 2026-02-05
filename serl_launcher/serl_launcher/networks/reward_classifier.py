import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from typing import Callable, Dict, List
import requests
import os
from tqdm import tqdm

from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper

# ==========================================
# 类定义：BinaryClassifier (二分类器)
# ==========================================
class BinaryClassifier(nn.Module):
    """
    逻辑：
    1. 输入图像经过 encoder_def 提取特征向量。
    2. 通过一个隐藏层进行非线性变换。
    3. 最后输出一个标量 Logit，用于 Sigmoid 二分类。
    
    输入: x (Dict 或 Array), 包含图像数据的观测。
    输出: 形状为 (Batch, 1) 的 Logits 数组。
    """
    encoder_def: nn.Module   # 特征提取器（通常是 ResNet）
    hidden_dim: int = 256    # 隐藏层神经元数量

    @nn.compact
    def __call__(self, x, train=False):
        # 提取特征
        x = self.encoder_def(x, train=train)
        # 全连接层
        x = nn.Dense(self.hidden_dim)(x)
        # 训练时使用 Dropout 防止过拟合
        x = nn.Dropout(0.1)(x, deterministic=not train)
        # 层归一化
        x = nn.LayerNorm()(x)
        # 激活函数
        x = nn.relu(x)
        # 输出层：输出 1 维 (用于判别 0 或 1)
        x = nn.Dense(1)(x)
        return x

# ==========================================
# 类定义：NWayClassifier (多分类器)
# ==========================================
class NWayClassifier(nn.Module):
    """
    逻辑：
    与 BinaryClassifier 类似，但输出层维度为 n_way，用于多任务分类或多状态识别。
    
    输入: x (Dict 或 Array)
    输出: 形状为 (Batch, n_way) 的 Logits 数组。
    """
    encoder_def: nn.Module
    hidden_dim: int = 256
    n_way: int = 3

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_way)(x)
        return x

# ==========================================
# 函数定义：create_classifier
# ==========================================
def create_classifier(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    n_way: int = 2,
):
    """
    执行逻辑：
    1. 定义 ResNet-10 架构。
    2. 为每一个摄像头(image_keys)创建一个独立的编码器，并用 EncodingWrapper 封装。
    3. 初始化分类器参数。
    4. 检查并从 GitHub 下载在 ImageNet 上预训练好的 ResNet-10 权重。
    5. 将下载的预训练权重手动覆盖（替换）到刚刚初始化的新参数中，实现迁移学习。

    参数:
        key: JAX 随机种子。
        sample: 样例输入数据，用于确定网络输入维度。
        image_keys: 图像键名列表，例如 ['wrist_1', 'side_1']。
        n_way: 分类类别数，2 代表二分类（奖励函数常用）。
    
    返回:
        classifier: 一个包含参数和优化器的 TrainState 对象。
    """
    # 1. 定义底层冻结的 ResNet 结构
    pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,
        name="pretrained_encoder",
    )
    
    # 2. 为每个相机分配编码器，并使用“空间学习嵌入”进行池化
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
    
    # 3. 封装所有编码器
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,    # 不使用本体感知数据
        enable_stacking=True, # 允许图像堆叠
        image_keys=image_keys,
    )
    
    # 4. 选择分类器类型
    if n_way == 2:
        classifier_def = BinaryClassifier(encoder_def=encoder_def)
    else:
        classifier_def = NWayClassifier(encoder_def=encoder_def, n_way=n_way)
        
    # 5. 初始化参数和 TrainState
    params = classifier_def.init(key, sample)["params"]
    classifier = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-4),
    )

    # 6. 权重下载逻辑
    file_name = "resnet10_params.pkl"
    file_path = os.path.expanduser("~/.serl/")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, file_name)

    if os.path.exists(file_path):
        print(f"The ResNet-10 weights already exist at '{file_path}'.")
    else:
        # 如果本地没有，则从 Berkeley 的服务器下载
        url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
        print(f"Downloading file from {url}")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(file_path, "wb") as f:
                for data in response.iter_content(1024):
                    t.update(len(data))
                    f.write(data)
            t.close()
        except Exception as e:
            raise RuntimeError(e)
        print("Download complete!")

    # 7. 权重加载与替换 (Transfer Learning)
    with open(file_path, "rb") as f:
        encoder_params = pkl.load(f)
            
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(encoder_params))
    print(f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K")
    
    # 遍历新初始化的参数，将其中属于 ResNet 层的部分替换为预训练好的参数
    new_params = classifier.params
    for image_key in image_keys:
        encoder_name = f"encoder_{image_key}"
        if "pretrained_encoder" in new_params["encoder_def"][encoder_name]:
            for k in new_params["encoder_def"][encoder_name]["pretrained_encoder"]:
                if k in encoder_params:
                    # 替换核心权重
                    new_params["encoder_def"][encoder_name]["pretrained_encoder"][k] = encoder_params[k]
                    print(f"replaced {k} in {encoder_name}")

    # 返回替换权重后的新分类器
    classifier = classifier.replace(params=new_params)
    return classifier

# ==========================================
# 函数定义：load_classifier_func
# ==========================================
def load_classifier_func(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    n_way: int = 2,
) -> Callable[[Dict], jnp.ndarray]:
    """
    执行逻辑：
    1. 调用 create_classifier 创建并初始化模型结构（加载预训练基础权重）。
    2. 使用 checkpoints.restore_checkpoint 加载用户训练好的特定任务权重。
    3. 定义一个简洁的推理函数，并使用 jax.jit 进行编译加速。

    参数:
        checkpoint_path: 你训练好的 classifier_ckpt 文件夹路径。
    
    返回:
        func: 一个可以接受 obs (观测字典) 并返回 Logits 的 JIT 编译函数。
    """
    # 创建模型
    from datetime import datetime
    import time
    classifier = create_classifier(key, sample, image_keys, n_way=n_way)
    # 从本地 checkpoint 恢复权重
    classifier = checkpoints.restore_checkpoint(
        checkpoint_path,
        target=classifier,
    )
    # 定义纯推理闭包
    def func(obs):
        return classifier.apply_fn(
            {"params": classifier.params}, obs, train=False
        )
    # JIT 编译提高预测效率
    return jax.jit(func)


'''
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from typing import Callable, Dict, List
import requests
import os
from tqdm import tqdm

from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class NWayClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256
    n_way: int = 3

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_way)(x)
        return x


def create_classifier(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    n_way: int = 2,
):
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
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )
    if n_way == 2:
        classifier_def = BinaryClassifier(encoder_def=encoder_def)
    else:
        classifier_def = NWayClassifier(encoder_def=encoder_def, n_way=n_way)
    params = classifier_def.init(key, sample)["params"]
    classifier = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-4),
    )

    file_name = "resnet10_params.pkl"
    # Construct the full path to the file
    file_path = os.path.expanduser("~/.serl/")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, file_name)
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"The ResNet-10 weights already exist at '{file_path}'.")
    else:
        url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
        print(f"Downloading file from {url}")

        # Streaming download with progress bar
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(file_path, "wb") as f:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise Exception("Error, something went wrong with the download")
        except Exception as e:
            raise RuntimeError(e)
        print("Download complete!")

    with open(file_path, "rb") as f:
        encoder_params = pkl.load(f)
            
    # param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )
    new_params = classifier.params
    for image_key in image_keys:
        if "pretrained_encoder" in new_params["encoder_def"][f"encoder_{image_key}"]:
            for k in new_params["encoder_def"][f"encoder_{image_key}"][
                "pretrained_encoder"
            ]:
                if k in encoder_params:
                    new_params["encoder_def"][f"encoder_{image_key}"][
                        "pretrained_encoder"
                    ][k] = encoder_params[k]
                    print(f"replaced {k} in encoder_{image_key}")

    classifier = classifier.replace(params=new_params)
    return classifier

def load_classifier_func(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    n_way: int = 2,
) -> Callable[[Dict], jnp.ndarray]:
    """
    Return: a function that takes in an observation
            and returns the logits of the classifier.
    """
    classifier = create_classifier(key, sample, image_keys, n_way=n_way)
    classifier = checkpoints.restore_checkpoint(
        checkpoint_path,
        target=classifier,
    )
    func = lambda obs: classifier.apply_fn(
        {"params": classifier.params}, obs, train=False
    )
    func = jax.jit(func)
    return func
'''