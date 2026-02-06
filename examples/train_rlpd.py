#!/usr/bin/env python3

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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
target_path = "/home/cuhk/Documents/visionpro-kinova-rl/hil-serl_KinovaGen3"
if os.path.exists(target_path) and target_path not in sys.path:
    sys.path.insert(0, target_path)

# 添加机器人控制 API 路径
manager_path = "/home/cuhk/Documents/visionpro-kinova-rl/robot_control/api_control"
if os.path.exists(manager_path) and manager_path not in sys.path:
    sys.path.insert(0, manager_path)

# 设置环境变量：解决 Protobuf 版本冲突，强制使用 Python 纯脚本实现
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


import glob
import time
import jax
import jax.numpy as jnp
# import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
# from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted
from datetime import datetime

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

# 定期发送运行耗时统计（Timer）给 Learner 记录
import threading

def async_send_stats(client, stats):
    def _send():
        try:
            # 这里的 request 依然会阻塞，但它是在子线程里，不会卡住主控制循环
            client.request("send-stats", stats)
        except Exception:
            pass 
    threading.Thread(target=_send, daemon=True).start()

def format_timestamp(ts):
    # 使用 datetime.fromtimestamp 将浮点数转换为本地日期时间对象
    dt = datetime.fromtimestamp(ts)
    # %H:%M:%S 对应时分秒，%f 对应微秒(取前3位即为毫秒)
    return dt.strftime('%H:%M:%S.%f')[:-3]
def generate_time(start_time, end_time):
    """
    将Unix时间戳转换为人类可读的时间字符串，并计算时间差。
    格式: HH:MM:SS.mmm
    """
    # 1. 格式化开始和结束时间
    formatted_start_time = format_timestamp(start_time)
    formatted_end_time = format_timestamp(end_time)

    # 2. 计算时间差 (Gap Time)
    gap_seconds = end_time - start_time
    
    # 手动计算差值的时、分、秒、毫秒
    m, s = divmod(gap_seconds, 60)
    h, m = divmod(m, 60)
    ms = int((gap_seconds - int(gap_seconds)) * 1000)
    
    formatted_gap_time = f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

    return formatted_start_time, formatted_end_time, formatted_gap_time


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)
# sharding = jax.sharding.NamedSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    【Actor 主循环函数】
    
    作用：
    1. 负责控制机器人执行动作（采样）。
    2. 处理人类干预：当人类接管机器人时，记录干预动作作为“专家示例”。
    3. 数据同步：将采集到的交互数据（Transition）发送给 Learner，并定期从 Learner 接收最新的模型权重。
    4. 评估模式：如果指定了检查点步数，则进入纯测试模式，不进行数据采集和训练。

    输入参数：
        - agent: SACAgent 对象，包含当前的策略网络和参数。
        - data_store: 普通经验池（QueuedDataStore），存储机器人自主探索的数据。
        - intvn_data_store: 干预经验池（QueuedDataStore），专门存储人类干预时的数据（作为 Demo）。
        - env: 机器人 Gymnasium 环境。
        - sampling_rng: JAX 随机数种子，用于动作采样。

    输出：
        - 该函数为死循环或长时循环，无直接返回值。
    """

    # ==========================================
    # 第一部分：模型评估逻辑 (Evaluation Mode)
    # 当启动脚本带有 --eval_checkpoint_step 参数时执行
    # ==========================================

    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        # 从指定路径恢复特定步数的模型权重
        print(f"\nFLAGS.checkpoint_path = {FLAGS.checkpoint_path}")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        action_warmup = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        env.reset()
        time_before_warmup = time.time()
        env.step(action_warmup)  # 预热一步，确保环境正常
        time_after_warmup = time.time()
        _, _, warm_up_duration = generate_time(time_before_warmup, time_after_warmup)
        print(f"Warm-up step duration: {warm_up_duration}")
        env.reset()

        # 循环执行测试轨迹
        for episode in range(FLAGS.eval_n_trajs):
            print(f"\n=== Evaluation Episode {episode + 1} ===")
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                # 采样动作，argmax=False 表示依然带有一定的随机性（也可以设为 True 进行纯确定性评估）
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                if done:
                    if reward: # 如果奖励为 True (1)，记录完成时间
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")
        env.reset()
        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # 评估完成，直接退出整个函数

    # ==========================================
    # 第二部分：训练初始化逻辑 (Training Initialization)
    # ==========================================
    
    # 自动计算断点续训的起始步数：读取 buffer 文件夹下最新的文件名，解析出步数数字
    # [12:-4] 是根据 "transitions_1000.pkl" 这种命名规则切片提取数字
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    # 定义数据存储字典，用于 client 端与 Learner 通信
    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }


    # 初始化分布式客户端，连接 Learner (Server)
    print(f"\nForming connection to learner at {FLAGS.ip}...")
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=100,
        # timeout_ms=3000,
    )
    print(f"\nConnected to learner at {FLAGS.ip}.")

    # 回调函数：当 Learner 发布新权重时，Actor 会调用此函数更新本地的 agent
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # 用于保存到本地磁盘的临时数据列表
    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # 状态计数器
    timer = Timer()
    running_return = 0.0          # 当前回合的累积奖励
    already_intervened = False    # 标记当前步是否处于被干预状态
    intervention_count = 0        # 当前回合发生了多少次干预切换
    intervention_steps = 0        # 当前回合总共干预了多少步

    # ==========================================
    # 第三部分：正式采集循环 (Main Interaction Loop)
    # ==========================================
    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        # --- 动作采样阶段 ---
        with timer.context("sample_actions"):
            # 如果在初始随机步数内，直接从空间中随机采样（用于填充 buffer）
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                # 正常使用策略网络采样
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # --- 环境交互阶段 ---
        with timer.context("step_env"):

            # 执行动作
            from datetime import datetime
            from scipy.spatial.transform import Rotation as R
            from pyquaternion import Quaternion


            def quat_2_euler(quat):
                """calculates and returns: yaw, pitch, roll from given quaternion"""
                return R.from_quat(quat).as_euler("xyz")

            # # --- 在你的训练/测试循环中 ---

            # # 1. 获取当前精确时间 (包含毫秒)
            # curr_time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 只取前3位毫秒

            # # 2. 格式化数据


            # # 3. 设置进度条描述 (将时间放在最前面，方便观察对齐)
            # pbar.set_description(
            #     f"[{curr_time_str}] | ACT: {printed_action.tolist()} | POS: {printed_pos.tolist()}"
            # )

            next_obs, reward, done, truncated, info = env.step(actions)


            # 清理 info 中的冗余信息（可能是针对双臂设置的清理）
            if "left" in info: info.pop("left")
            if "right" in info: info.pop("right")

            # 【核心逻辑：处理人为干预】
            # 如果人类触碰了遥操作设备，env 内部会捕捉到并返回 "intervene_action"
            if "intervene_action" in info:
                # 使用人类的动作覆盖 AI 的动作
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1 # 统计干预发生的次数
                already_intervened = True
            else:
                already_intervened = False # 当前步是 AI 自动执行的

            running_return += reward
            
            # 构造转换数据 (Transition)
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            
            # 针对特定任务的惩罚项逻辑
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            
            # 将数据存入普通经验池
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            
            # 如果这一步是干预发生的，则额外存入“干预经验池”（即 Demo 数据）
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            # --- 回合结束处理 ---
            if done or truncated:

                # 将本回合的干预统计信息发送给 Learner 进行日志记录
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}

                pbar.set_description(f"last return: {running_return}")
                obs, _ = env.reset()

                original_timeout = client.req_rep_client.timeout_ms
                client.req_rep_client.timeout_ms = 5000  # 延长超时时间，确保 stats 能发送成功

                client_request_start_time = time.time()
                client.request("send-stats", stats)
                client_request_end_time = time.time()
                _, _, gap_time = generate_time(client_request_start_time, client_request_end_time)
                print(f" Client request sent with time {gap_time}.")

                
                # 重置回合状态
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                
                # 触发通信更新，同步数据到服务器
                client_update_start_time = time.time()
                client.update()
                client_update_end_time = time.time()
                _, _, gap_time = generate_time(client_update_start_time, client_update_end_time)
                print(f" Client updated with time {gap_time}.")

                client.req_rep_client.timeout_ms = original_timeout


        # --- 定期保存数据到磁盘 (Checkpointing Buffer) ---

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            
            if not os.path.exists(buffer_path): os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path): os.makedirs(demo_buffer_path)
            
            # 保存普通数据
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = [] # 清空列表，节省内存
                
            # 保存专家干预数据
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []
        timer.tock("total")


        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            print(f"\nClient sending stats2 with timeout {client.req_rep_client.timeout_ms}...")
            client_request_start_time = time.time()
            client.request("send-stats", stats)
            client_request_end_time = time.time()
            _, _, gap_time = generate_time(client_request_start_time, client_request_end_time)
            print(f" Client request sent with time {gap_time}.\n")
    env.reset()

##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    【Learner 训练主循环函数】

    作用：
    1. 启动数据服务器 (TrainerServer)，接收来自 Actor 的数据并存入 Buffer。
    2. 实施 RLPD (Reinforcement Learning from Prior Data) 策略，混合采样专家演示和在线数据。
    3. 执行神经网络梯度更新（更新 Critic、Actor 和温度系数参数）。
    4. 定期将更新后的网络参数发布给 Actor，并保存模型检查点。

    输入参数：
        - rng: JAX 随机数种子。
        - agent: 强化学习 Agent 对象（SAC 或其变体）。
        - replay_buffer: 在线交互经验回放池。
        - demo_buffer: 专家演示/人为干预经验回放池。
        - wandb_logger: Weights & Biases 日志记录器，用于监控训练曲线。

    输出：
        - 该函数为训练长循环，执行直至达到最大步数。
    """
    # --- 1. 断点续训逻辑 ---
    # 获取最新检查点的步数，例如 "checkpoint_1000" 会解析出 1000
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    # --- 2. 定义回调函数 ---
    def stats_callback(type: str, payload: dict) -> dict:
        """当服务器接收到来自 Actor 的统计数据（如奖励、干预次数）时调用此函数"""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            # 将 Actor 传回的 info 信息记录到 wandb
            wandb_logger.log(payload, step=step)
        return {}  # 无需给 Actor 返回数据

    # --- 3. 启动数据服务器 ---
    # TrainerServer 负责在后台线程中与 Actor 通信，自动把 Actor 发来的数据塞进指定的 buffer
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # --- 4. 等待数据填充 ---
    # 训练开始前，必须保证 replay_buffer 中已经有足够的随机采样数据
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1) # 每秒检查一次
    pbar.close()

    # 将初始（或恢复的）网络参数发布给 Actor，确保双方步调一致
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # --- 5. 设置采样迭代器 (RLPD 核心) ---
    # 从在线池采样一半，从演示池采样一半 (50/50 sampling)
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True, # 将 obs 和 next_obs 打包提高传输效率
        },
        device=sharding.replicate(), # 将数据复制到所有 GPU 设备上
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    timer = Timer()
    
    # --- 6. 确定更新策略 ---
    # 根据 Agent 类型决定需要更新哪些子网络
    # SACAgent 通常包含 Actor, Critic, Temperature
    # HybridAgent 则多了一个 grasp_critic 用于处理夹爪逻辑
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    # --- 7. 正式训练循环 ---
    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # 【优化技巧：CTA Ratio】
        # 为了提高吞吐量，通常运行多次 Critic 更新才进行一次 Actor 更新 (Critic-to-Actor ratio)
        # 前 n-1 次只更新 Critic
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                # 混合两类数据形成一个完整的 Batch
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                # 仅执行 Critic 的梯度更新
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        # 第 n 次更新：同时更新 Critic 和 Actor
        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        # --- 8. 同步与日志 ---
        # 定期将最新的模型参数发布给 Actor 端
        if step > 0 and step % (config.steps_per_update) == 0:
            # block_until_ready 确保异步计算完成，防止参数还没算完就发布了
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # 记录训练数据到 WandB
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)


        # print(f"\nStep = {step}, config.checkpoint_period = {config.checkpoint_period}\n")
        # 定期保存检查点到磁盘
        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            print(f"\nSaving checkpoint at step {step}...\n")
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    """
    【Main 总调度函数】

    作用：
    1. 配置加载：根据命令行参数加载对应的任务配置（如实验名、种子等）。
    2. 环境初始化：创建机器人 Gym 环境，并根据角色（Learner/Actor）决定是否启用虚拟环境。
    3. 智能体实例化：根据 setup_mode（单/双臂、是否学习夹爪）创建对应的 SAC Agent。
    4. 权重管理：处理模型断点续训（Checkpoint）的恢复。
    5. 数据管理：初始化经验回放池（Replay Buffer），并加载离线 Demo 或历史 Buffer。
    6. 任务分发：根据参数进入 learner() 逻辑或 actor() 逻辑。

    输入参数：
        - _: 接收来自 absl.app.run 的剩余参数（此处未使用）。
    """
    global config
    # 根据实验名称从映射表中获取具体的配置类并实例化
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    print(f"config.shape = {config.shape}")
    exit()

    # 确保 Batch Size 能够被当前可用的计算设备（GPU/TPU）整除，以便并行计算
    assert config.batch_size % num_devices == 0
    
    # 初始化 JAX 随机数种子，并分裂出用于采样的种子
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."

    # --- 1. 创建环境 ---
    # fake_env=FLAGS.learner: 如果是训练端，通常不需要开启真实的机器人连接/图形界面，使用虚拟环境
    # classifier=True: 开启基于视觉的分类器，用于自动判断任务成功并给出奖励
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    # 包装环境以自动统计每个 Episode 的奖励、长度等信息
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    # --- 2. 根据机器人配置创建 Agent (智能体) ---
    # 逻辑猜测：setup_mode 决定了网络结构。'learned-gripper' 表示夹爪动作是策略的一部分，
    # 需要额外的 'grasp_penalty' 损失来优化夹爪的使用。
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False # 固定夹爪无需抓取惩罚
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # --- 3. 硬件加速配置 ---
    # 将 agent 数据结构放入计算设备（GPU/TPU）。
    # sharding.replicate() 意味着在多卡环境下，每个卡都持有一份完整的模型拷贝
    agent = jax.device_put(
        # jax.tree_map(jnp.array, agent), sharding.replicate()
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )


    # --- 4. 加载模型检查点 (Checkpoint) ---
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        # 交互式提示：防止误操作覆盖已有的训练进度
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        # 从路径名中解析出恢复的具体步数并打印
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    # 内部辅助函数：用于快速创建缓冲区和日志器
    def create_replay_buffer_and_wandb_logger():
        # MemoryEfficient 表示使用了内存优化技术（如存储 JPEG 编码图像而非原始数组）
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # 初始化 WandB 实验追踪
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    # ==========================================
    # 分支一：启动 Learner (训练服务器)
    # ==========================================
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        
        # 为演示数据（Demo/Intervention）创建专门的 Buffer
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        # 【核心步骤：加载离线演示数据】
        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    # 兼容性处理：提取 grasp_penalty 字段
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # 【可选步骤：恢复上次运行的 Buffer 状态】
        # 如果训练中断，不仅恢复模型权重，还恢复当时已经采集到的数据，保证训练连续性
        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        # 进入训练主循环
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    # ==========================================
    # 分支二：启动 Actor (交互客户端)
    # ==========================================
    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        
        # QueuedDataStore 是一个先进先出的队列，作为 Actor 和 Learner 之间的缓冲区
        data_store = QueuedDataStore(50000) 
        intvn_data_store = QueuedDataStore(50000)

        # 进入交互主循环
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
