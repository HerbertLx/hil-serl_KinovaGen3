import copy
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gym import Env
from kinova_env.utils.transformations import (
    construct_transform_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.transform_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update transform matrix
        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # Update transform matrix
        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        将观测数据从基座坐标系（Base Frame）转换到参考/目标坐标系（Body/Relative Frame）。
        
        输入:
            obs (dict): 原始观测字典，包含:
                - obs["state"]["tcp_pose"]: [x, y, z, qx, qy, qz, qw] 基座下的末端位姿
                - obs["state"]["tcp_vel"]:  [vx, vy, vz] 基座下的末端线速度
        
        输出:
            obs (dict): 转换后的观测字典，主要更新了 tcp_vel 和 tcp_pose。
            
        运行逻辑:
            1. 速度变换：将末端速度从基座坐标系投影到末端执行器当前的方向上（Body-frame velocity）。
            2. 位姿变换：如果开启了 relative_pose，则计算末端相对于“某个参考点（目标）”的位姿，
            而不是相对于基座。这使得策略能够直接感知“离目标还有多远”。
        """
        # 1. 速度变换：将全局速度向量转为局部速度向量
        # 这样机器人学到的是“向前冲”而不是“向地图的北边冲”
        # print(f"\nIn RelativeFrame, before, \nobs['state']['tcp_vel'] = {obs['state']['tcp_vel']}")
        # print(f"obs['state']['tcp_pose'] = {obs['state']['tcp_pose']}")
        # print(f"self.include_relative_pose = {self.include_relative_pose}")

        transform_inv = np.linalg.inv(self.transform_matrix)
        obs["state"]["tcp_vel"] = transform_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            # 2. 构建当前末端相对于基座的变换矩阵 T_base_to_object
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            
            # 3. 计算末端相对于“目标”的变换矩阵 T_target_to_object
            # 计算公式：T_relative = T_base_to_target_inv * T_base_to_object
            T_b_r = self.T_r_o_inv @ T_b_o

            # 4. 重新打包成 [位置, 四元数] 格式
            p_b_r = T_b_r[:3, 3] # 这就是“末端距离目标点的 XYZ 偏移”
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))
        # print(f"\nIn RelativeFrame, after, \nobs['state']['tcp_vel'] = {obs['state']['tcp_vel']}")
        # print(f"obs['state']['tcp_pose'] = {obs['state']['tcp_pose']}")

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the transform matrix. 
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.transform_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the transform matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.transform_matrix) @ action[:6]
        return action


class DualRelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
            }
        ),
        ......
    }, and at least 12 DoF action space
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.left_transform_matrix = np.zeros((6, 6))
        self.right_transform_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.zeros((4, 4))
            self.right_T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update transform matrix
        self.left_transform_matrix = construct_transform_matrix(obs["state"]["left/tcp_pose"])
        self.right_transform_matrix = construct_transform_matrix(obs["state"]["right/tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update transform matrix
        self.left_transform_matrix = construct_transform_matrix(obs["state"]["left/tcp_pose"])
        self.right_transform_matrix = construct_transform_matrix(obs["state"]["right/tcp_pose"])

        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            )
            self.right_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            )
        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the transform matrix
        """
        left_transform_inv = np.linalg.inv(self.left_transform_matrix)
        obs["state"]["left/tcp_vel"] = left_transform_inv @ obs["state"]["left/tcp_vel"]

        right_transform_inv = np.linalg.inv(self.right_transform_matrix)
        obs["state"]["right/tcp_vel"] = right_transform_inv @ obs["state"]["right/tcp_vel"]

        if self.include_relative_pose:
            left_T_b_o = construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            left_T_b_r = self.left_T_r_o_inv @ left_T_b_o

            # Reconstruct transformed tcp_pose vector
            left_p_b_r = left_T_b_r[:3, 3]
            left_theta_b_r = R.from_matrix(left_T_b_r[:3, :3]).as_quat()
            obs["state"]["left/tcp_pose"] = np.concatenate((left_p_b_r, left_theta_b_r))

            right_T_b_o = construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            right_T_b_r = self.right_T_r_o_inv @ right_T_b_o

            # Reconstruct transformed tcp_pose vector
            right_p_b_r = right_T_b_r[:3, 3]
            right_theta_b_r = R.from_matrix(right_T_b_r[:3, :3]).as_quat()
            obs["state"]["right/tcp_pose"] = np.concatenate((right_p_b_r, right_theta_b_r))


        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the transform matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        if len(action) == 12:
            action[:6] = self.left_transform_matrix @ action[:6]
            action[6:] = self.right_transform_matrix @ action[6:]
        elif len(action) == 14:
            action[:6] = self.left_transform_matrix @ action[:6]
            action[7:13] = self.right_transform_matrix @ action[7:13]
        else:
            raise ValueError("Action dimension not supported")
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the transform matrix.
        """
        action = np.array(action)
        if len(action) == 12:
            action[:6] = np.linalg.inv(self.left_transform_matrix) @ action[:6]
            action[6:] = np.linalg.inv(self.right_transform_matrix) @ action[6:]
        elif len(action) == 14:
            action[:6] = np.linalg.inv(self.left_transform_matrix) @ action[:6]
            action[7:13] = np.linalg.inv(self.right_transform_matrix) @ action[7:13]
        else:
            raise ValueError("Action dimension not supported")
        return action