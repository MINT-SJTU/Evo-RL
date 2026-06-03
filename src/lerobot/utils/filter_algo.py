"""Brain Filter Algorthim."""

import numpy as np


class SimpleKalmanFilter:
    """
    简化的一维卡尔曼滤波器

    状态向量: x = [position]
    测量向量: z = [position]
    """

    def __init__(
        self,
        process_variance=0.01,
        measurement_variance=0.1,
        dims=6,
        initial_position=None,
    ):
        """
        初始化卡尔曼滤波器

        参数:
        process_variance: 过程噪声方差
        measurement_variance: 测量噪声方差
        initial_position: 初始位置估计
        """
        self.dims = dims
        # 状态转移矩阵 (假设位置基本不变或缓慢变化)
        self.F = [np.array([[1]]) for _ in range(dims)]

        # 观测矩阵
        self.H = [np.array([[1]]) for _ in range(dims)]

        # 过程噪声协方差
        self.Q = [np.array([[process_variance]]) for _ in range(dims)]

        # 测量噪声协方差
        self.R = [np.array([[measurement_variance]]) for _ in range(dims)]

        # 状态协方差
        self.P = [np.array([[1]]) for _ in range(dims)]

        # 状态向量 [位置]
        self.x = [np.array([[initial_position[i]]]) for i in range(dims)]

    def predict(self):
        """
        预测步骤
        """
        # 预测状态
        for i in range(self.dims):
            self.x[i] = self.F[i] @ self.x[i]

            # 预测协方差
            self.P[i] = self.F[i] @ self.P[i] @ self.F[i].T + self.Q[i]

        return self.x

    def update(self, measurement):
        """
        更新步骤

        参数:
        measurement: 测量值
        """
        for i in range(self.dims):
            # 计算卡尔曼增益
            S = self.H[i] @ self.P[i] @ self.H[i].T + self.R[i]
            K = self.P[i] @ self.H[i].T @ np.linalg.inv(S)

            # 更新状态估计
            z = np.array([[measurement[i]]])
            y = z - self.H[i] @ self.x[i]
            self.x[i] = self.x[i] + K @ y

            # 更新协方差估计
            I = np.eye(self.P[i].shape[0])  # noqa: E741
            self.P[i] = (I - K @ self.H[i]) @ self.P[i]

        return np.array([self.x[i][0, 0] for i in range(self.dims)])

    def get_position(self):
        """获取当前估计的位置"""
        return self.x[0, 0]

    def get_uncertainty(self):
        """获取当前位置的不确定性"""
        return self.P[0, 0]
