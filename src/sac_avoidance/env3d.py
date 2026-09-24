"""3D point-mass dynamics, moving spheres and 32 Fibonacci ray sensors.

Extracted from the original shared environment; no ADP networks are constructed.
"""
import numpy as np

NUM_SENSORS = 32                # 3D 传感器方向数（Fibonacci 球面分布）
SENSOR_MAX_RANGE = 5.0          # 传感器最大探测距离
NUM_DYNAMIC_OBS = 3             # 动态障碍物数量
DYNAMIC_OBS_SPEED_LO = 0.25     # 动态障碍物最低速度
DYNAMIC_OBS_SPEED_HI = 0.55     # 动态障碍物最高速度
ENV_XMIN, ENV_XMAX = -1.0, 9.0
ENV_YMIN, ENV_YMAX = -1.0, 9.0
ENV_ZMIN, ENV_ZMAX = -1.0, 9.0


# =========================================================================
# 辅助函数：Fibonacci 球面均匀分布方向
# =========================================================================
def generate_fibonacci_sphere_directions(n):
    """在单位球面上生成 n 个近似均匀分布的方向向量。"""
    points = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # 黄金角
    for i in range(n):
        y = 1.0 - (i / (n - 1)) * 2.0    # y 从 1 → -1
        radius_at_y = np.sqrt(1.0 - y * y)
        theta = phi * i
        points[i, 0] = np.cos(theta) * radius_at_y
        points[i, 1] = y
        points[i, 2] = np.sin(theta) * radius_at_y
    return points


# =========================================================================
# 辅助函数：3D 球面参数化（用于可视化）
# =========================================================================
def sphere_surface(center, radius, resolution=15):
    """返回球面的 X, Y, Z 网格，用于 plot_surface / plot_wireframe。"""
    u = np.linspace(0, 2 * np.pi, resolution * 2)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


# =========================================================================
# 球体障碍物（静态，3D）
# =========================================================================
class SphereObstacle:
    """三维球体障碍物，由球心坐标和半径定义。"""

    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)


# =========================================================================
# 动态障碍物（3D）
# =========================================================================
class DynamicObstacle3D:
    """具有恒定速度和六面边界反弹行为的 3D 动态球体障碍物。"""

    def __init__(self, center, radius, velocity):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.velocity = np.array(velocity, dtype=float)
        self.trajectory = [self.center.copy()]

    def step(self, dt):
        """移动一步并在六面边界处反弹。"""
        self.center += self.velocity * dt
        r = self.radius
        if self.center[0] - r < ENV_XMIN:
            self.center[0] = ENV_XMIN + r
            self.velocity[0] *= -1.0
        elif self.center[0] + r > ENV_XMAX:
            self.center[0] = ENV_XMAX - r
            self.velocity[0] *= -1.0
        if self.center[1] - r < ENV_YMIN:
            self.center[1] = ENV_YMIN + r
            self.velocity[1] *= -1.0
        elif self.center[1] + r > ENV_YMAX:
            self.center[1] = ENV_YMAX - r
            self.velocity[1] *= -1.0
        if self.center[2] - r < ENV_ZMIN:
            self.center[2] = ENV_ZMIN + r
            self.velocity[2] *= -1.0
        elif self.center[2] + r > ENV_ZMAX:
            self.center[2] = ENV_ZMAX - r
            self.velocity[2] *= -1.0
        self.trajectory.append(self.center.copy())


# =========================================================================
# 三维避障环境
# =========================================================================

class AvoidanceEnv3D:
    """41 observations; bounded 3-axis acceleration at a 0.05 s period."""

    def __init__(self, start=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 goal=(7.0, 7.0, 7.0),
                 static_obstacles=None, dt=0.05, gamma=0.99, max_acc=1.2,
                 safe_margin=0.65, seed=0):
        self.rng = np.random.default_rng(seed)
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.dt = dt
        self.gamma = gamma
        self.max_acc = max_acc
        self.safe_margin = safe_margin
        self.max_speed = 2.0

        # 归一化尺度
        self.state_scale = max(np.linalg.norm(self.goal - self.start[:3]), 1.0)

        # 静态障碍物（3D 球体）
        if static_obstacles is None:
            static_obstacles = [
                SphereObstacle(center=(2.5, 3.5, 2.0), radius=1.0),
                SphereObstacle(center=(5.0, 5.5, 4.5), radius=1.0),
                SphereObstacle(center=(6.0, 6.5, 6.0), radius=0.8),
            ]
        self.static_obstacles = static_obstacles

        # 动态障碍物（3D）
        self.dynamic_obstacles = []
        self._init_dynamic_obstacles()

        # ---- 32 方向 Fibonacci 球面传感器 ----
        self.sensor_dirs = generate_fibonacci_sphere_directions(NUM_SENSORS)

        # 状态维度：3(位置) + 3(速度) + 3(目标) + 32(传感器) = 41
        self.state_dim = 3 + 3 + 3 + NUM_SENSORS


    def _init_dynamic_obstacles(self):
        """在 3D 空间随机位置和随机速度方向生成动态障碍物。"""
        self.dynamic_obstacles = []
        for _ in range(NUM_DYNAMIC_OBS):
            for _attempt in range(200):
                cx = self.rng.uniform(ENV_XMIN + 1.5, ENV_XMAX - 1.5)
                cy = self.rng.uniform(ENV_YMIN + 1.5, ENV_YMAX - 1.5)
                cz = self.rng.uniform(ENV_ZMIN + 1.5, ENV_ZMAX - 1.5)
                center = np.array([cx, cy, cz])
                radius = self.rng.uniform(0.35, 0.65)

                # 不与静态障碍物重叠
                overlap = False
                for obs in self.static_obstacles:
                    if np.linalg.norm(center - obs.center) < radius + obs.radius + 0.5:
                        overlap = True
                        break
                # 不与起点 / 目标太近
                if np.linalg.norm(center - self.start[:3]) < 1.8:
                    overlap = True
                if np.linalg.norm(center - self.goal) < 1.8:
                    overlap = True
                # 不与其他动态障碍物重叠
                for dob in self.dynamic_obstacles:
                    if np.linalg.norm(center - dob.center) < radius + dob.radius + 0.3:
                        overlap = True
                        break

                if not overlap:
                    speed = self.rng.uniform(DYNAMIC_OBS_SPEED_LO, DYNAMIC_OBS_SPEED_HI)
                    # 随机 3D 方向
                    v3d = self.rng.normal(size=3)
                    v3d /= np.linalg.norm(v3d)
                    velocity = v3d * speed
                    dob = DynamicObstacle3D(center, radius, velocity)
                    self.dynamic_obstacles.append(dob)
                    break

    def _reset_dynamic_obstacles(self):
        self._init_dynamic_obstacles()

    def _randomize_static_obstacles(self, num_static=3):
        """在 3D 空间中随机生成静态球体障碍物。"""
        new_obs = []
        for _ in range(num_static):
            for _ in range(200):
                cx = self.rng.uniform(ENV_XMIN + 1.0, ENV_XMAX - 1.0)
                cy = self.rng.uniform(ENV_YMIN + 1.0, ENV_YMAX - 1.0)
                cz = self.rng.uniform(ENV_ZMIN + 1.0, ENV_ZMAX - 1.0)
                center = np.array([cx, cy, cz])
                radius = self.rng.uniform(0.5, 1.2)
                if np.linalg.norm(center - self.start[:3]) < 1.8:
                    continue
                if np.linalg.norm(center - self.goal) < 1.8:
                    continue
                overlap = False
                for o in new_obs:
                    if np.linalg.norm(center - o.center) < radius + o.radius + 0.3:
                        overlap = True
                        break
                if not overlap:
                    new_obs.append(SphereObstacle(center, radius))
                    break
        self.static_obstacles = new_obs

    def _all_obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def _sensor_readings(self, agent_pos):
        """
        计算 32 个 Fibonacci 球面方向上的最近障碍物距离。

        对每个方向发射 3D 射线，通过射线-球体相交解析解计算
        每个方向上最近的障碍物表面距离。
        """
        readings = np.full(NUM_SENSORS, SENSOR_MAX_RANGE, dtype=float)
        all_obs = self._all_obstacles()

        for i, d in enumerate(self.sensor_dirs):
            for obs in all_obs:
                v = agent_pos - obs.center          # 3D 向量
                b_half = np.dot(v, d)               # v·D
                c_val = np.dot(v, v) - obs.radius ** 2
                disc = b_half ** 2 - c_val

                if disc < 0:
                    continue

                sqrt_disc = np.sqrt(disc)
                t1 = -b_half - sqrt_disc
                t2 = -b_half + sqrt_disc

                if t1 > 1e-6:
                    t = t1
                elif t2 > 1e-6:
                    t = t2
                else:
                    continue

                if t < readings[i]:
                    readings[i] = t

        return np.clip(readings, 0.0, SENSOR_MAX_RANGE)

    def _state_features(self, agent_state):
        """
        构建 41 维归一化状态特征向量。

        agent_state: [x, y, z, vx, vy, vz]
        返回:
          [0:3]   (p - goal) / state_scale   — 相对目标位置
          [3:6]   v / max_speed               — 归一化速度
          [6:9]   goal / state_scale          — 目标位置
          [9:41]  sensors / SENSOR_MAX_RANGE  — 归一化传感器读数
        """
        p = agent_state[:3]
        v = agent_state[3:]

        rel_pos = (p - self.goal) / self.state_scale
        vel_norm = v / self.max_speed
        goal_norm = self.goal / self.state_scale
        sensors = self._sensor_readings(p) / SENSOR_MAX_RANGE

        return np.concatenate([rel_pos, vel_norm, goal_norm, sensors])

    def _agent_dynamics(self, s, u):
        """3D 智能体动力学（一阶欧拉积分 + 边界钳位）。"""
        x, y, z, vx, vy, vz = s
        ax, ay, az = np.clip(u, -self.max_acc, self.max_acc)
        vx2 = np.clip(vx + ax * self.dt, -self.max_speed, self.max_speed)
        vy2 = np.clip(vy + ay * self.dt, -self.max_speed, self.max_speed)
        vz2 = np.clip(vz + az * self.dt, -self.max_speed, self.max_speed)
        x2 = np.clip(x + vx * self.dt, ENV_XMIN, ENV_XMAX)
        y2 = np.clip(y + vy * self.dt, ENV_YMIN, ENV_YMAX)
        z2 = np.clip(z + vz * self.dt, ENV_ZMIN, ENV_ZMAX)
        return np.array([x2, y2, z2, vx2, vy2, vz2], dtype=float)

    def step(self, s, u):
        """环境步进：移动智能体 + 移动所有动态障碍物。"""
        s_next = self._agent_dynamics(s, u)
        for dob in self.dynamic_obstacles:
            dob.step(self.dt)
        return s_next

    def obstacle_penalty(self, p):
        """
        3D 障碍物惩罚：穿透 → 1000，安全距离内 → 二次惩罚。
        """
        pen = 0.0
        for obs in self._all_obstacles():
            d = np.linalg.norm(p - obs.center) - obs.radius
            if d < 0.0:
                return 1000.0
            if d < self.safe_margin:
                pen += (self.safe_margin - d) ** 2 * 80.0
        return pen

    def _boundary_penalty(self, p):
        """边界软约束：越靠近边界，惩罚越大。"""
        pen = 0.0
        for pos, lo, hi in [(p[0], ENV_XMIN, ENV_XMAX),
                             (p[1], ENV_YMIN, ENV_YMAX),
                             (p[2], ENV_ZMIN, ENV_ZMAX)]:
            margin = 0.5
            if pos - lo < margin:
                pen += (margin - (pos - lo)) ** 2 * 30.0
            if hi - pos < margin:
                pen += (margin - (hi - pos)) ** 2 * 30.0
        return pen

    def shaped_reward(self, s, u, s_next):
        """
        Shaped 奖励 = 距离缩减量 - 控制代价 - 障碍物代价
                      - 边界代价 + 目标邻近奖励
        """
        old_dist = np.linalg.norm(s[:3] - self.goal)
        new_dist = np.linalg.norm(s_next[:3] - self.goal)
        dist_reduction = old_dist - new_dist

        control_cost = 0.001 * np.linalg.norm(u) ** 2
        obs_pen = self.obstacle_penalty(s_next[:3])
        obs_cost = min(obs_pen, 50.0)
        bnd_cost = min(self._boundary_penalty(s_next[:3]), 30.0)

        proximity_bonus = 0.0
        if new_dist < 2.5:
            proximity_bonus = (2.5 - new_dist) * 0.4

        return 15.0 * dist_reduction - control_cost - obs_cost - bnd_cost + proximity_bonus
