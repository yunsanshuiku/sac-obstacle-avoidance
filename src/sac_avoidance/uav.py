"""SAC acceleration policy bridged to a 17-state quadrotor simulation."""
import hashlib
import json
import os
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .env3d import AvoidanceEnv3D
from .policies import load_policy

class QuadrotorModel:
    def __init__(self):
        self.m = 1.5
        self.g = 9.81
        self.J = np.array([
            [1.318e-2, 0, 0],
            [0, 1.318e-2, 0],
            [0, 0, 2.365e-2],
        ])
        self.d = 0.225
        self.Ct = 1.253e-5
        self.Cm = 1.852e-7
        self.CR = 578.95
        self.omega_b = 147.64
        self.motor_tau = 0.05
        self.Cd_v = 0.1
        self.Cd_w = 0.01
        sqrt2 = np.sqrt(2)
        self.B = np.array([
            [self.Ct, self.Ct, self.Ct, self.Ct],
            [self.Ct * self.d / sqrt2, -self.Ct * self.d / sqrt2, -self.Ct * self.d / sqrt2, self.Ct * self.d / sqrt2],
            [self.Ct * self.d / sqrt2, self.Ct * self.d / sqrt2, -self.Ct * self.d / sqrt2, -self.Ct * self.d / sqrt2],
            [self.Cm, -self.Cm, self.Cm, -self.Cm],
        ])

    def motor_to_thrust_torque(self, omega_actual):
        forces_moments = self.B @ (omega_actual ** 2)
        return forces_moments[0], forces_moments[1:]

    def quaternion_to_rotation_matrix(self, q):
        qw, qx, qy, qz = q
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            qw, qx, qy, qz = q / norm
        else:
            qw, qx, qy, qz = 1, 0, 0, 0
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)],
        ])

    def quaternion_derivative(self, q, omega):
        Omega = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0],
        ])
        return 0.5 * Omega @ q

    def dynamics(self, state, u_func):
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        motor_omega = state[13:17]
        u = u_func(state)
        omega_cmd = self.CR * u + self.omega_b
        motor_omega_dot = (omega_cmd - motor_omega) / self.motor_tau
        F_total, tau = self.motor_to_thrust_torque(motor_omega)
        R = self.quaternion_to_rotation_matrix(quat)
        thrust_body = np.array([0, 0, -F_total])
        gravity_inertial = np.array([0, 0, self.m * self.g])
        drag_force = -self.Cd_v * vel
        acceleration = (R @ thrust_body + gravity_inertial + drag_force) / self.m
        damping_torque = -self.Cd_w * omega
        alpha = np.linalg.solve(self.J, tau + damping_torque - np.cross(omega, self.J @ omega))
        state_dot = np.zeros(17)
        state_dot[0:3] = vel
        state_dot[3:6] = acceleration
        state_dot[6:10] = self.quaternion_derivative(quat, omega)
        state_dot[10:13] = alpha
        state_dot[13:17] = motor_omega_dot
        return state_dot

    def normalize_quaternion(self, state):
        q = state[6:10]
        n = np.linalg.norm(q)
        if n > 1e-10:
            state[6:10] = q / n
        return state


def sphere_surface(center, radius, resolution=12):
    u = np.linspace(0, 2 * np.pi, resolution * 2)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_to_yaw(q):
    qw, qx, qy, qz = q
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class SACBridgeRateController:
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.env = trainer.env
        self.B_inv = np.linalg.pinv(model.B)
        self.KQ_ATT = np.array([8.0, 8.0, 4.0])
        self.KP_RATE = np.array([0.14, 0.14, 0.08])
        self.MAX_TILT = np.deg2rad(35)
        self.MAX_THRUST = 30.0
        self.MAX_TORQUE = np.array([1.2, 1.2, 0.8])
        self.motor_kp = 0.003
        self.policy_dt = 0.05
        self.last_policy_t = -1e9
        self.a_uav_cmd = np.zeros(3)
        self.debug = {}

    @staticmethod
    def adp_to_uav_pos(p_adp):
        return np.array([p_adp[0], p_adp[1], -p_adp[2]], dtype=float)

    @staticmethod
    def uav_to_adp_pos(p_uav):
        return np.array([p_uav[0], p_uav[1], -p_uav[2]], dtype=float)

    @staticmethod
    def adp_to_uav_vel(v_adp):
        return np.array([v_adp[0], v_adp[1], -v_adp[2]], dtype=float)

    @staticmethod
    def uav_to_adp_vel(v_uav):
        return np.array([v_uav[0], v_uav[1], -v_uav[2]], dtype=float)

    def thrust_to_attitude(self, thrust_dir, current_yaw):
        tx = np.clip(thrust_dir[0], -np.sin(self.MAX_TILT), np.sin(self.MAX_TILT))
        ty = np.clip(thrust_dir[1], -np.sin(self.MAX_TILT), np.sin(self.MAX_TILT))
        tz = thrust_dir[2]
        desired_roll = np.arcsin(np.clip(ty, -1.0, 1.0))
        desired_pitch = np.arctan2(-tx, max(-tz, 1e-10))
        return euler_to_quaternion(desired_roll, desired_pitch, current_yaw)

    def attitude_to_rate(self, quat, desired_quat):
        qd = desired_quat
        qe_w = qd[0] * quat[0] + qd[1] * quat[1] + qd[2] * quat[2] + qd[3] * quat[3]
        qe_x = qd[0] * quat[1] - qd[1] * quat[0] - qd[2] * quat[3] + qd[3] * quat[2]
        qe_y = qd[0] * quat[2] + qd[1] * quat[3] - qd[2] * quat[0] + qd[3] * quat[1]
        qe_z = qd[0] * quat[3] + qd[1] * quat[2] - qd[2] * quat[1] - qd[3] * quat[0]
        if qe_w < 0:
            qe_x, qe_y, qe_z = -qe_x, -qe_y, -qe_z
        att_error = 2.0 * np.array([qe_x, qe_y, qe_z])
        return -self.KQ_ATT * att_error, att_error

    def rate_control(self, omega, omega_ref):
        return np.clip(self.KP_RATE * (omega_ref - omega), -self.MAX_TORQUE, self.MAX_TORQUE)

    def motor_allocation(self, Fz, tau, motor_omega_actual):
        omega_sq_des = self.B_inv @ np.array([Fz, tau[0], tau[1], tau[2]])
        omega_sq_des = np.maximum(omega_sq_des, 0.0)
        omega_des = np.sqrt(omega_sq_des)
        u_ff = (omega_des - self.model.omega_b) / self.model.CR
        u_fb = self.motor_kp * (omega_des - motor_omega_actual)
        return np.clip(u_ff + u_fb, 0.0, 1.0), omega_des

    def update_policy_acc_cmd(self, t, pos_uav, vel_uav):
        if (t - self.last_policy_t) < self.policy_dt:
            return
        p_adp = self.uav_to_adp_pos(pos_uav)
        v_adp = self.uav_to_adp_vel(vel_uav)
        s_raw = np.concatenate([p_adp, v_adp])
        sf = self.env._state_features(s_raw).astype(np.float32)
        a_adp = self.trainer.select_action(sf, deterministic=True)
        self.a_uav_cmd = self.adp_to_uav_vel(a_adp)
        self.last_policy_t = t
        self.debug['a_adp'] = a_adp
        self.debug['a_uav'] = self.a_uav_cmd

    def __call__(self, t, state):
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]
        motor_omega = state[13:17]
        self.update_policy_acc_cmd(t, pos, vel)
        desired_thrust = self.model.m * (self.a_uav_cmd - np.array([0, 0, self.model.g]))
        thrust_mag = np.linalg.norm(desired_thrust)
        thrust_dir = desired_thrust / thrust_mag if thrust_mag >= 1e-6 else np.array([0.0, 0.0, -1.0])
        Fz = np.clip(thrust_mag, 0.1 * self.model.m * self.model.g, self.MAX_THRUST)
        q_des = self.thrust_to_attitude(thrust_dir, quat_to_yaw(quat))
        omega_ref, att_error = self.attitude_to_rate(quat, q_des)
        tau = self.rate_control(omega, omega_ref)
        u, omega_des = self.motor_allocation(Fz, tau, motor_omega)
        self.debug['att_error'] = att_error
        self.debug['omega_ref'] = omega_ref
        self.debug['tau'] = tau
        self.debug['Fz'] = Fz
        self.debug['omega_des'] = omega_des
        return u


def quat_to_euler_series(qw, qx, qy, qz):
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return roll, pitch, yaw


def simulate(model_path=None, out_dir='runs/uav', seed=42, duration=20.0, device='cpu'):
    """Run the original 100 Hz rigid-body simulation with a 20 Hz SAC command.

    Policy coordinates use +z up, the quadrotor model uses +z down.
    Returns numerical diagnostics and writes trajectory/control plots.
    """
    if not np.isfinite(duration) or duration < 0.01:
        raise ValueError('duration must be finite and at least 0.01 seconds')
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    actor = load_policy('dynamic3d', model_path, device=device)
    model_path = actor.source
    trainer = SimpleNamespace(env=AvoidanceEnv3D(seed=seed), select_action=actor.act)
    trainer.env._reset_dynamic_obstacles()
    trainer.env._randomize_static_obstacles()

    model = QuadrotorModel()
    ctrl = SACBridgeRateController(model, trainer)
    env = trainer.env

    initial_state = np.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        model.omega_b, model.omega_b, model.omega_b, model.omega_b,
    ], dtype=float)

    dt = 0.01
    t_final = duration
    n_steps = int(t_final / dt) + 1
    ts = np.arange(n_steps) * dt
    xs = np.zeros((n_steps, 17))
    xs[0] = initial_state

    min_obs_dist = 1e9
    reached_step = None
    collided_step = None
    a_cmd_norm_hist = []
    dob_trajs = [[] for _ in range(len(env.dynamic_obstacles))]
    dob_times = [0.0]
    for i, dob in enumerate(env.dynamic_obstacles):
        dob_trajs[i].append(dob.center.copy())

    u_hist, tau_hist, Fz_hist, omega_ref_hist, att_err_hist = [], [], [], [], []
    policy_stride = max(1, int(round(ctrl.policy_dt / dt)))

    for k in range(n_steps - 1):
        xk = xs[k].copy()
        if (k % policy_stride) == 0:
            for dob in env.dynamic_obstacles:
                dob.step(ctrl.policy_dt)
            dob_times.append(ts[k] + ctrl.policy_dt)
            for i, dob in enumerate(env.dynamic_obstacles):
                dob_trajs[i].append(dob.center.copy())

        u_now = ctrl(ts[k], xk)
        x_next = xk + dt * model.dynamics(xk, lambda _: u_now)
        x_next = model.normalize_quaternion(x_next)
        xs[k + 1] = x_next

        if 'a_uav' in ctrl.debug:
            a_cmd_norm_hist.append(float(np.linalg.norm(ctrl.debug['a_uav'])))
        u_hist.append(u_now.copy())
        tau_hist.append(ctrl.debug.get('tau', np.zeros(3)).copy())
        Fz_hist.append(float(ctrl.debug.get('Fz', 0.0)))
        omega_ref_hist.append(ctrl.debug.get('omega_ref', np.zeros(3)).copy())
        att_err_hist.append(ctrl.debug.get('att_error', np.zeros(3)).copy())

        p_adp = ctrl.uav_to_adp_pos(x_next[:3])
        d_now = min(np.linalg.norm(p_adp - obs.center) - obs.radius for obs in env._all_obstacles())
        min_obs_dist = min(min_obs_dist, d_now)
        if collided_step is None and env.obstacle_penalty(p_adp) >= 1000.0:
            collided_step = k + 1
        if reached_step is None:
            goal_dist = np.linalg.norm(p_adp - env.goal)
            speed_adp = np.linalg.norm(ctrl.uav_to_adp_vel(x_next[3:6]))
            if goal_dist < 0.40 and speed_adp < 0.50:
                reached_step = k + 1
        if reached_step is not None or collided_step is not None:
            break

    last_idx = np.where(np.linalg.norm(xs, axis=1) > 0)[0][-1]
    traj = xs[:last_idx + 1]
    t_used = ts[:last_idx + 1]
    p_uav = traj[:, :3]
    p_adp = np.stack([p_uav[:, 0], p_uav[:, 1], -p_uav[:, 2]], axis=1)

    u_hist = np.array(u_hist[:len(t_used)-1]) if u_hist else np.zeros((0, 4))
    tau_hist = np.array(tau_hist[:len(t_used)-1]) if tau_hist else np.zeros((0, 3))
    omega_ref_hist = np.array(omega_ref_hist[:len(t_used)-1]) if omega_ref_hist else np.zeros((0, 3))

    p_end_adp = p_adp[-1]
    goal_err = np.linalg.norm(p_end_adp - env.goal)

    print('=' * 60)
    print('SAC -> UAV 集成仿真完成')
    print(f'模型路径: {model_path}')
    print(f'目标点(SAC/ADP): ({env.goal[0]:.2f}, {env.goal[1]:.2f}, {env.goal[2]:.2f})')
    print(f'最终点(SAC/ADP): ({p_end_adp[0]:.3f}, {p_end_adp[1]:.3f}, {p_end_adp[2]:.3f})')
    print(f'最终目标误差: {goal_err:.3f} m')
    print(f'最小障碍距离: {min_obs_dist:.3f} m')
    print(f'到达目标: {"是" if reached_step is not None else "否"}')
    print(f'发生碰撞: {"是" if collided_step is not None else "否"}')
    print(f'四元数范数: {np.linalg.norm(traj[-1, 6:10]):.6f}')
    if a_cmd_norm_hist:
        print(f'a_cmd(SAC->UAV) 均值范数: {np.mean(a_cmd_norm_hist):.3f}')

    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    for i, obs in enumerate(env.static_obstacles):
        xs_s, ys_s, zs_s = sphere_surface(obs.center, obs.radius, resolution=14)
        ax3d.plot_surface(xs_s, ys_s, zs_s, color='#888888', alpha=0.40,
                          label='Static Obstacle' if i == 0 else None)
    dyn_colors = ['#FF6B6B', '#FFA500', '#FFD700', '#66CCFF']
    for i, tr in enumerate(dob_trajs):
        tr_arr = np.array(tr)
        c = dyn_colors[i % len(dyn_colors)]
        ax3d.plot(tr_arr[:, 0], tr_arr[:, 1], tr_arr[:, 2], '--', color=c, alpha=0.7,
                  lw=1.5, label=f'Dynamic Obs {i+1} Path')
        r = env.dynamic_obstacles[i].radius
        xs0, ys0, zs0 = sphere_surface(tr_arr[0], r, resolution=10)
        ax3d.plot_surface(xs0, ys0, zs0, color=c, alpha=0.45)
    ax3d.plot(p_adp[:, 0], p_adp[:, 1], p_adp[:, 2], 'b-', lw=2.2, label='UAV traj')
    ax3d.scatter(*env.goal, c='r', marker='*', s=140, label='Goal')
    ax3d.scatter(p_adp[0, 0], p_adp[0, 1], p_adp[0, 2], c='g', s=80, label='Start')
    ax3d.set_title('SAC-UAV Trajectory with Obstacles')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.legend(fontsize=7, loc='upper left')
    fig3d.tight_layout()
    traj_path = os.path.join(out_dir, 'sac_uav_trajectory_3d.png')
    fig3d.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig3d)

    roll, pitch, yaw = quat_to_euler_series(traj[:, 6], traj[:, 7], traj[:, 8], traj[:, 9])
    fig_ctrl, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes[0, 0].plot(t_used, roll, label='roll')
    axes[0, 0].plot(t_used, pitch, label='pitch')
    axes[0, 0].plot(t_used, yaw, label='yaw')
    axes[0, 0].set_title('Attitude (Euler, rad)')
    axes[0, 0].grid(True); axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(t_used, traj[:, 10], label='wx')
    axes[0, 1].plot(t_used, traj[:, 11], label='wy')
    axes[0, 1].plot(t_used, traj[:, 12], label='wz')
    if len(omega_ref_hist) > 0:
        t_ctrl = t_used[:-1]
        axes[0, 1].plot(t_ctrl, omega_ref_hist[:, 0], '--', alpha=0.6, label='wx_ref')
        axes[0, 1].plot(t_ctrl, omega_ref_hist[:, 1], '--', alpha=0.6, label='wy_ref')
        axes[0, 1].plot(t_ctrl, omega_ref_hist[:, 2], '--', alpha=0.6, label='wz_ref')
    axes[0, 1].set_title('Angular Rate (rad/s)')
    axes[0, 1].grid(True); axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 0].plot(t_used, traj[:, 3], label='vx')
    axes[1, 0].plot(t_used, traj[:, 4], label='vy')
    axes[1, 0].plot(t_used, traj[:, 5], label='vz')
    axes[1, 0].set_title('Velocity (m/s)')
    axes[1, 0].grid(True); axes[1, 0].legend(fontsize=8)
    if len(u_hist) > 0:
        t_ctrl = t_used[:-1]
        for i in range(4):
            axes[1, 1].plot(t_ctrl, u_hist[:, i], label=f'u{i+1}')
        axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title('Motor Outputs (throttle)')
    axes[1, 1].grid(True); axes[1, 1].legend(fontsize=8)
    if len(tau_hist) > 0:
        t_ctrl = t_used[:-1]
        axes[2, 0].plot(t_ctrl, tau_hist[:, 0], label='tau_x')
        axes[2, 0].plot(t_ctrl, tau_hist[:, 1], label='tau_y')
        axes[2, 0].plot(t_ctrl, tau_hist[:, 2], label='tau_z')
    axes[2, 0].set_title('Control Torque (N·m)')
    axes[2, 0].grid(True); axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(t_used, p_adp[:, 0], label='x')
    axes[2, 1].plot(t_used, p_adp[:, 1], label='y')
    axes[2, 1].plot(t_used, p_adp[:, 2], label='z')
    axes[2, 1].axhline(env.goal[0], ls='--', alpha=0.4, color='C0')
    axes[2, 1].axhline(env.goal[1], ls='--', alpha=0.4, color='C1')
    axes[2, 1].axhline(env.goal[2], ls='--', alpha=0.4, color='C2')
    axes[2, 1].set_title('Position (SAC/ADP frame)')
    axes[2, 1].grid(True); axes[2, 1].legend(fontsize=8)
    fig_ctrl.tight_layout()
    ctrl_path = os.path.join(out_dir, 'sac_uav_states_controls.png')
    fig_ctrl.savefig(ctrl_path, dpi=150, bbox_inches='tight')
    plt.close(fig_ctrl)
    print(f'静态图已保存: {traj_path}')
    print(f'状态与控制图已保存: {ctrl_path}')



    summary = {
        'checkpoint_sha256': hashlib.sha256(actor.source.read_bytes()).hexdigest(),
        'seed': seed, 'duration_seconds': float(t_used[-1]),
        'success': reached_step is not None and collided_step is None,
        'collision': collided_step is not None, 'final_distance': float(goal_err),
        'min_clearance': float(min_obs_dist),
        'quaternion_norm': float(np.linalg.norm(traj[-1, 6:10])),
        'max_quaternion_norm_error': float(np.max(np.abs(np.linalg.norm(traj[:, 6:10], axis=1) - 1.0))),
        'throttle_min': float(u_hist.min()), 'throttle_max': float(u_hist.max()),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    np.savez_compressed(os.path.join(out_dir, 'rollout.npz'), time=t_used, state=traj, throttle=u_hist)
    return summary
