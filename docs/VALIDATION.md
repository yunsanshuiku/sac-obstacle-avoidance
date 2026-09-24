# 本地验证记录

整理日期：2026-09-24。环境：Windows，Python 3.10.20，PyTorch 2.12.0+cpu，NumPy 2.2.6，Matplotlib 3.10.9。所有本地验证使用 CPU；没有重新执行数千回合完整训练。

## 数值与集成检查

`python -m unittest discover -s tests -v`：**7 项通过**。

其中集成检查对三个场景分别执行 2 回合、每回合至多 8 步的训练，批量大小 4、回放容量 32、预热 0，检查优化器实际更新和数值有限性，再导出模型并经独立命令评估；另外检查四旋翼短程仿真。其他检查覆盖尺寸/限幅、Fibonacci 方向、种子复现、权重校验、模型往返保存和悬停平衡。

## 原代码迁移一致性

三个策略的全部参数张量与原 checkpoint **逐项完全相同**。分别以种子 2026 比较原始环境和独立仓库环境的前 40 步，观测、确定性动作、奖励及状态一致。记录见 [`migration.json`](../results/validation/migration.json)。这是有限轨迹上的迁移检查，不构成所有输入上的等价证明。

## 独立预训练评估

各场景使用连续整数种子 `10000, 10001, ...`、单 CPU 线程和确定性策略均值。所有样本均纳入统计，没有筛选成功案例。

| 场景 | 样本数 | 成功 | 碰撞 | 平均实际步数 | 平均最小净空 | 平均路径长度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static2d | 20 | 20 | 0 | 138.4 | 0.5118 | 11.9238 |
| dynamic2d | 30 | 29 | 1 | 199.0 | 0.4549 | 12.6826 |
| dynamic3d | 10 | 10 | 0 | 120.8 | 1.2086 | 12.1429 |

步数上限依次为 420、500、600。汇总 JSON 和逐次 CSV 位于 [`results/validation/`](../results/validation/)。动态二维的 1 次碰撞保留在记录中；不能把有限样本成功率当作任意场景的安全保证。不同场景任务定义不同，表格不用于算法优劣排序。

复现命令：

```bash
python -m sac_avoidance evaluate --scenario static2d --episodes 20 --max-steps 420 --seed 10000
python -m sac_avoidance evaluate --scenario dynamic2d --episodes 30 --max-steps 500 --seed 10000
python -m sac_avoidance evaluate --scenario dynamic3d --episodes 10 --max-steps 600 --seed 10000
python -m sac_avoidance uav --seed 42
```

## 四旋翼闭环

种子 42 下，仿真在 7.21 s 到达目标，未碰撞，最终位置误差 **0.397522 m**，最小障碍物净空 **1.282381 m**。全轨迹最大四元数单位范数误差约 **2.22×10⁻¹⁶**；油门范围约 **[0.5195, 1.0]**，包含上限饱和。数值结果与原仿真的约 0.398 m / 1.282 m 摘要一致。

## 打包与平台范围

通过 setuptools 构建并安装 wheel；在隔离工作目录中加载三个附带模型，并运行三维命令行评估，确认不依赖原实验目录或源码目录。

发布 ZIP 排除本地 `runs/`、构建目录、缓存和完整训练 checkpoint，仅保留明确发布的 Actor、参考日志、验证摘要和图示。GitHub Actions 已配置 Linux / Windows 和 Python 3.10 / 3.12；尚未在远程运行，不能据此声称全部平台已验证。
