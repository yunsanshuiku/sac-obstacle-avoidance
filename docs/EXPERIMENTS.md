# 实验来源与复现

## 代码映射

| 原工作区来源 | 本仓库模块 |
| --- | --- |
| `adp22s/sac_project/sac_train_and_plot.py` | `static2d/core.py` 中的环境与 SAC 核心 |
| `adp22s/sac_project/sac_fast_train_and_plot.py` | `static2d/fast.py` 中的快速路径奖励 |
| `adp22s/sac_project/sac_fast_full1400_train_and_plot.py` | `static2d/training.py` |
| `adp33s/sac_project/adpex_sac_dynamic.py` | `dynamic2d.py` |
| `adp44s/projects/sac_3d/adpex_dynamic_3d.py` | `env3d.py` 中的环境、几何与奖励，移除 ADP 网络和训练逻辑 |
| `adp44s/projects/sac_3d/adpex_sac_3d.py` | `dynamic3d.py` |
| `sac_uav_integration.py` | `uav.py` |

本仓库可以独立运行，不需要上述原始目录。静态场景选择保留完整 1400 回合快速路径版本；早期重复训练入口、仅加载 Actor 的续训实验及非 SAC 项目未纳入。

## 权重与日志

三个预训练 Actor 从原工作区最佳模型提取，参数张量原样保留。`src/sac_avoidance/pretrained/manifest.json` 记录原始文件来源、SHA-256、导出文件 SHA-256、回合标签和可获得的配置。静态场景原 checkpoint 未包含完整配置，其超参数依据原训练脚本。

导出文件只包含 Actor 和基本元数据，项目通过 `torch.load(..., weights_only=True)` 加载，不依赖自定义 pickle 对象。不同场景权重不可互换；训练器完整 checkpoint 与 `policy.pt` 格式不同。当前 CLI 的 `--checkpoint` 接受后者。

`results/reference/provenance.json` 记录复制的日志、图片来源和校验值。二维动态历史 JSON 包含逐回合及评估记录，原 `dynamic2d_rollout.txt` 是单次推演摘要。三维日志保留到标签 2800：原程序从标签 0 开始，标签数值不应解释为精确的已完成回合总数。

## 历史口径

- 静态场景最佳模型的 20 次评估来自同一固定布局，主要扰动起点，不能外推为任意障碍布局成功率。
- 二维动态训练评估把未成功样本的步数记为回合上限；代表性推演摘要为单条轨迹。
- 三维训练评估的 `avg_steps` 只在成功样本上平均；`avg_min_obstacle_distance` 在所有评估样本上平均，且初始上界是传感器最大量程。
- 新 `evaluate` 命令使用独立连续整数种子，记录所有样本的实际执行步数，几何净空不受传感器量程截断，并在 JSON 中记录模型哈希。不要直接混用新旧指标。
- 训练中的评估和训练共用部分随机数状态；在原实验中改变评估间隔可能改变后续训练轨迹。仅设置相同种子不保证跨 PyTorch 版本、CPU/GPU 平台逐位一致。

默认训练配置来自原脚本，二维动态最佳权重的回合标签为 3200，其 checkpoint 记录了某次 1000 回合训练配置；历史 JSON 覆盖的训练范围与单次训练配置不同。三维默认脚本配置为 5000 回合，保留结果的实际日志止于标签 2800。不能据默认配置声称历史实验完成了全部默认回合。

## 整理改动

保留原有网络、奖励、动力学、目标更新和终止约定；主要改动为相对包导入、可配置输出目录/设备、统一 CLI、预训练 Actor 导出、独立评估器与数值检查。二维动态新 checkpoint 的 NumPy 元数据转为基本 Python 类型，使新文件可安全加载；其损失缓存改为有界窗口。

三维环境拆除不参与 SAC 的 ADP 网络。UAV 仿真使用轻量 Actor 接口，并增加 JSON/NPZ 输出。原浮点策略调度和障碍物更新时间保留，不把该模型标为真实飞行验证。

## 本地验证

本次整理的验证结果及环境版本记录在 [VALIDATION.md](VALIDATION.md)。短程训练只验证数据采集、反向传播、模型保存及加载链路；不作为新训练收敛或泛化成功率证据。
