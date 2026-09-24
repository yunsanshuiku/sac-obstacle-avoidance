# 上传到 GitHub

建议仓库名：`sac-obstacle-avoidance`。建议描述：

> PyTorch Soft Actor-Critic for 2D/3D obstacle avoidance and quadrotor closed-loop simulation.

## 用 Git 上传

在 GitHub 创建空仓库，然后在本项目根目录运行。将 `YOUR_USERNAME` 替换为自己的账号：

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial SAC obstacle avoidance project"
git remote add origin https://github.com/YOUR_USERNAME/sac-obstacle-avoidance.git
git push -u origin main
```

若尚未配置 Git 作者信息，请使用自己的用户名和邮箱配置后再提交。仓库当前尚未绑定远程地址，也未替你发布到 GitHub。

## 用网页上传

也可以解压交付的 ZIP，将项目文件夹内的内容上传到空仓库根目录。不要将整个 ZIP 文件作为唯一内容上传，否则 GitHub 无法直接显示 README、代码和工作流。

网页上传时注意 `.github/`、`.gitignore`、`.gitattributes` 等隐藏文件；Git 命令会一并纳入。仓库中的模型小于 GitHub 单文件限制，无需 Git LFS。后续大型训练 checkpoint、回放池可单独放到 Release 或外部存储中。

## 发布内容

- 核心代码、测试、依赖文件、配置、中文 README。
- 三个精简后的策略权重及来源/校验信息。
- 历史实验日志和精选图片。
- GitHub Actions CPU 检查配置。

`runs/`、缓存、完整训练权重与回放数据默认忽略。保留的历史日志仅描述已有实验；新的实验应保留配置、种子和模型校验值。许可证由作者决定，当前 [LICENSE.md](../LICENSE.md) 明确标注尚未授予开源许可。
