# VQ-VAE 动漫头像训练项目

本项目基于 PyTorch Lightning 实现 VQ-VAE / VQ-GAN 模型，用于学习动漫头像的离散潜空间表示。

## 📋 目录

- [环境搭建](#环境搭建)
- [数据集准备](#数据集准备)
- [项目结构](#项目结构)
- [配置文件说明](#配置文件说明)
- [训练步骤](#训练步骤)
- [验证和测试](#验证和测试)
- [常见问题](#常见问题)

---

## 🚀 环境搭建

### 1. 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n VQ python=3.10
conda activate VQ
```

### 2. 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- SwanLab (实验日志)
- 其他依赖见 `requirements.txt`

---

## 📦 数据集准备

### 数据集格式

项目支持从目录直接加载图片，要求：

- **图片格式**：`.png` 或 `.jpg`
- **分辨率**：建议 256×256（程序会自动 resize）
- **目录结构**：

```
datas/
  your_dataset/
    image1.png
    image2.jpg
    ...
```

### 数据集配置

在 `conf/config.yaml` 中修改数据集路径：

```yaml
data:
  class_path: dataloader.dataModule.AnimeDataModule
  init_args:
    data_dir: /path/to/your/dataset  # 修改为你的数据集路径
    image_size: 256
    batch_size: 32
    num_workers: 4
    val_split: 0.2  # 验证集比例（20%）
    train_augment: true  # 是否启用数据增强
```

### 示例数据集

项目配置中使用的示例数据集路径（Kaggle）：
```yaml
data_dir: /home/lick/project/VQ/datas/soumikrakshit/anime-faces/versions/1/data
```

---

## 📁 项目结构

```
VQ/
├── main.py                 # 训练主程序（LightningCLI）
├── conf/
│   └── config.yaml        # 配置文件（模型、训练、数据参数）
├── model/
│   └── modelModule.py     # VQ-VAE 模型定义
├── dataloader/
│   └── dataModule.py      # 数据加载模块
├── callback/
│   └── MyCallback.py      # 自定义回调（日志、可视化）
├── bash/
│   └── run.sh             # 训练脚本示例
├── checkpoints/           # 模型检查点保存目录
├── requirements.txt       # Python 依赖
└── README.md             # 本文件
```

---

## ⚙️ 配置文件说明

配置文件位于 `conf/config.yaml`，主要包含三部分：

### 1. Trainer 配置

```yaml
trainer:
  accelerator: gpu         # 使用 GPU
  devices: 1              # GPU 数量
  max_epochs: 100         # 最大训练轮数
  precision: 16-mixed     # 混合精度训练
  gradient_clip_val: 1.0  # 梯度裁剪
  accumulate_grad_batches: 1  # 梯度累积步数
```

**多 GPU 训练**：
- 设置 `devices: 2` 使用 2 个 GPU
- 取消注释 `strategy: ddp` 启用分布式训练

### 2. Model 配置

```yaml
model:
  class_path: model.modelModule.VQVAEModel
  init_args:
    # 模型结构
    latent_channels: 256           # 潜变量通道数
    encoder_channels: [128, 256, 256, 256]
    decoder_channels: [256, 256, 256, 128]
    
    # 量化器参数
    num_embeddings: 1024          # Codebook 大小
    embedding_dim: 256            # Embedding 维度
    
    # 损失权重
    lambda_rec: 1.0               # 重建损失
    lambda_vq: 1.0                # VQ 损失
    lambda_commit: 0.25           # Commitment 损失
    lambda_perc: 0.5              # 感知损失（LPIPS）
    lambda_gan: 0.0               # GAN 损失（VQ-GAN 扩展）
    
    # 优化器
    learning_rate: 2.0e-4
    betas: [0.9, 0.99]
```

### 3. Data 配置

```yaml
data:
  class_path: dataloader.dataModule.AnimeDataModule
  init_args:
    data_dir: /path/to/dataset
    batch_size: 32
    image_size: 256
    val_split: 0.2
    train_augment: true
```

---

## 🏃 训练步骤

### 1. 验证配置

在开始训练前，验证配置文件是否正确：

```bash
python main.py fit --config conf/config.yaml --print_config
```

这会打印完整的配置信息，包括所有参数。

### 2. 快速测试运行

使用 `fast_dev_run` 模式快速验证代码是否能正常运行：

```bash
CUDA_VISIBLE_DEVICES=1 python main.py fit --config conf/config.yaml --trainer.fast_dev_run=true
```

### 3. 开始训练

使用提供的脚本：

```bash
cd bash
bash run.sh
```

或者直接运行：

```bash
python main.py fit --config conf/config.yaml
```

### 4. 从检查点继续训练

如果训练中断，可以从上次保存的检查点继续：

```bash
python main.py fit --config conf/config.yaml --ckpt_path checkpoints/last.ckpt
```

### 5. 使用特定 GPU

通过环境变量指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py fit --config conf/config.yaml
```

---

## 🔍 验证和测试

### 验证模式

使用验证集评估模型：

```bash
python main.py validate --config conf/config.yaml --ckpt_path checkpoints/best.ckpt
```

### 测试模式

```bash
python main.py test --config conf/config.yaml --ckpt_path checkpoints/best.ckpt
```

---

## 📊 监控和日志

### SwanLab 日志

项目集成了 SwanLab 进行实验跟踪，包括：

- 训练/验证损失曲线
- 重建图像可视化
- Codebook 使用统计
- 超参数记录

训练开始后，SwanLab 会自动记录日志。可以通过配置文件修改项目名称和实验名称：

```yaml
trainer:
  logger:
    - class_path: swanlab.integration.pytorch_lightning.SwanLabLogger
      init_args:
        project: vq-vae-anime
        experiment_name: vqvae-baseline
```

### Checkpoint 保存

模型检查点保存在 `checkpoints/` 目录：

- `vqvae-{epoch}-{loss}.ckpt`：按验证损失保存的最佳模型
- `last.ckpt`：最后一次训练的检查点

配置说明：
- `save_top_k: 3`：保留最好的 3 个模型
- `every_n_train_steps: 10000`：每 10000 步保存一次

---

## ❓ 常见问题

### 1. 显存不足（OOM）

**解决方案**：
- 减小 `batch_size`（如从 32 降到 16）
- 使用梯度累积：`accumulate_grad_batches: 2`
- 确保使用混合精度：`precision: 16-mixed`

### 2. 数据集路径错误

**错误信息**：`在 {data_dir} 中未找到图片文件`

**解决方案**：
- 检查 `conf/config.yaml` 中的 `data_dir` 路径是否正确
- 确保目录中包含 `.png` 或 `.jpg` 文件

### 3. Codebook 使用不足

**症状**：只有少数 code 被使用，重建图像模糊

**解决方案**：
- 增加 `lambda_commit`（如从 0.25 增加到 0.5）
- 减少 `num_embeddings`（如从 1024 降到 512）
- 检查学习率是否过大

### 4. 重建图像模糊

**解决方案**：
- 增加感知损失权重：`lambda_perc: 1.0`
- 增加 codebook 大小：`num_embeddings: 2048`
- 增加模型容量（更多 ResBlock 或更大通道数）

### 5. 训练不稳定

**解决方案**：
- 降低学习率：`learning_rate: 1.0e-4`
- 启用梯度裁剪：`gradient_clip_val: 1.0`
- 检查数据增强是否过度

### 6. 从检查点加载失败

**解决方案**：
- 确保检查点文件路径正确
- 检查模型配置是否与训练时一致
- 使用 `--ckpt_path` 参数指定完整路径

---

## 🔧 高级配置

### 多 GPU 训练

在 `conf/config.yaml` 中配置：

```yaml
trainer:
  accelerator: gpu
  devices: 2              # 使用 2 个 GPU
  strategy: ddp           # 启用分布式训练
```

注意：`batch_size` 是每个 GPU 的 batch size，总 batch size = `batch_size × num_gpus`。

### 自定义数据增强

修改 `dataloader/dataModule.py` 中的 `_get_train_transform()` 方法来自定义数据增强。

### 调整损失权重

根据训练效果调整 `conf/config.yaml` 中的损失权重：

- `lambda_rec`：重建损失（默认 1.0）
- `lambda_vq`：VQ 损失（默认 1.0）
- `lambda_commit`：Commitment 损失（默认 0.25）
- `lambda_perc`：感知损失（默认 0.5）

---

## 📝 训练建议

### 推荐训练流程

1. **阶段 1：基础训练**
   - 使用默认配置开始训练
   - 监控验证损失和重建质量
   - 训练至重建效果稳定

2. **阶段 2：调优**
   - 根据重建质量调整损失权重
   - 如果图像模糊，增加 `lambda_perc`
   - 如果 code 使用不足，调整 `lambda_commit`

3. **阶段 3：扩展（可选）**
   - 启用 GAN 损失扩展为 VQ-GAN
   - 逐步增加 `lambda_gan`（从 0 到 0.1）

### 训练时长估算

- 数据集：~10k 图像
- Batch size：32
- 每个 epoch：~320 steps
- 推荐训练：100-200k steps（约 320-640 epochs）

---

## 📚 相关文档

- `conf/README.md`：配置文件详细说明
- `model/README.md`：模型架构说明
- `dataloader/README.md`：数据加载说明
- `callback/README.md`：回调函数说明

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
