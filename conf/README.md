# 配置文件说明

本目录包含 VQ-VAE 训练的配置文件，使用 PyTorch Lightning 的 LightningCLI 进行配置管理。

## 📋 文件说明

- `config.yaml`: 主配置文件，包含所有模块的参数

## 🚀 快速开始

### 基本使用

使用 LightningCLI 从配置文件启动训练：

```bash
python main.py fit --config conf/config.yaml
```

### 其他命令

```bash
# 验证配置（不实际训练）
python main.py fit --config conf/config.yaml --print_config

# 测试运行（快速验证）
python main.py fit --config conf/config.yaml --trainer.fast_dev_run=true

# 从 checkpoint 继续训练
python main.py fit --config conf/config.yaml --ckpt_path checkpoints/last.ckpt

# 仅验证
python main.py validate --config conf/config.yaml --ckpt_path checkpoints/best.ckpt

# 测试
python main.py test --config conf/config.yaml --ckpt_path checkpoints/best.ckpt
```

## 📚 配置文件结构

### 全局配置

```yaml
seed_everything: 42  # 全局随机种子
```

### Trainer 配置

```yaml
trainer:
  accelerator: gpu
  devices: 1
  max_epochs: 100
  precision: 16-mixed
  callbacks: [...]
  logger: [...]
```

### Model 配置

```yaml
model:
  class_path: model.modelModule.VQVAEModel
  init_args:
    latent_channels: 256
    num_embeddings: 1024
    # ... 其他参数
```

### Data 配置

```yaml
data:
  class_path: dataloader.dataModule.AnimeDataModule
  init_args:
    data_dir: /path/to/images
    batch_size: 32
    # ... 其他参数
```

## 💡 配置示例

### 修改数据路径

```yaml
data:
  init_args:
    data_dir: /your/custom/path/to/images
```

### 修改模型参数

```yaml
model:
  init_args:
    latent_channels: 384  # 更大的潜在空间
    num_embeddings: 2048   # 更大的 codebook
    lambda_perc: 1.0       # 增大感知损失权重
```

### 修改训练参数

```yaml
trainer:
  max_epochs: 200
  devices: 2  # 使用 2 个 GPU
  strategy: ddp  # 分布式训练
```

### 添加多个 Logger

```yaml
trainer:
  logger:
    - class_path: swanlab.integration.pytorch_lightning.SwanLabLogger
      init_args:
        project: vq-vae-anime
        experiment_name: vqvae-baseline
    - class_path: lightning.pytorch.loggers.TensorBoardLogger
      init_args:
        save_dir: logs/
        name: vqvae_anime
```

## 🔧 常用配置修改

### 1. 调整批次大小

```yaml
data:
  init_args:
    batch_size: 16  # 如果显存不足，减小批次大小
```

### 2. 启用多 GPU 训练

```yaml
trainer:
  devices: 2  # 或 [0, 1] 指定 GPU
  strategy: ddp
```

### 3. 修改学习率

```yaml
model:
  init_args:
    learning_rate: 1.0e-4  # 降低学习率
```

### 4. 调整损失权重

```yaml
model:
  init_args:
    lambda_commit: 0.5  # 如果 code 使用不足，增大此值
    lambda_perc: 1.0    # 如果重建模糊，增大此值
```

### 5. 禁用数据增强

```yaml
data:
  init_args:
    train_augment: false
```

## 📝 创建自定义配置

可以基于 `config.yaml` 创建新的配置文件：

```bash
cp conf/config.yaml conf/my_experiment.yaml
# 然后编辑 my_experiment.yaml
```

使用自定义配置：

```bash
python main.py fit --config conf/my_experiment.yaml
```

## 🐛 常见问题

### Q1: 配置文件加载失败

**解决方案**:
- 检查 YAML 语法（缩进、引号等）
- 使用 `--print_config` 验证配置
- 检查 `class_path` 是否正确

### Q2: 模块导入错误

**解决方案**:
- 确保在项目根目录运行
- 检查 `class_path` 中的模块路径是否正确
- 确保所有依赖已安装

### Q3: 参数类型错误

**解决方案**:
- 列表参数使用 YAML 列表格式：`[128, 256, 256, 256]`
- 布尔值使用 `true`/`false`
- 浮点数可以使用科学计数法：`2.0e-4`

## 📄 更多信息

- [PyTorch Lightning CLI 文档](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
- [配置文件格式说明](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html)

