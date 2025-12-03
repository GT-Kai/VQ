# SwanLab 回调模块

本模块提供了 `SwanLabCallback`，用于在训练过程中自动记录指标和图像到 SwanLab。

## 📋 功能特性

- ✅ **自动记录训练指标**：通过 PyTorch Lightning 的 `log()` 方法自动记录
- ✅ **图像可视化**：自动记录训练/验证/测试的重建图像
- ✅ **灵活配置**：可配置图像记录频率和数量
- ✅ **智能检测**：自动检测是否使用 SwanLab Logger
- ✅ **错误处理**：即使 SwanLab 未安装也不会中断训练

## 🚀 快速开始

### 基本使用

```python
from callback.MyCallback import SwanLabCallback
from lightning.pytorch.loggers import SwanLabLogger
import lightning.pytorch as pl
from model.modelModule import VQVAEModel
from dataloader.dataModule import AnimeDataModule

# 创建数据模块
datamodule = AnimeDataModule(
    data_dir="/path/to/images",
    batch_size=32,
)

# 创建模型
model = VQVAEModel(
    latent_channels=256,
    num_embeddings=1024,
)

# 创建 SwanLab Logger
swanlab_logger = SwanLabLogger(
    project="vq-vae-anime",
    experiment_name="vqvae-baseline",
)

# 创建 SwanLab 回调
swanlab_callback = SwanLabCallback(
    log_images_every_n_steps=500,    # 每 500 步记录一次训练图像
    log_images_every_n_epochs=1,     # 每个 epoch 记录一次验证图像
    n_samples=8,                     # 每次记录 8 张图像
)

# 创建训练器
trainer = pl.Trainer(
    max_epochs=100,
    logger=swanlab_logger,
    callbacks=[swanlab_callback],  # 添加回调
    accelerator="gpu",
    devices=1,
)

# 训练
trainer.fit(model, datamodule=datamodule)
```

## 📚 API 文档

### `SwanLabCallback`

SwanLab 回调类，用于记录训练指标和图像。

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_images_every_n_steps` | int | `500` | 训练时每隔多少步记录一次图像 |
| `log_images_every_n_epochs` | int | `1` | 验证时每隔多少个 epoch 记录一次图像 |
| `n_samples` | int | `8` | 每次记录的图像数量 |

#### 回调方法

- `on_train_batch_end()`: 训练批次结束时调用，记录训练图像
- `on_validation_epoch_end()`: 验证 epoch 结束时调用，记录验证图像
- `on_test_epoch_end()`: 测试 epoch 结束时调用，记录测试图像
- `on_train_start()`: 训练开始时调用
- `on_train_end()`: 训练结束时调用

## 💡 使用示例

### 示例 1: 基础配置

```python
swanlab_callback = SwanLabCallback(
    log_images_every_n_steps=500,
    log_images_every_n_epochs=1,
    n_samples=8,
)
```

### 示例 2: 更频繁的图像记录

```python
swanlab_callback = SwanLabCallback(
    log_images_every_n_steps=100,  # 每 100 步记录一次
    log_images_every_n_epochs=1,
    n_samples=16,  # 记录更多图像
)
```

### 示例 3: 减少图像记录频率（节省资源）

```python
swanlab_callback = SwanLabCallback(
    log_images_every_n_steps=1000,  # 每 1000 步记录一次
    log_images_every_n_epochs=5,    # 每 5 个 epoch 记录一次验证图像
    n_samples=4,  # 记录更少图像
)
```

### 示例 4: 完整训练脚本

```python
from callback.MyCallback import SwanLabCallback
from lightning.pytorch.loggers import SwanLabLogger
import lightning.pytorch as pl
from model.modelModule import VQVAEModel
from dataloader.dataModule import AnimeDataModule

# 数据模块
datamodule = AnimeDataModule(
    data_dir="/path/to/images",
    batch_size=32,
    val_split=0.2,
)

# 模型
model = VQVAEModel(
    latent_channels=256,
    num_embeddings=1024,
    embedding_dim=256,
    lambda_rec=1.0,
    lambda_vq=1.0,
    lambda_commit=0.25,
    lambda_perc=0.5,
)

# Logger
swanlab_logger = SwanLabLogger(
    project="vq-vae-anime",
    experiment_name="vqvae-baseline",
    config={
        "latent_channels": 256,
        "num_embeddings": 1024,
        "batch_size": 32,
    },
)

# 回调
swanlab_callback = SwanLabCallback(
    log_images_every_n_steps=500,
    log_images_every_n_epochs=1,
    n_samples=8,
)

# 训练器
trainer = pl.Trainer(
    max_epochs=100,
    logger=swanlab_logger,
    callbacks=[swanlab_callback],
    accelerator="gpu",
    devices=1,
    precision=16,  # 混合精度
    gradient_clip_val=1.0,
)

# 训练
trainer.fit(model, datamodule=datamodule)
```

## 📊 记录的内容

### 自动记录的指标

通过 PyTorch Lightning 的 `log()` 方法，以下指标会自动记录到 SwanLab：

**训练指标**:
- `train/recon_loss`: 重建损失
- `train/vq_loss`: VQ 损失
- `train/commitment_loss`: Commitment 损失
- `train/perc_loss`: 感知损失
- `train/total_loss`: 总损失
- `train/perplexity`: Code 困惑度
- `train/code_usage_rate`: Code 使用率

**验证指标**:
- `val/recon_loss`: 验证重建损失
- `val/vq_loss`: 验证 VQ 损失
- `val/total_loss`: 验证总损失
- 等等...

### 回调记录的图像

**训练图像** (`train/reconstruction`):
- 每 `log_images_every_n_steps` 步记录一次
- 包含原始图像和重建图像（上下排列）

**验证图像** (`val/reconstruction`):
- 每 `log_images_every_n_epochs` 个 epoch 记录一次
- 包含原始图像和重建图像（上下排列）

**测试图像** (`test/reconstruction`):
- 测试 epoch 结束时记录一次
- 包含原始图像和重建图像（上下排列）

## 🔧 工作原理

1. **自动检测**: 回调会自动检测是否使用了 SwanLab Logger
2. **图像获取**: 在训练/验证/测试过程中获取模型输出
3. **格式转换**: 将 PyTorch tensor 转换为 SwanLab 需要的格式
4. **记录**: 使用 `swanlab.log()` 记录图像到 SwanLab

## ⚠️ 注意事项

1. **SwanLab 安装**: 确保已安装 SwanLab (`pip install swanlab`)
2. **Logger 配置**: 必须使用 `SwanLabLogger`，否则回调不会记录图像
3. **图像格式**: 图像会自动归一化到 [0, 1] 范围
4. **性能影响**: 频繁记录图像可能会影响训练速度，建议合理设置频率

## 🐛 常见问题

### Q1: 回调没有记录图像

**解决方案**:
- 确保使用了 `SwanLabLogger`
- 检查 SwanLab 是否已安装
- 查看控制台是否有错误信息

### Q2: 图像格式不正确

**解决方案**:
- 回调会自动处理图像格式转换
- 如果仍有问题，检查模型输出是否为 `(B, C, H, W)` 格式

### Q3: 训练速度变慢

**解决方案**:
- 增大 `log_images_every_n_steps`（减少记录频率）
- 减小 `n_samples`（记录更少图像）

## 📝 测试

运行模块自带的测试代码：

```bash
cd /home/lick/project/VQ
python callback/MyCallback.py
```

## 📄 许可证

本项目遵循项目根目录的许可证。

