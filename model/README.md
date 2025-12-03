# VQ-VAE / VQ-GAN 模型模块

本模块实现了完整的 VQ-VAE / VQ-GAN 模型，基于 PyTorch Lightning 框架，支持 SwanLab 实验跟踪和可视化。

## 📋 目录

- [功能特性](#功能特性)
- [依赖要求](#依赖要求)
- [模块结构](#模块结构)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [SwanLab 集成](#swanlab-集成)
- [使用示例](#使用示例)
- [模型架构](#模型架构)

## ✨ 功能特性

- ✅ **完整 VQ-VAE 实现**：包含 Encoder、Decoder、Vector Quantizer 等核心组件
- ✅ **EMA 向量量化**：使用 EMA 更新策略，提升训练稳定性
- ✅ **感知损失支持**：集成 LPIPS 感知损失，提升重建质量
- ✅ **PyTorch Lightning 集成**：完全兼容 Lightning 框架
- ✅ **SwanLab 日志记录**：自动记录训练指标、图像和超参数
- ✅ **灵活配置**：支持自定义模型结构、损失权重等参数
- ✅ **Code 使用监控**：实时监控 codebook 使用情况和困惑度

## 📦 依赖要求

确保已安装以下依赖：

```bash
torch>=2.0.0
torchvision>=0.15.0
lightning>=2.0.0
swanlab>=0.3.0
numpy>=1.24.0
```

安装所有依赖：

```bash
pip install -r requirements.txt
```

## 🏗️ 模块结构

```
model/
├── modelModule.py    # 主要模型文件
└── README.md         # 本文档
```

### 主要组件

1. **ResBlock**: ResNet 风格的残差块
2. **Encoder**: VQ-VAE 编码器（4 个下采样 stage）
3. **Decoder**: VQ-VAE 解码器（4 个上采样 stage）
4. **VectorQuantizerEMA**: EMA 版向量量化器
5. **LPIPS**: 感知损失实现（基于 VGG）
6. **VQVAEModel**: PyTorch Lightning 模型类

## 🚀 快速开始

### 基本使用

```python
from model.modelModule import VQVAEModel
import lightning.pytorch as pl
from lightning.pytorch.loggers import SwanLabLogger
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
    embedding_dim=256,
    lambda_rec=1.0,
    lambda_vq=1.0,
    lambda_commit=0.25,
    lambda_perc=0.5,
)

# 创建 SwanLab Logger
swanlab_logger = SwanLabLogger(
    project="vq-vae-anime",
    experiment_name="vqvae-baseline",
)

# 创建训练器
trainer = pl.Trainer(
    max_epochs=100,
    logger=swanlab_logger,
    accelerator="gpu",
    devices=1,
)

# 训练
trainer.fit(model, datamodule=datamodule)
```

## 📚 API 文档

### `VQVAEModel`

PyTorch Lightning 模型类，实现完整的 VQ-VAE 训练流程。

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_channels` | int | `3` | 输入图像通道数 |
| `latent_channels` | int | `256` | 潜在空间通道数 |
| `encoder_channels` | Tuple[int, ...] | `(128, 256, 256, 256)` | Encoder 各 stage 通道数 |
| `decoder_channels` | Tuple[int, ...] | `(256, 256, 256, 128)` | Decoder 各 stage 通道数 |
| `num_res_blocks` | int | `2` | 每个 stage 的 ResBlock 数量 |
| `num_embeddings` | int | `1024` | Codebook 大小（K） |
| `embedding_dim` | int | `256` | Embedding 维度（D_e） |
| `commitment_cost` | float | `0.25` | Commitment 损失权重 |
| `decay` | float | `0.99` | EMA 衰减系数 |
| `lambda_rec` | float | `1.0` | 重建损失权重 |
| `lambda_vq` | float | `1.0` | VQ 损失权重 |
| `lambda_commit` | float | `0.25` | Commitment 损失权重 |
| `lambda_perc` | float | `0.5` | 感知损失权重 |
| `lambda_gan` | float | `0.0` | GAN 损失权重（VQ-GAN 扩展） |
| `use_lpips` | bool | `True` | 是否使用 LPIPS 感知损失 |
| `learning_rate` | float | `2e-4` | 学习率 |
| `betas` | Tuple[float, float] | `(0.9, 0.99)` | Adam 优化器 betas |
| `weight_decay` | float | `0.0` | 权重衰减 |
| `log_every_n_steps` | int | `100` | 日志记录频率 |
| `log_images_every_n_steps` | int | `500` | 图像记录频率 |

#### 方法

- `forward(x)`: 前向传播，返回重建图像和损失信息
- `encode(x)`: 编码图像到离散 code indices
- `decode(indices)`: 从 code indices 解码图像
- `training_step(batch, batch_idx)`: 训练步骤
- `validation_step(batch, batch_idx)`: 验证步骤
- `configure_optimizers()`: 配置优化器和学习率调度器

### `Encoder`

VQ-VAE 编码器，将输入图像编码到潜在空间。

**输入**: `(B, 3, 256, 256)` RGB 图像，值域 `[-1, 1]`  
**输出**: `(B, C_latent, 16, 16)` 特征图

### `Decoder`

VQ-VAE 解码器，从潜在空间重建图像。

**输入**: `(B, C_latent, 16, 16)` 量化后的特征图  
**输出**: `(B, 3, 256, 256)` RGB 图像，值域 `[-1, 1]`

### `VectorQuantizerEMA`

EMA 版向量量化器，实现离散化潜在表示。

**特性**:
- EMA 更新 codebook，提升训练稳定性
- 自动计算困惑度（perplexity）和 code 使用率
- 支持 straight-through estimator

### `LPIPS`

感知损失实现，使用预训练 VGG16 特征提取器。

**用途**: 提升重建图像的感知质量，特别是细节部分（线条、眼睛、发丝等）

## 🦢 SwanLab 集成

SwanLab 是一个开源的 AI 训练跟踪与可视化工具，支持云端和自托管使用。本模块已完全集成 SwanLab，可以自动记录训练过程中的各项指标。

### 安装 SwanLab

```bash
pip install swanlab
```

### 基本使用

```python
from lightning.pytorch.loggers import SwanLabLogger
import lightning.pytorch as pl

# 创建 SwanLab Logger
swanlab_logger = SwanLabLogger(
    project="vq-vae-anime",           # 项目名称
    experiment_name="vqvae-baseline",  # 实验名称
    config={                           # 超参数配置（可选）
        "latent_channels": 256,
        "num_embeddings": 1024,
        "learning_rate": 2e-4,
    }
)

# 在 Trainer 中使用
trainer = pl.Trainer(
    logger=swanlab_logger,
    max_epochs=100,
)
```

### 自动记录的内容

使用 SwanLab Logger 后，模型会自动记录：

1. **训练指标**:
   - `train/recon_loss`: 重建损失
   - `train/vq_loss`: VQ 损失
   - `train/commitment_loss`: Commitment 损失
   - `train/perc_loss`: 感知损失
   - `train/total_loss`: 总损失
   - `train/perplexity`: Code 困惑度
   - `train/code_usage_rate`: Code 使用率

2. **验证指标**:
   - `val/recon_loss`: 验证重建损失
   - `val/vq_loss`: 验证 VQ 损失
   - `val/perplexity`: 验证困惑度
   - 等等...

3. **图像可视化**:
   - `train/reconstruction`: 训练重建图像（每 500 步）
   - `val/reconstruction`: 验证重建图像（每个 epoch）

4. **超参数**:
   - 所有模型超参数（通过 `save_hyperparameters()` 自动保存）

### 查看实验结果

训练开始后，SwanLab 会自动生成一个 URL，可以在浏览器中查看实时训练进度：

```
SwanLab experiment started: https://swanlab.cn/your-username/vq-vae-anime/runs/xxx
```

在 SwanLab 界面中，您可以：
- 实时查看损失曲线
- 查看重建图像
- 对比不同实验的超参数
- 下载模型 checkpoint

### 高级配置

```python
swanlab_logger = SwanLabLogger(
    project="vq-vae-anime",
    experiment_name="vqvae-baseline",
    config={
        "model": {
            "latent_channels": 256,
            "num_embeddings": 1024,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 2e-4,
        },
    },
    tags=["baseline", "vqvae"],  # 实验标签
    notes="初始 VQ-VAE 实验",      # 实验备注
)
```

更多信息请参考 [SwanLab 官方文档](https://docs.swanlab.cn/)。

## 💡 使用示例

### 示例 1: 基础训练

```python
from model.modelModule import VQVAEModel
from dataloader.dataModule import AnimeDataModule
from lightning.pytorch.loggers import SwanLabLogger
import lightning.pytorch as pl

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
)

# Logger
logger = SwanLabLogger(project="vq-vae-anime")

# 训练器
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    accelerator="gpu",
    devices=1,
    precision=16,  # 混合精度训练
)

# 训练
trainer.fit(model, datamodule=datamodule)
```

### 示例 2: 自定义模型结构

```python
model = VQVAEModel(
    latent_channels=384,  # 更大的潜在空间
    encoder_channels=(128, 256, 384, 384),  # 自定义通道数
    decoder_channels=(384, 384, 256, 128),
    num_res_blocks=3,  # 更多的 ResBlock
    num_embeddings=2048,  # 更大的 codebook
    embedding_dim=384,
)
```

### 示例 3: 调整损失权重

```python
model = VQVAEModel(
    lambda_rec=1.0,      # 重建损失
    lambda_vq=1.0,      # VQ 损失
    lambda_commit=0.5,  # 增大 commitment（如果 code 使用不足）
    lambda_perc=1.0,    # 增大感知损失（如果重建模糊）
)
```

### 示例 4: 编码/解码图像

```python
# 编码图像到离散 codes
indices = model.encode(x)  # x: (B, 3, 256, 256)
# indices: (B, 256) - 每张图 16×16=256 个 code indices

# 从 codes 解码图像
x_recon = model.decode(indices)  # (B, 3, 256, 256)
```

### 示例 5: 禁用 LPIPS（加快训练）

```python
model = VQVAEModel(
    use_lpips=False,  # 禁用感知损失
    lambda_perc=0.0,
)
```

## 🏛️ 模型架构

### Encoder 架构

```
输入 (3, 256, 256)
  ↓
Conv 3×3 → 128 channels
  ↓
Stage 1: Conv 4×4 stride=2 → 128 → ResBlock × 2
  ↓ (128, 128, 128)
Stage 2: Conv 4×4 stride=2 → 256 → ResBlock × 2
  ↓ (64, 64, 256)
Stage 3: Conv 4×4 stride=2 → 256 → ResBlock × 2
  ↓ (32, 32, 256)
Stage 4: Conv 4×4 stride=2 → 256 → ResBlock × 2
  ↓ (16, 16, 256)
Conv 3×3 → latent_channels
  ↓
输出 (latent_channels, 16, 16)
```

### Decoder 架构

```
输入 (latent_channels, 16, 16)
  ↓
Conv 3×3 → 256 channels
  ↓
Stage 1: ResBlock × 2 → Upsample 2× → Conv 3×3 → 256
  ↓ (32, 32, 256)
Stage 2: ResBlock × 2 → Upsample 2× → Conv 3×3 → 256
  ↓ (64, 64, 256)
Stage 3: ResBlock × 2 → Upsample 2× → Conv 3×3 → 256
  ↓ (128, 128, 256)
Stage 4: ResBlock × 2 → Conv 3×3 → 128
  ↓ (128, 128, 128)
Conv 3×3 → 64 → ReLU → Conv 3×3 → 3 → Tanh
  ↓
输出 (3, 256, 256)
```

### 损失函数

总损失 = λ_rec × L_rec + λ_vq × L_vq + λ_commit × L_commit + λ_perc × L_perc

- **L_rec**: L1 重建损失
- **L_vq**: VQ codebook 损失
- **L_commit**: Commitment 损失
- **L_perc**: LPIPS 感知损失

## 📊 监控指标

### 困惑度 (Perplexity)

困惑度衡量 codebook 的使用分布均匀程度：

- **高困惑度** (> 500): Code 使用较均匀，模型学习良好
- **低困惑度** (< 100): 只有少数 code 被使用，可能存在 code collapse

### Code 使用率

Code 使用率 = 被使用的 code 数量 / 总 code 数量

- **目标**: > 80% 的 code 被使用
- **如果 < 50%**: 考虑减小 `num_embeddings` 或增大 `lambda_commit`

## 🐛 常见问题

### Q1: 训练时显存不足

**解决方案**:
- 减小 `batch_size`
- 使用混合精度训练 (`precision=16`)
- 减小 `latent_channels` 或 `num_embeddings`
- 使用梯度累积

### Q2: Code 使用率过低

**解决方案**:
- 增大 `lambda_commit`（如 0.25 → 0.5）
- 减小 `num_embeddings`（如 1024 → 512）
- 检查学习率是否过大

### Q3: 重建图像模糊

**解决方案**:
- 增大 `lambda_perc`（如 0.5 → 1.0）
- 增加 `num_res_blocks`
- 增大 `latent_channels` 或 `num_embeddings`

### Q4: SwanLab 无法记录图像

**解决方案**:
- 确保安装了最新版本的 SwanLab
- 检查网络连接（如果使用云端版本）
- 查看控制台错误信息

## 📝 测试

运行模块自带的测试代码：

```bash
cd /home/lick/project/VQ
python model/modelModule.py
```

测试会验证：
- 模型前向传播
- 编码/解码功能
- 损失计算

## 📄 许可证

本项目遵循项目根目录的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

