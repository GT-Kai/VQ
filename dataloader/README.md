# 数据加载模块 (DataLoader Module)

本模块提供了用于动漫头像 VQ-VAE / VQ-GAN 训练的数据加载功能，基于 PyTorch Lightning 框架实现。

## 📋 目录

- [功能特性](#功能特性)
- [依赖要求](#依赖要求)
- [模块结构](#模块结构)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [使用示例](#使用示例)
- [数据格式要求](#数据格式要求)

## ✨ 功能特性

- ✅ **自动数据集划分**：自动将数据集划分为训练集和验证集
- ✅ **数据增强**：支持水平翻转、颜色抖动等增强策略
- ✅ **标准化预处理**：自动将图像归一化到 `[-1, 1]` 范围
- ✅ **PyTorch Lightning 集成**：完全兼容 Lightning 框架
- ✅ **高性能配置**：支持多进程加载、内存固定等优化
- ✅ **可复现性**：支持随机种子设置，确保数据集划分可复现

## 📦 依赖要求

确保已安装以下依赖：

```bash
torch>=2.0.0
torchvision>=0.15.0
lightning>=2.0.0
Pillow>=9.0.0
```

安装所有依赖：

```bash
pip install -r requirements.txt
```

## 🏗️ 模块结构

```
dataloader/
├── dataModule.py    # 主要数据模块文件
└── README.md        # 本文档
```

### 主要组件

1. **AnimeFaceDataset**: PyTorch Dataset 类，用于加载单张图片
2. **AnimeDataModule**: PyTorch Lightning DataModule 类，管理整个数据流程
3. **get_train_transform()**: 获取训练时的数据变换（含增强）
4. **get_val_transform()**: 获取验证时的数据变换（无增强）

## 🚀 快速开始

### 基本使用

```python
from dataloader.dataModule import AnimeDataModule
import lightning.pytorch as pl

# 创建数据模块
datamodule = AnimeDataModule(
    data_dir="/path/to/your/images",
    image_size=256,
    batch_size=32,
    num_workers=4,
    val_split=0.2,
    train_augment=True,
)

# 在 PyTorch Lightning 训练器中使用
trainer = pl.Trainer()
trainer.fit(model, datamodule=datamodule)
```

### 独立使用 Dataset

```python
from dataloader.dataModule import AnimeFaceDataset, get_train_transform
from torch.utils.data import DataLoader

# 创建数据集
transform = get_train_transform(image_size=256, augment=True)
dataset = AnimeFaceDataset(
    data_dir="/path/to/images",
    image_size=256,
    transform=transform,
)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 📚 API 文档

### `AnimeFaceDataset`

PyTorch Dataset 类，用于加载动漫头像图片。

#### 参数

- `data_dir` (str): 图片数据目录路径
- `image_size` (int, optional): 目标图像尺寸，默认 `256`
- `transform` (transforms.Compose, optional): 数据变换，默认 `None`

#### 返回值

- `image` (torch.Tensor): 形状为 `(3, H, W)` 的 tensor，值域 `[-1, 1]`

#### 示例

```python
from dataloader.dataModule import AnimeFaceDataset, get_train_transform

transform = get_train_transform(image_size=256, augment=True)
dataset = AnimeFaceDataset(
    data_dir="/path/to/images",
    image_size=256,
    transform=transform,
)

# 获取单张图片
image = dataset[0]  # shape: (3, 256, 256), range: [-1, 1]
```

---

### `AnimeDataModule`

PyTorch Lightning DataModule 类，管理训练/验证数据流程。

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | str | - | 图片数据目录路径（必需） |
| `image_size` | int | `256` | 目标图像尺寸 |
| `batch_size` | int | `32` | 批次大小 |
| `num_workers` | int | `4` | DataLoader 工作进程数 |
| `val_split` | float | `0.2` | 验证集比例（0-1之间） |
| `train_augment` | bool | `True` | 训练时是否使用数据增强 |
| `seed` | int | `42` | 随机种子（用于数据集划分） |

#### 方法

- `setup(stage=None)`: 设置数据集，划分训练集和验证集
- `train_dataloader()`: 返回训练集 DataLoader
- `val_dataloader()`: 返回验证集 DataLoader
- `test_dataloader()`: 返回测试集 DataLoader（与验证集相同）

#### 示例

```python
from dataloader.dataModule import AnimeDataModule

datamodule = AnimeDataModule(
    data_dir="/path/to/images",
    image_size=256,
    batch_size=32,
    num_workers=4,
    val_split=0.2,
    train_augment=True,
    seed=42,
)

# 设置数据集（Lightning 会自动调用，也可手动调用）
datamodule.setup()

# 获取 DataLoader
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
```

---

### `get_train_transform()`

获取训练时的数据变换（包含数据增强）。

#### 参数

- `image_size` (int, optional): 目标图像尺寸，默认 `256`
- `augment` (bool, optional): 是否使用数据增强，默认 `True`

#### 返回

- `transforms.Compose`: 数据变换组合

#### 变换流程

1. Resize 到 `(image_size, image_size)`
2. （如果 `augment=True`）随机水平翻转（p=0.5）
3. （如果 `augment=True`）颜色抖动（亮度、对比度、饱和度 ±10%）
4. 转换为 Tensor
5. 归一化到 `[-1, 1]`

#### 示例

```python
from dataloader.dataModule import get_train_transform

transform = get_train_transform(image_size=256, augment=True)
```

---

### `get_val_transform()`

获取验证时的数据变换（不包含数据增强）。

#### 参数

- `image_size` (int, optional): 目标图像尺寸，默认 `256`

#### 返回

- `transforms.Compose`: 数据变换组合

#### 变换流程

1. Resize 到 `(image_size, image_size)`
2. 转换为 Tensor
3. 归一化到 `[-1, 1]`

#### 示例

```python
from dataloader.dataModule import get_val_transform

transform = get_val_transform(image_size=256)
```

## 💡 使用示例

### 示例 1: 在 PyTorch Lightning 中使用

```python
import lightning.pytorch as pl
from dataloader.dataModule import AnimeDataModule
from model.vqvae_module import VQVAEModel

# 创建数据模块
datamodule = AnimeDataModule(
    data_dir="/path/to/images",
    image_size=256,
    batch_size=32,
    num_workers=4,
    val_split=0.2,
)

# 创建模型
model = VQVAEModel(...)

# 创建训练器
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)

# 训练
trainer.fit(model, datamodule=datamodule)
```

### 示例 2: 手动测试数据加载

```python
from dataloader.dataModule import AnimeDataModule

# 创建数据模块
datamodule = AnimeDataModule(
    data_dir="/path/to/images",
    image_size=256,
    batch_size=4,
    num_workers=2,
    val_split=0.2,
    train_augment=True,
)

# 设置数据集
datamodule.setup()

# 获取训练集 DataLoader
train_loader = datamodule.train_dataloader()
print(f"训练集批次数量: {len(train_loader)}")

# 获取一个批次
batch = next(iter(train_loader))
print(f"批次形状: {batch.shape}")  # (batch_size, 3, 256, 256)
print(f"批次值域: [{batch.min():.3f}, {batch.max():.3f}]")  # 应该在 [-1, 1] 范围内

# 获取验证集 DataLoader
val_loader = datamodule.val_dataloader()
val_batch = next(iter(val_loader))
print(f"验证批次形状: {val_batch.shape}")
```

### 示例 3: 自定义数据增强

```python
from dataloader.dataModule import AnimeFaceDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# 自定义变换
custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    # 添加其他自定义增强...
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 使用自定义变换创建数据集
dataset = AnimeFaceDataset(
    data_dir="/path/to/images",
    image_size=256,
    transform=custom_transform,
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 示例 4: 禁用数据增强

```python
from dataloader.dataModule import AnimeDataModule

# 创建数据模块，禁用训练时的数据增强
datamodule = AnimeDataModule(
    data_dir="/path/to/images",
    train_augment=False,  # 禁用增强
)
```

## 📁 数据格式要求

### 目录结构

数据目录应包含图片文件（`.png` 或 `.jpg` 格式）：

```
data_dir/
├── 1.png
├── 2.png
├── 3.jpg
└── ...
```

### 图片要求

- **格式**: PNG 或 JPG
- **颜色模式**: 任意（会自动转换为 RGB）
- **尺寸**: 任意（会自动 resize 到指定尺寸）
- **数量**: 建议至少 1000 张以上

### 数据预处理

所有图片会自动进行以下处理：

1. **转换为 RGB**: 自动将图片转换为 RGB 格式
2. **Resize**: 调整到 `image_size × image_size`（默认 256×256）
3. **归一化**: 归一化到 `[-1, 1]` 范围

### 数据增强（训练时）

如果 `train_augment=True`，训练时会应用以下增强：

- **随机水平翻转**: 概率 50%
- **颜色抖动**: 
  - 亮度: ±10%
  - 对比度: ±10%
  - 饱和度: ±10%
  - 色调: 不扰动（避免颜色偏移过大）

## ⚙️ 性能优化建议

### 1. 调整 `num_workers`

根据 CPU 核心数调整工作进程数：

```python
import os
num_workers = min(os.cpu_count(), 8)  # 不超过 8 个进程

datamodule = AnimeDataModule(
    data_dir="/path/to/images",
    num_workers=num_workers,
)
```

### 2. 使用 `pin_memory`

如果使用 GPU，`pin_memory=True` 可以加速数据传输（已默认启用）。

### 3. 调整 `batch_size`

根据 GPU 显存调整批次大小：

- RTX 4090 (24GB): 建议 `batch_size=32`
- RTX 3090 (24GB): 建议 `batch_size=32`
- RTX 3080 (10GB): 建议 `batch_size=16`

如果显存不足，可以减小 `batch_size` 并使用梯度累积。

## 🐛 常见问题

### Q1: 找不到图片文件

**错误信息**: `ValueError: 在 ... 中未找到图片文件（.png 或 .jpg）`

**解决方案**: 
- 检查 `data_dir` 路径是否正确
- 确认目录中包含 `.png` 或 `.jpg` 文件
- 检查文件权限

### Q2: 内存不足

**解决方案**:
- 减小 `batch_size`
- 减小 `num_workers`
- 使用梯度累积

### Q3: 数据加载速度慢

**解决方案**:
- 增加 `num_workers`
- 使用 SSD 存储数据
- 确保 `pin_memory=True`（已默认启用）

### Q4: 数据集划分不一致

**解决方案**:
- 设置固定的 `seed` 参数
- 确保每次运行使用相同的 `seed`

## 📝 测试

运行模块自带的测试代码：

```bash
cd /home/lick/project/VQ
python dataloader/dataModule.py
```

测试会输出：
- 找到的图片数量
- 训练集和验证集大小
- 批次形状和值域范围

## 📄 许可证

本项目遵循项目根目录的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

