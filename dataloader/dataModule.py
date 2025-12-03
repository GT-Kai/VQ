"""
动漫头像 VQ-VAE / VQ-GAN 数据模块
实现 PyTorch Lightning 的 LightningDataModule
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning.pytorch as pl


class AnimeFaceDataset(Dataset):
    """动漫头像数据集类"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_dir: 图片数据目录路径
            image_size: 目标图像尺寸（默认 256）
            transform: 数据变换（包括预处理和增强）
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        
        # 获取所有图片文件路径
        self.image_paths = sorted(
            list(self.data_dir.glob("*.png")) + list(self.data_dir.glob("*.jpg"))
        )
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在 {data_dir} 中未找到图片文件（.png 或 .jpg）")
        
        print(f"找到 {len(self.image_paths)} 张图片")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        返回:
            image: 形状为 (3, H, W) 的 tensor，值域 [-1, 1]
        """
        img_path = self.image_paths[idx]
        
        # 加载图片并转换为 RGB
        image = Image.open(img_path).convert("RGB")
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        else:
            # 默认变换：resize + 归一化到 [-1, 1]
            transform_default = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
            ])
            image = transform_default(image)
        
        return image


def get_train_transform(image_size: int = 256, augment: bool = True) -> transforms.Compose:
    """
    获取训练时的数据变换（包含数据增强）
    
    Args:
        image_size: 目标图像尺寸
        augment: 是否使用数据增强
    
    Returns:
        transforms.Compose: 数据变换组合
    """
    transform_list = []
    
    # 预处理：将图片调整为正方形并 resize
    # 使用 CenterCrop 或 Resize，这里使用 Resize 保持简单
    # 如果原图不是正方形，会拉伸，也可以先 CenterCrop 再 Resize
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    if augment:
        # 数据增强
        # 水平翻转
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # 颜色抖动（亮度、对比度、饱和度）
        # 轻微扰动，避免破坏人脸结构
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.1,  # 亮度扰动 ±10%
                contrast=0.1,    # 对比度扰动 ±10%
                saturation=0.1,  # 饱和度扰动 ±10%
                hue=0.0          # 色调不扰动（避免颜色偏移过大）
            )
        )
    
    # 转换为 tensor 并归一化到 [-1, 1]
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    
    return transforms.Compose(transform_list)


def get_val_transform(image_size: int = 256) -> transforms.Compose:
    """
    获取验证时的数据变换（不包含数据增强）
    
    Args:
        image_size: 目标图像尺寸
    
    Returns:
        transforms.Compose: 数据变换组合
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class AnimeDataModule(pl.LightningDataModule):
    """PyTorch Lightning 数据模块"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        train_augment: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: 图片数据目录路径
            image_size: 目标图像尺寸（默认 256）
            batch_size: 批次大小（默认 32）
            num_workers: DataLoader 工作进程数（默认 4）
            val_split: 验证集比例（默认 0.2，即 20%）
            train_augment: 训练时是否使用数据增强（默认 True）
            seed: 随机种子（用于数据集划分）
        """
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.train_augment = train_augment
        self.seed = seed
        
        # 数据集对象（在 setup 中初始化）
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集（在训练/验证/测试前调用）"""
        # 获取所有图片路径
        data_dir_path = Path(self.data_dir)
        all_image_paths = sorted(
            list(data_dir_path.glob("*.png")) + list(data_dir_path.glob("*.jpg"))
        )
        
        if len(all_image_paths) == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到图片文件（.png 或 .jpg）")
        
        # 划分训练集和验证集索引
        dataset_size = len(all_image_paths)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        # 使用固定种子确保可复现
        torch.manual_seed(self.seed)
        shuffled_indices = torch.randperm(dataset_size).tolist()
        
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        
        # 为训练集和验证集分别设置变换
        train_transform = get_train_transform(
            image_size=self.image_size,
            augment=self.train_augment
        )
        val_transform = get_val_transform(image_size=self.image_size)
        
        # 创建训练集和验证集
        self.train_dataset = AnimeFaceDataset(
            data_dir=self.data_dir,
            image_size=self.image_size,
            transform=train_transform,
        )
        self.val_dataset = AnimeFaceDataset(
            data_dir=self.data_dir,
            image_size=self.image_size,
            transform=val_transform,
        )
        
        # 使用划分后的索引
        self.train_dataset.image_paths = [all_image_paths[i] for i in train_indices]
        self.val_dataset.image_paths = [all_image_paths[i] for i in val_indices]
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.val_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """返回训练集 DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """返回验证集 DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """返回测试集 DataLoader（与验证集相同）"""
        return self.val_dataloader()


if __name__ == "__main__":
    # 测试代码
    data_dir = "/home/lick/project/VQ/datas/soumikrakshit/anime-faces/versions/1/data"
    
    # 创建数据模块
    datamodule = AnimeDataModule(
        data_dir=data_dir,
        image_size=256,
        batch_size=4,
        num_workers=2,
        val_split=0.2,
        train_augment=True,
    )
    
    # 设置数据集
    datamodule.setup()
    
    # 测试训练集
    train_loader = datamodule.train_dataloader()
    print(f"\n训练集批次数量: {len(train_loader)}")
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print(f"批次形状: {batch.shape}")
    print(f"批次值域: [{batch.min():.3f}, {batch.max():.3f}]")
    print(f"期望值域: [-1.0, 1.0]")
    
    # 测试验证集
    val_loader = datamodule.val_dataloader()
    print(f"\n验证集批次数量: {len(val_loader)}")
    
    val_batch = next(iter(val_loader))
    print(f"验证批次形状: {val_batch.shape}")
    print(f"验证批次值域: [{val_batch.min():.3f}, {val_batch.max():.3f}]")
    
    print("\n数据模块测试通过！")

