"""
SwanLab 回调函数
用于记录训练/验证/测试指标和图像到 SwanLab
"""

from typing import Optional
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
import swanlab


class SwanLabCallback(Callback):
    """SwanLab 回调函数，用于记录训练指标和图像"""
    
    def __init__(
        self,
        log_images_every_n_steps: int = 500,
        log_images_every_n_epochs: int = 1,
        n_samples: int = 8,
    ):
        """
        Args:
            log_images_every_n_steps: 训练时每隔多少步记录一次图像
            log_images_every_n_epochs: 验证时每隔多少个 epoch 记录一次图像
            n_samples: 每次记录的图像数量
        """
        super().__init__()
        self.log_images_every_n_steps = log_images_every_n_steps
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.n_samples = n_samples
        
        # 导入 swanlab
        self.swanlab = swanlab
        self.swanlab_available = True

    
    def _is_swanlab_logger(self, logger) -> bool:
        """检查是否为 SwanLab Logger"""
        if logger is None:
            return False
        logger_name = type(logger).__name__
        return logger_name == "SwanLabLogger" or "swanlab" in str(type(logger)).lower()
    
    def _log_images_to_swanlab(self, images: torch.Tensor, key: str, step: int):
        """将图像记录到 SwanLab"""
        # 确保图像在 CPU 上且为 numpy 格式
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().detach().numpy()
        else:
            images_np = np.array(images)
        
        # 转换图像格式: (B, C, H, W) -> (B, H, W, C)
        batch_size = images_np.shape[0]
        swanlab_images = []
        
        for i in range(batch_size):
            img = images_np[i]
            
            # 如果是 torch.Tensor，转换为 numpy
            if isinstance(img, torch.Tensor):
                img = img.cpu().detach().numpy()
            
            # 确保是 numpy 数组
            img = np.array(img)
            
            # 处理不同的输入格式
            if len(img.shape) == 4:
                # 如果是 (1, C, H, W)，取第一个
                img = img[0]
            
            if len(img.shape) == 3:
                # (C, H, W) 格式，转换为 (H, W, C)
                if img.shape[0] in [1, 3]:
                    img = img.transpose(1, 2, 0)
                # 如果已经是 (H, W, C) 格式，保持不变
            elif len(img.shape) == 2:
                # (H, W) 格式，转换为 (H, W, 3) RGB
                img = np.stack([img, img, img], axis=-1)
            
            # 确保是 3 通道 RGB 格式
            if len(img.shape) == 3:
                if img.shape[2] == 1:
                    # 单通道，复制为 3 通道
                    img = np.repeat(img, 3, axis=2)
                elif img.shape[2] != 3:
                    # 如果不是 1 或 3 通道，取前 3 个通道或填充
                    if img.shape[2] > 3:
                        img = img[:, :, :3]
                    else:
                        # 填充到 3 通道
                        padding = np.zeros((img.shape[0], img.shape[1], 3 - img.shape[2]), dtype=img.dtype)
                        img = np.concatenate([img, padding], axis=2)
            
            # 确保值域在 [0, 1]
            img = np.clip(img, 0.0, 1.0)
            
            # 转换为 [0, 255] 的 uint8 格式（SwanLab 要求）
            if img.dtype != np.uint8:
                # 如果值域已经是 [0, 1]，直接乘以 255
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    # 如果值域是 [0, 255]，直接转换类型
                    img = img.astype(np.uint8)
            
            # 使用 swanlab.Image 封装图像
            try:
                swanlab_img = self.swanlab.Image(img)
                swanlab_images.append(swanlab_img)
            except Exception as e:
                print(f"警告: 创建 swanlab.Image 失败 (图像 {i}): {e}")
                print(f"  图像形状: {img.shape}, 数据类型: {img.dtype}")
                # 如果 swanlab.Image 不可用，尝试直接使用 numpy 数组
                swanlab_images.append(img)
        
        # 记录到 SwanLab
        if len(swanlab_images) > 0:
            try:
                self.swanlab.log({key: swanlab_images}, step=step)
            except Exception as e:
                print(f"错误: 记录 {key} 到 SwanLab 失败: {e}")
                raise

    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """训练批次结束时的回调"""
        # 每隔一定步数记录图像
        if trainer.global_step % self.log_images_every_n_steps == 0:
            # 获取模型输出（如果可用）
            if isinstance(outputs, dict) and "x_recon" in outputs:
                x = batch
                x_recon = outputs["x_recon"]
            elif hasattr(pl_module, "last_batch") and hasattr(pl_module, "last_recon"):
                x = pl_module.last_batch
                x_recon = pl_module.last_recon
            else:
                # 重新前向传播获取重建图像
                pl_module.eval()
                with torch.no_grad():
                    x_recon, _ = pl_module(batch)
                pl_module.train()
                x = batch
            
            # 选择前 n_samples 张图像
            n_samples = min(self.n_samples, x.shape[0])
            images = torch.cat([
                x[:n_samples],
                x_recon[:n_samples]
            ], dim=0)
            
            # 归一化到 [0, 1]
            images = (images + 1) / 2
            images = torch.clamp(images, 0.0, 1.0)
            
            # 记录到 SwanLab
            self._log_images_to_swanlab(
                images,
                key="train/reconstruction",
                step=trainer.global_step
            )
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """训练 epoch 结束时的回调"""
        # 可以在这里记录 epoch 级别的统计信息
        pass
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """验证批次结束时的回调"""
        # 验证时通常不需要每个 batch 都记录图像
        pass
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """验证 epoch 结束时的回调"""
        # 只在有 SwanLab Logger 时记录图像
        if not self._is_swanlab_logger(trainer.logger):
            return
        
        # 每隔一定 epoch 记录图像
        if trainer.current_epoch % self.log_images_every_n_epochs == 0:
            # 获取验证数据
            if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                try:
                    val_loader = trainer.datamodule.val_dataloader()
                    if val_loader is not None and len(val_loader) > 0:
                        batch = next(iter(val_loader))
                        # 确保 batch 在正确的设备上
                        if isinstance(batch, torch.Tensor):
                            batch = batch.to(pl_module.device)
                        elif isinstance(batch, (list, tuple)):
                            batch = [b.to(pl_module.device) if isinstance(b, torch.Tensor) else b for b in batch]
                            batch = batch[0] if len(batch) > 0 else None
                        
                        if batch is not None:
                            pl_module.eval()
                            with torch.no_grad():
                                x_recon, _ = pl_module(batch)
                            pl_module.train()
                            
                            # 选择前 n_samples 张图像
                            n_samples = min(self.n_samples, batch.shape[0])
                            images = torch.cat([
                                batch[:n_samples],
                                x_recon[:n_samples]
                            ], dim=0)
                            
                            # 归一化到 [0, 1]
                            images = (images + 1) / 2
                            images = torch.clamp(images, 0.0, 1.0)
                            
                            # 记录到 SwanLab
                            self._log_images_to_swanlab(
                                images,
                                key="val/reconstruction",
                                step=trainer.global_step
                            )
                except Exception as e:
                    print(f"警告: 验证 epoch 结束时记录图像失败: {e}")
    
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """测试批次结束时的回调"""
        # 测试时通常不需要每个 batch 都记录图像
        pass
    
    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """测试 epoch 结束时的回调"""
        # 只在有 SwanLab Logger 时记录图像
        if not self._is_swanlab_logger(trainer.logger):
            return
        
        # 获取测试数据
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            try:
                test_loader = trainer.datamodule.test_dataloader()
                if test_loader is not None and len(test_loader) > 0:
                    batch = next(iter(test_loader))
                    # 确保 batch 在正确的设备上
                    if isinstance(batch, torch.Tensor):
                        batch = batch.to(pl_module.device)
                    elif isinstance(batch, (list, tuple)):
                        batch = [b.to(pl_module.device) if isinstance(b, torch.Tensor) else b for b in batch]
                        batch = batch[0] if len(batch) > 0 else None
                    
                    if batch is not None:
                        pl_module.eval()
                        with torch.no_grad():
                            x_recon, _ = pl_module(batch)
                        
                        # 选择前 n_samples 张图像
                        n_samples = min(self.n_samples, batch.shape[0])
                        images = torch.cat([
                            batch[:n_samples],
                            x_recon[:n_samples]
                        ], dim=0)
                        
                        # 归一化到 [0, 1]
                        images = (images + 1) / 2
                        images = torch.clamp(images, 0.0, 1.0)
                        
                        # 记录到 SwanLab
                        self._log_images_to_swanlab(
                            images,
                            key="test/reconstruction",
                            step=trainer.global_step
                        )
            except Exception as e:
                print(f"警告: 测试 epoch 结束时记录图像失败: {e}")
    
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """训练开始时的回调"""
        if self.swanlab_available and self._is_swanlab_logger(trainer.logger):
            print("SwanLab 回调已启用，将记录训练指标和图像")
    
    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """训练结束时的回调"""
        if self.swanlab_available and self._is_swanlab_logger(trainer.logger):
            print("训练完成，所有指标已记录到 SwanLab")


if __name__ == "__main__":
    # 测试代码
    print("SwanLab 回调模块测试")
    
    callback = SwanLabCallback(
        log_images_every_n_steps=500,
        log_images_every_n_epochs=1,
        n_samples=8,
    )
    
    print(f"SwanLab 可用: {callback.swanlab_available}")
    print("回调模块加载成功！")

