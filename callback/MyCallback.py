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
            images_np = images.cpu().numpy()
        else:
            images_np = images
        
        # 转换图像格式: (B, C, H, W) -> (B, H, W, C)
        if images_np.shape[1] == 3 or images_np.shape[1] == 1:  # (B, C, H, W)
            images_list = []
            for i in range(len(images_np)):
                img = images_np[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                # 确保值域在 [0, 1]
                img = np.clip(img, 0, 1)
                images_list.append(img)
        else:  # 已经是 (B, H, W, C) 格式
            images_list = [np.clip(images_np[i], 0, 1) for i in range(len(images_np))]
            
        # 记录到 SwanLab
        self.swanlab.log({key: images_list}, step=step)
    
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
            images = torch.clamp(images, 0, 1)
            
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
                val_loader = trainer.datamodule.val_dataloader()
                if len(val_loader) > 0:
                    batch = next(iter(val_loader))
                    batch = batch.to(pl_module.device)
                    
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
                    images = torch.clamp(images, 0, 1)
                    
                    # 记录到 SwanLab
                    self._log_images_to_swanlab(
                        images,
                        key="val/reconstruction",
                        step=trainer.global_step
                    )
    
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
            test_loader = trainer.datamodule.test_dataloader()
            if len(test_loader) > 0:
                batch = next(iter(test_loader))
                batch = batch.to(pl_module.device)
                
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
                images = torch.clamp(images, 0, 1)
                
                # 记录到 SwanLab
                self._log_images_to_swanlab(
                    images,
                    key="test/reconstruction",
                    step=trainer.global_step
                )
    
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

