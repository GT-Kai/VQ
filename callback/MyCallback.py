# callback/MyCallback.py

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

import matplotlib.pyplot as plt


# 尝试导入 swanlab
try:
    import swanlab  # type: ignore
    SWANLAB_AVAILABLE = True
except ImportError:
    swanlab = None  # type: ignore
    SWANLAB_AVAILABLE = False


class SwanLabCallback(Callback):
    """专为 VQVAEModel 定制的 SwanLab 图像回调（使用 Matplotlib 记录）
    
    注意：此 callback 会在 on_train_start 时手动调用 swanlab.init()，
    避免通过 SwanLabLogger 导致 DataPorter 被重复初始化。
    """

    def __init__(
        self,
        log_images_every_n_steps: int = 500,
        log_images_every_n_epochs: int = 1,
        n_samples: int = 8,
        project: str = "vq-vae-anime",
        experiment_name: str = "vqvae-baseline",
        description: str = "VQ-VAE baseline experiment on anime faces",
    ):
        """
        Args:
            log_images_every_n_steps: 训练时每隔多少步记录一次图像
            log_images_every_n_epochs: 验证/测试时每隔多少个 epoch 记录一次图像
            n_samples: 每次记录的样本数（原图 + 重建图 各 n_samples）
            project: SwanLab 项目名称
            experiment_name: SwanLab 实验名称
            description: SwanLab 实验描述
        """
        super().__init__()
        self.log_images_every_n_steps = log_images_every_n_steps
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.n_samples = n_samples
        
        # SwanLab 配置
        self.project = project
        self.experiment_name = experiment_name
        self.description = description
        
        self.swanlab = swanlab
        self.swanlab_available = SWANLAB_AVAILABLE
        self.swanlab_initialized = False

        if not self.swanlab_available:
            print("[SwanLabCallback] 未检测到 swanlab 包，图像记录将被跳过。")

    # ------------------------------------------------------------------
    # 工具：初始化 SwanLab（仅一次）
    # ------------------------------------------------------------------
    def _ensure_swanlab_initialized(self):
        """确保 SwanLab 被正确初始化（仅初始化一次）"""
        if not self.swanlab_available:
            return
        
        if self.swanlab_initialized:
            return
        
        try:
            # 检查是否已有活跃的 swanlab run
            # 如果没有，则初始化一个新的
            if not hasattr(self.swanlab, 'get_run') or self.swanlab.get_run() is None:
                self.swanlab.init(
                    project=self.project,
                    experiment_name=self.experiment_name,
                    description=self.description,
                    config={
                        "tags": ["vqvae", "anime", "baseline"]
                    }
                )
            
            self.swanlab_initialized = True
            print("[SwanLabCallback] SwanLab 已初始化，将记录训练/验证/测试图像。")
        except Exception as e:
            print(f"[SwanLabCallback] 初始化 SwanLab 失败: {e}")
            self.swanlab_available = False

    # ------------------------------------------------------------------
    # 核心：用 Matplotlib 画原图 + 重建图，并用 swanlab.Image(plt) 记录
    # ------------------------------------------------------------------
    def _plot_and_log(self, x: torch.Tensor, x_recon: torch.Tensor, key: str, step: int):
        """
        x, x_recon: (B, 3, H, W)，值域假设为 [-1, 1]
        用 matplotlib 画成 2 行 B 列的子图，然后:

            swanlab.log({key: swanlab.Image(plt)})

        就是你文档里的 Matplotlib 示例用法。
        """
        if not (self.swanlab_available and self.swanlab is not None):
            return

        # 限制最多 4 个样本，避免图太大
        B = min(self.n_samples, x.size(0), x_recon.size(0), 4)
        if B <= 0:
            return

        # 拿前 B 个样本，搬到 CPU
        x = x[:B].detach().to("cpu")
        x_recon = x_recon[:B].detach().to("cpu")

        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        x_recon = (x_recon + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        x_recon = torch.clamp(x_recon, 0.0, 1.0)

        # 创建画布：2 行 B 列
        fig, axes = plt.subplots(2, B, figsize=(3 * B, 6))

        # 当 B = 1 时，axes 不是二维数组，特殊处理一下
        if B == 1:
            axes = [[axes[0]], [axes[1]]]

        for i in range(B):
            # 原图
            img = x[i]
            # (C,H,W) -> (H,W,C)
            if img.dim() == 3 and img.size(0) in (1, 3):
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = img.numpy()

            axes[0][i].imshow(img_np)
            axes[0][i].set_title(f"Input {i}")
            axes[0][i].axis("off")

            # 重建图
            rec = x_recon[i]
            if rec.dim() == 3 and rec.size(0) in (1, 3):
                rec_np = rec.permute(1, 2, 0).numpy()
            else:
                rec_np = rec.numpy()

            axes[1][i].imshow(rec_np)
            axes[1][i].set_title(f"Recon {i}")
            axes[1][i].axis("off")

        plt.suptitle(f"{key} (step {step})")

        # 关键：用 Matplotlib 的 plt 对象创建 swanlab.Image
        img_obj = self.swanlab.Image(plt)
        self.swanlab.log({key: img_obj}, step=step)

        # 关闭图像，防止内存泄露
        plt.close(fig)

    # ------------------------------------------------------------------
    # Lightning 回调
    # ------------------------------------------------------------------
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """训练开始时初始化 SwanLab"""
        if self.swanlab_available:
            self._ensure_swanlab_initialized()
            print("[SwanLabCallback] 训练开始，SwanLab 已就绪。")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """
        训练阶段：每隔 log_images_every_n_steps 重新跑一遍 forward，
        拿到 (x_recon, info)，画 Matplotlib 图。
        """
        if not self.swanlab_available:
            return

        if self.log_images_every_n_steps <= 0:
            return

        if trainer.global_step % self.log_images_every_n_steps != 0:
            return

        # 你的 DataLoader: batch 就是 x，形状 (B, 3, 256, 256)
        if not isinstance(batch, torch.Tensor):
            return

        x = batch.to(pl_module.device)

        # 重新跑一次 forward 拿到重建图像
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            x_recon, _ = pl_module(x)
        if was_training:
            pl_module.train()

        self._plot_and_log(x, x_recon, key="train/reconstruction", step=trainer.global_step)

    # ---------------------- 验证 / 测试 公共逻辑 ----------------------
    def _get_first_batch(self, loader, device):
        """从 dataloader 中取一个 batch，并搬到指定 device 上"""
        if loader is None:
            return None

        try:
            batch = next(iter(loader))
        except Exception:
            return None

        if isinstance(batch, torch.Tensor):
            return batch.to(device)

        return None

    def _log_epoch_images_from_loader(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loader,
        key_prefix: str,
    ):
        x = self._get_first_batch(loader, pl_module.device)
        if x is None:
            return

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            x_recon, _ = pl_module(x)
        if was_training:
            pl_module.train()

        self._plot_and_log(x, x_recon, key=f"{key_prefix}/reconstruction", step=trainer.global_step)

    # ---------------------- 验证 / 测试 回调 ----------------------
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """每隔若干个 epoch 在验证集上记录一次重建图像"""
        if not self.swanlab_available:
            return

        if self.log_images_every_n_epochs <= 0:
            return

        # 跳过 sanity check
        if getattr(trainer, "sanity_checking", False):
            return

        if trainer.current_epoch % self.log_images_every_n_epochs != 0:
            return

        if not hasattr(trainer, "datamodule") or trainer.datamodule is None:
            return

        val_loader = trainer.datamodule.val_dataloader()
        self._log_epoch_images_from_loader(trainer, pl_module, val_loader, key_prefix="val")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """在测试集上记录一次重建图像"""
        if not self.swanlab_available:
            return

        if not hasattr(trainer, "datamodule") or trainer.datamodule is None:
            return

        test_loader = trainer.datamodule.test_dataloader()
        self._log_epoch_images_from_loader(trainer, pl_module, test_loader, key_prefix="test")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """训练结束时清理"""
        if self.swanlab_available and self.swanlab_initialized:
            try:
                # 确保 SwanLab run 被正确关闭
                if hasattr(self.swanlab, 'finish'):
                    self.swanlab.finish()
            except Exception as e:
                print(f"[SwanLabCallback] 关闭 SwanLab 时出错: {e}")
            print("[SwanLabCallback] 训练结束。")


if __name__ == "__main__":
    cb = SwanLabCallback()
    print("SwanLabCallback 加载成功，swanlab_available =", cb.swanlab_available)
