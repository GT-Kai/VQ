"""
SwanLab 回调函数
用于记录训练/验证/测试指标和图像到 SwanLab
"""

from typing import Optional, Any
import os
import uuid

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.loggers.logger import LoggerCollection

import numpy as np


import swanlab  # type: ignore
SWANLAB_AVAILABLE = True



class SwanLabCallback(Callback):
    """SwanLab 回调函数，用于记录训练指标和图像"""

    def __init__(
        self,
        log_images_every_n_steps: int = 500,
        log_images_every_n_epochs: int = 1,
        n_samples: int = 8,
        image_output_dir: str = "swanlab_images",
    ):
        """
        Args:
            log_images_every_n_steps: 训练时每隔多少步记录一次图像
            log_images_every_n_epochs: 验证/测试时每隔多少个 epoch 记录一次图像
            n_samples: 每次记录的图像数量（原图 + 重建图 各 n_samples）
            image_output_dir: 临时保存图像的目录
        """
        super().__init__()
        self.log_images_every_n_steps = log_images_every_n_steps
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.n_samples = n_samples
        self.image_output_dir = image_output_dir

        self.swanlab = swanlab
        self.swanlab_available = SWANLAB_AVAILABLE

        if not self.swanlab_available:
            print("[SwanLabCallback] 未检测到 swanlab 包，图像不会被上传到 SwanLab。")

    # ---------------------- 一些通用小工具 ----------------------

    def _is_swanlab_logger(self, logger) -> bool:
        """检查是否为 SwanLab Logger（不依赖 LoggerCollection 类型，兼容不同 lightning 版本）"""
        if logger is None:
            return False

        # 单个 logger：看类名里有没有 swanlab
        name = type(logger).__name__
        if name == "SwanLabLogger" or "swanlab" in name.lower():
            return True

        # 可能是某种 logger 集合：尝试从常见属性里拿内部 loggers
        candidates = []
        for attr in ("logger_iterable", "_logger_iterable", "loggers"):
            if hasattr(logger, attr):
                candidates = getattr(logger, attr)
                break

        # 如果上面没拿到，再尝试 logger 本身可迭代
        if not candidates:
            try:
                candidates = list(logger)
            except TypeError:
                candidates = []

        for lg in candidates:
            lg_name = type(lg).__name__
            if lg_name == "SwanLabLogger" or "swanlab" in lg_name.lower():
                return True

        return False


    def _extract_x_from_batch(self, batch: Any, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        从 batch 中尽可能提取出图像张量 x。
        支持：
        - Tensor
        - (x, y) / [x, y]
        - {"x": x, ...} / {"image": x, ...} / {"images": x, ...} / {"input": x, ...} / {"inputs": x, ...}
        """
        x = None

        if isinstance(batch, torch.Tensor):
            x = batch
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            # 假设第一个元素是 x
            if isinstance(batch[0], torch.Tensor):
                x = batch[0]
        elif isinstance(batch, dict):
            for key in ["x", "image", "images", "input", "inputs"]:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    x = batch[key]
                    break

        if x is None:
            return None

        if device is not None:
            x = x.to(device)

        return x

    def _extract_x_recon_from_output(self, out: Any) -> Optional[torch.Tensor]:
        """
        从模型输出中提取 x_recon：
        - Tensor
        - (x_recon, ...)
        - {"x_recon": x_recon, ...}
        """
        if out is None:
            return None

        if isinstance(out, torch.Tensor):
            return out

        if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            return out[0]

        if isinstance(out, dict):
            # 优先找常见 key
            for key in ["x_recon", "recon", "reconstruction"]:
                if key in out and isinstance(out[key], torch.Tensor):
                    return out[key]

            # 次选：字典中第一个 Tensor
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v

        return None

    def _run_model_and_get_recon(
        self,
        pl_module: pl.LightningModule,
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """在 eval 模式下前向一次，取出重建图像"""
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            out = pl_module(x)
        if was_training:
            pl_module.train()

        x_recon = self._extract_x_recon_from_output(out)
        return x_recon

    # ---------------------- 图像保存与上传 ----------------------

    def _normalize_to_uint8(self, img: np.ndarray) -> np.ndarray:
        """
        将任意数值图像转为 [0,255] 的 uint8。
        自动处理 0-1 和 0-255 两种常见范围。
        """
        img = img.astype(np.float32)

        max_val = img.max() if img.size > 0 else 1.0
        if max_val > 1.0:
            # 假定是 0-255
            img = img / 255.0

        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)
        return img

    def _save_images_locally(self, images: np.ndarray, key: str, step: int):
        """将图像保存到本地，并返回保存路径列表"""
        from PIL import Image

        os.makedirs(self.image_output_dir, exist_ok=True)
        saved_paths = []

        for i, img in enumerate(images):
            try:
                # 确保是 numpy 数组
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()

                # 处理 batch 维度
                if len(img.shape) == 4:
                    img = img[0]

                # 通道维度处理
                if len(img.shape) == 3:
                    # (C, H, W) -> (H, W, C)
                    if img.shape[0] in [1, 3]:
                        img = img.transpose(1, 2, 0)
                elif len(img.shape) == 2:
                    # (H, W) -> (H, W, 3)
                    img = np.stack([img] * 3, axis=-1)

                # 确保是 3 通道
                if len(img.shape) == 3:
                    if img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=2)
                    elif img.shape[2] < 3:
                        pad = 3 - img.shape[2]
                        img = np.pad(img, ((0, 0), (0, 0), (0, pad)))
                    elif img.shape[2] > 3:
                        img = img[:, :, :3]

                # 归一化为 uint8
                img = self._normalize_to_uint8(img)

                # 调整大小（最大 256x256）
                if img.shape[0] > 256 or img.shape[1] > 256:
                    pil_img = Image.fromarray(img)
                    pil_img.thumbnail((256, 256), Image.LANCZOS)
                    img = np.array(pil_img)

                filename = f"{key.replace('/', '_')}_step{step}_{i}_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join(self.image_output_dir, filename)
                Image.fromarray(img).save(filepath)
                saved_paths.append(filepath)
            except Exception as e:
                print(f"[SwanLabCallback] 警告: 保存图像 {i} 到本地失败: {e}")

        return saved_paths

    def _upload_images_in_background(self, key: str, step: int, image_paths: list):
        """在后台线程中上传图像到 SwanLab，然后删除本地文件"""
        import threading

        if not (self.swanlab_available and self.swanlab is not None):
            return

        def upload():
            try:
                swanlab_images = []
                for img_path in image_paths:
                    try:
                        swanlab_img = self.swanlab.Image(img_path)
                        swanlab_images.append(swanlab_img)
                    except Exception as e:
                        print(f"[SwanLabCallback] 警告: 创建 swanlab.Image 失败 ({img_path}): {e}")

                if swanlab_images:
                    self.swanlab.log({key: swanlab_images}, step=step)
                    print(f"[SwanLabCallback] 成功上传 {len(swanlab_images)} 张图像到 SwanLab (step={step})")

                # 上传后删除临时文件
                for img_path in image_paths:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except Exception as e:
                        print(f"[SwanLabCallback] 警告: 删除临时文件 {img_path} 失败: {e}")
            except Exception as e:
                print(f"[SwanLabCallback] 警告: 上传图像到 SwanLab 失败: {e}")

        # 后台线程上传（如果进程非常快结束，最后一批图可能来不及传完）
        thread = threading.Thread(target=upload, daemon=True)
        thread.start()

    def _log_images_to_swanlab(self, images: torch.Tensor, key: str, step: int):
        """将图像记录到 SwanLab（限制张数、保存本地、后台上传）"""
        if not (self.swanlab_available and self.swanlab is not None):
            return

        try:
            if isinstance(images, torch.Tensor):
                images_np = images.detach().cpu().numpy()
            else:
                images_np = np.array(images)

            if images_np.ndim < 3:
                print("[SwanLabCallback] 警告: images 维度过低，跳过记录。")
                return

            # 最多 4 张
            max_images = min(4, images_np.shape[0])
            images_np = images_np[:max_images]

            saved_paths = self._save_images_locally(images_np, key, step)
            if saved_paths:
                self._upload_images_in_background(key, step, saved_paths)
        except Exception as e:
            print(f"[SwanLabCallback] 警告: 记录 {key} 到 SwanLab 时出错: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------- Lightning 回调 ----------------------

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.swanlab_available and self._is_swanlab_logger(trainer.logger):
            print("[SwanLabCallback] SwanLab 回调已启用，将记录训练/验证/测试图像。")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """训练批次结束时：按步数间隔记录重建图像"""
        if not (self.swanlab_available and self._is_swanlab_logger(trainer.logger)):
            return

        if self.log_images_every_n_steps <= 0:
            return

        # global_step 从 0 开始，这里可以按需调整策略
        if trainer.global_step % self.log_images_every_n_steps != 0:
            return

        # 从 batch 提取输入 x（不需要显式 to(device)，Lightning 已经处理过）
        x = self._extract_x_from_batch(batch)
        if x is None or not isinstance(x, torch.Tensor):
            return

        # 尝试从 outputs 里直接拿重建图
        x_recon = None
        if isinstance(outputs, dict) and "x_recon" in outputs and isinstance(outputs["x_recon"], torch.Tensor):
            x_recon = outputs["x_recon"]
        else:
            x_recon = self._extract_x_recon_from_output(outputs)

        # 如果 outputs 中拿不到，就再跑一次前向
        if x_recon is None:
            x_recon = self._run_model_and_get_recon(pl_module, x)

        if x_recon is None or not isinstance(x_recon, torch.Tensor):
            return

        n_samples = min(self.n_samples, x.shape[0], x_recon.shape[0])
        if n_samples <= 0:
            return

        images = torch.cat([x[:n_samples], x_recon[:n_samples]], dim=0)

        # 假设输入范围是 [-1,1]，映射到 [0,1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)

        self._log_images_to_swanlab(images, key="train/reconstruction", step=trainer.global_step)

    def _log_epoch_images_from_loader(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        loader,
        key_prefix: str,
    ):
        """从给定 dataloader 中取一个 batch，跑一遍模型并记录图像"""
        if loader is None:
            return

        # 有些 DataLoader 实现没有 __len__，这里 try 一下
        try:
            if len(loader) == 0:  # type: ignore
                return
        except TypeError:
            pass

        try:
            batch = next(iter(loader))
        except StopIteration:
            return

        batch = self._move_batch_to_device(batch, pl_module.device)
        x = self._extract_x_from_batch(batch)
        if x is None or not isinstance(x, torch.Tensor):
            return

        x_recon = self._run_model_and_get_recon(pl_module, x)
        if x_recon is None or not isinstance(x_recon, torch.Tensor):
            return

        n_samples = min(self.n_samples, x.shape[0], x_recon.shape[0])
        if n_samples <= 0:
            return

        images = torch.cat([x[:n_samples], x_recon[:n_samples]], dim=0)
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)

        step = trainer.global_step
        key = f"{key_prefix}/reconstruction"
        self._log_images_to_swanlab(images, key=key, step=step)

    def _move_batch_to_device(self, batch: Any, device: torch.device):
        """递归地把 batch 里的 Tensor 都挪到指定 device 上（用于 val/test）"""

        if isinstance(batch, torch.Tensor):
            return batch.to(device)

        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(b, device) for b in batch)

        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}

        return batch

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """每隔若干个 epoch 在验证集上记录一次重建图像"""
        if not (self.swanlab_available and self._is_swanlab_logger(trainer.logger)):
            return

        if self.log_images_every_n_epochs <= 0:
            return

        if trainer.current_epoch % self.log_images_every_n_epochs != 0:
            return

        if not hasattr(trainer, "datamodule") or trainer.datamodule is None:
            return

        try:
            val_loader = trainer.datamodule.val_dataloader()
        except Exception as e:
            print(f"[SwanLabCallback] 警告: 获取 val_dataloader 失败: {e}")
            return

        self._log_epoch_images_from_loader(trainer, pl_module, val_loader, key_prefix="val")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """在测试集上记录一次重建图像"""
        if not (self.swanlab_available and self._is_swanlab_logger(trainer.logger)):
            return

        if not hasattr(trainer, "datamodule") or trainer.datamodule is None:
            return

        try:
            test_loader = trainer.datamodule.test_dataloader()
        except Exception as e:
            print(f"[SwanLabCallback] 警告: 获取 test_dataloader 失败: {e}")
            return

        self._log_epoch_images_from_loader(trainer, pl_module, test_loader, key_prefix="test")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.swanlab_available and self._is_swanlab_logger(trainer.logger):
            print("[SwanLabCallback] 训练完成，所有可用的图像已尝试上传到 SwanLab。")


if __name__ == "__main__":
    # 简单测试：检查回调是否能正常构造
    print("SwanLab 回调模块测试")
    callback = SwanLabCallback(
        log_images_every_n_steps=500,
        log_images_every_n_epochs=1,
        n_samples=8,
    )
    print(f"SwanLab 可用: {callback.swanlab_available}")
    print("回调模块加载成功！")
