"""
VQ-VAE / VQ-GAN 模型模块
实现基于 PyTorch Lightning 的完整 VQ-VAE 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import lightning.pytorch as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np


class ResBlock(nn.Module):
    """ResNet 风格的残差块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        activation: str = "relu",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.activation = activation
        
        # 归一化层
        if norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        else:  # batch
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:  # silu
            self.act = nn.SiLU(inplace=True)
        
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 残差连接（如果通道数不同，需要 1x1 卷积）
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        
        return out + residual


class Encoder(nn.Module):
    """VQ-VAE Encoder
    
    输入: (B, 3, 256, 256) RGB 图像，值域 [-1, 1]
    输出: (B, C_latent, 16, 16) 特征图
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 256,
        channel_multipliers: Tuple[int, ...] = (128, 256, 256, 256),
        num_res_blocks: int = 2,
        norm_type: str = "group",
        activation: str = "relu",
    ):
        super().__init__()
        self.latent_channels = latent_channels
        
        # 输入层
        self.conv_in = nn.Conv2d(in_channels, channel_multipliers[0], kernel_size=3, padding=1)
        
        # 4 个下采样 stage: 256 → 128 → 64 → 32 → 16
        self.stages = nn.ModuleList()
        in_ch = channel_multipliers[0]
        
        for i, out_ch in enumerate(channel_multipliers):
            stage = nn.ModuleList()
            
            # 下采样卷积（stride=2）
            stage.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            
            # ResBlocks
            for _ in range(num_res_blocks):
                stage.append(ResBlock(out_ch, out_ch, norm_type, activation))
            
            self.stages.append(stage)
            in_ch = out_ch
        
        # 输出层：映射到 latent_channels
        self.conv_out = nn.Conv2d(
            channel_multipliers[-1],
            latent_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (B, 3, 256, 256)
        x = self.conv_in(x)
        
        # 4 个下采样 stage
        for stage in self.stages:
            # 下采样
            x = stage[0](x)
            # ResBlocks
            for res_block in stage[1:]:
                x = res_block(x)
        
        # 输出层
        x = self.conv_out(x)
        # 输出: (B, C_latent, 16, 16)
        return x


class Decoder(nn.Module):
    """VQ-VAE Decoder
    
    输入: (B, C_latent, 16, 16) 量化后的特征图
    输出: (B, 3, 256, 256) RGB 图像，值域 [-1, 1]
    """
    
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 256,
        channel_multipliers: Tuple[int, ...] = (256, 256, 256, 128),
        num_res_blocks: int = 2,
        norm_type: str = "group",
        activation: str = "relu",
    ):
        super().__init__()
        
        # 输入层：从 latent_channels 映射到第一个通道数
        self.conv_in = nn.Conv2d(
            latent_channels,
            channel_multipliers[0],
            kernel_size=3,
            padding=1
        )
        
        # 4 个上采样 stage: 16 → 32 → 64 → 128 → 256
        self.stages = nn.ModuleList()
        in_ch = channel_multipliers[0]
        
        for i, out_ch in enumerate(channel_multipliers):
            stage = nn.ModuleList()
            
            # ResBlocks
            for _ in range(num_res_blocks):
                stage.append(ResBlock(in_ch, in_ch, norm_type, activation))
            
            # 上采样（使用最近邻插值 + 卷积）
            # 所有 stage 都需要上采样以实现 16 → 32 → 64 → 128 → 256
            stage.append(nn.Upsample(scale_factor=2, mode="nearest"))
            stage.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            
            self.stages.append(stage)
            in_ch = out_ch
        
        # 输出层：映射到 RGB，使用 tanh 激活
        self.conv_out = nn.Sequential(
            nn.Conv2d(channel_multipliers[-1], 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (B, C_latent, 16, 16)
        x = self.conv_in(x)
        
        # 4 个上采样 stage
        for stage in self.stages:
            # 按顺序执行 stage 中的每个模块
            for module in stage:
                x = module(x)
        
        # 输出层
        x = self.conv_out(x)
        # 输出: (B, 3, 256, 256)
        return x


class VectorQuantizerEMA(nn.Module):
    """EMA 版向量量化器（类似 VQ-VAE v2 / VQ-GAN）"""
    
    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Codebook（可学习的 embedding）
        self.register_buffer("embedding", torch.randn(embedding_dim, num_embeddings))
        self.embedding.data.normal_() / embedding_dim
        
        # EMA 统计量
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embedding_avg", self.embedding.clone())
        
        # 用于记录 code 使用情况
        self.register_buffer("usage_count", torch.zeros(num_embeddings))
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            inputs: (B, C, H, W) 编码器输出特征图
        
        Returns:
            quantized: (B, C, H, W) 量化后的特征图
            vq_loss: 向量量化损失
            info: 包含损失和统计信息的字典
        """
        # 展平空间维度: (B, C, H, W) -> (B*H*W, C)
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # 计算与 codebook 的距离
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=0, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding)
        )
        
        # 找到最近的 code
        encoding_indices = torch.argmin(distances, dim=1)
        
        # 量化：使用最近的 embedding
        quantized = self.embedding[:, encoding_indices].t()
        quantized = quantized.reshape(inputs.shape)
        
        # EMA 更新（仅在训练时）
        if self.training:
            # 统计每个 code 的使用次数
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            
            # 更新 cluster_size（EMA）
            self.cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            
            # 更新 embedding_avg（EMA）
            n = encodings.sum(0)
            embedding_sum = flat_input.t() @ encodings
            self.embedding_avg.mul_(self.decay).add_(
                embedding_sum, alpha=1 - self.decay
            )
            
            # 更新 codebook（使用 EMA 统计量）
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embedding.data.copy_(
                self.embedding_avg / cluster_size.unsqueeze(0)
            )
            
            # 更新使用计数（用于统计）
            self.usage_count.add_(encodings.sum(0))
        
        # 计算损失
        # VQ loss: 更新 codebook（使用 stop_gradient）
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        # Commitment loss: 约束 encoder 输出
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = self.commitment_cost * q_latent_loss
        
        # 总 VQ 损失
        vq_loss = e_latent_loss + commitment_loss
        
        # 使用 straight-through estimator（梯度直通）
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算 perplexity（困惑度，用于监控 code 使用情况）
        avg_probs = self.cluster_size / (self.cluster_size.sum() + self.epsilon)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.epsilon)))
        
        # 计算 code 使用率（有多少 code 被使用）
        used_codes = (self.cluster_size > self.epsilon).float().sum()
        code_usage_rate = used_codes / self.num_embeddings
        
        info = {
            "vq_loss": vq_loss,
            "e_latent_loss": e_latent_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "code_usage_rate": code_usage_rate,
            "encoding_indices": encoding_indices.reshape(inputs.shape[0], -1),
        }
        
        return quantized, vq_loss, info


class LPIPS(nn.Module):
    """简化的 LPIPS（感知损失）实现
    
    使用预训练的 VGG 特征提取器
    """
    
    def __init__(self):
        super().__init__()
        # try:
        #     # 尝试使用 torch.hub 加载预训练模型
        #     vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        # except Exception:
        #     # 如果失败，使用 torchvision 直接加载
        #     from torchvision.models import vgg16, VGG16_Weights
        #     vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        
        vgg_features = vgg.features
        
        # 提取不同层的特征
        self.slice1 = vgg_features[:4]   # relu1_2
        self.slice2 = vgg_features[4:9]  # relu2_2
        self.slice3 = vgg_features[9:16] # relu3_3
        self.slice4 = vgg_features[16:23] # relu4_3
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 归一化（ImageNet 统计量）
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y: (B, 3, H, W) 图像，值域 [-1, 1]
        
        Returns:
            loss: 感知损失标量
        """
        # 归一化到 [0, 1] 然后标准化
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        y = (y + 1) / 2
        
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # 提取特征并计算 L2 距离
        loss = 0.0
        weights = [0.1, 0.1, 1.0, 1.0]  # 不同层的权重
        
        # 累积地通过每个 slice，保存中间特征
        x_feat = x
        y_feat = y
        
        for i, (slice_fn, weight) in enumerate(zip(
            [self.slice1, self.slice2, self.slice3, self.slice4],
            weights
        )):
            x_feat = slice_fn(x_feat)
            y_feat = slice_fn(y_feat)
            
            # L2 距离
            loss += weight * F.mse_loss(x_feat, y_feat)
        
        return loss


class VQVAEModel(pl.LightningModule):
    """VQ-VAE 模型（PyTorch Lightning Module）"""
    
    def __init__(
        self,
        # 模型结构参数
        in_channels: int = 3,
        latent_channels: int = 256,
        encoder_channels: Tuple[int, ...] = (128, 256, 256, 256),
        decoder_channels: Tuple[int, ...] = (256, 256, 256, 128),
        num_res_blocks: int = 2,
        # 量化器参数
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        # 损失权重
        lambda_rec: float = 1.0,
        lambda_vq: float = 1.0,
        lambda_commit: float = 0.25,
        lambda_perc: float = 0.5,
        lambda_gan: float = 0.0,  # VQ-GAN 扩展，默认关闭
        use_lpips: bool = True,
        # 优化器参数
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        # 其他
        log_every_n_steps: int = 100,
        log_images_every_n_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 模型组件
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            channel_multipliers=encoder_channels,
            num_res_blocks=num_res_blocks,
        )
        
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )
        
        self.decoder = Decoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            channel_multipliers=decoder_channels,
            num_res_blocks=num_res_blocks,
        )
        
        # 感知损失（可选）
        if use_lpips:
            self.lpips = LPIPS()
        else:
            self.lpips = None
        
        # 损失权重
        self.lambda_rec = lambda_rec
        self.lambda_vq = lambda_vq
        self.lambda_commit = lambda_commit
        self.lambda_perc = lambda_perc
        self.lambda_gan = lambda_gan
        
        # 训练参数
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.log_every_n_steps = log_every_n_steps
        self.log_images_every_n_steps = log_images_every_n_steps
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
        
        Returns:
            x_recon: (B, 3, H, W) 重建图像
            info: 包含损失和统计信息的字典
        """
        # Encode
        z_e = self.encoder(x)
        
        # Quantize
        z_q, vq_loss, quantizer_info = self.quantizer(z_e)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        # 计算重建损失
        recon_loss = F.l1_loss(x_recon, x)
        
        # 计算感知损失（如果启用）
        perc_loss = torch.tensor(0.0, device=x.device)
        if self.lpips is not None and self.lambda_perc > 0:
            perc_loss = self.lpips(x, x_recon)
        
        # 总损失
        total_loss = (
            self.lambda_rec * recon_loss
            + self.lambda_vq * quantizer_info["e_latent_loss"]
            + self.lambda_commit * quantizer_info["commitment_loss"]
            + self.lambda_perc * perc_loss
        )
        
        info = {
            "recon_loss": recon_loss,
            "vq_loss": quantizer_info["vq_loss"],
            "e_latent_loss": quantizer_info["e_latent_loss"],
            "commitment_loss": quantizer_info["commitment_loss"],
            "perc_loss": perc_loss,
            "total_loss": total_loss,
            "perplexity": quantizer_info["perplexity"],
            "code_usage_rate": quantizer_info["code_usage_rate"],
        }
        
        return x_recon, info
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        x = batch
        
        # 前向传播
        x_recon, info = self.forward(x)
        
        # 记录损失
        self.log("train/recon_loss", info["recon_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/vq_loss", info["vq_loss"], on_step=True, on_epoch=True)
        self.log("train/commitment_loss", info["commitment_loss"], on_step=True, on_epoch=True)
        self.log("train/perc_loss", info["perc_loss"], on_step=True, on_epoch=True)
        self.log("train/total_loss", info["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", info["perplexity"], on_step=True, on_epoch=True)
        self.log("train/code_usage_rate", info["code_usage_rate"], on_step=True, on_epoch=True)
        
        return info["total_loss"]
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        """验证步骤"""
        x = batch
        
        # 前向传播
        x_recon, info = self.forward(x)
        
        # 记录损失
        self.log("val/recon_loss", info["recon_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/vq_loss", info["vq_loss"], on_step=False, on_epoch=True)
        self.log("val/commitment_loss", info["commitment_loss"], on_step=False, on_epoch=True)
        self.log("val/perc_loss", info["perc_loss"], on_step=False, on_epoch=True)
        self.log("val/total_loss", info["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", info["perplexity"], on_step=False, on_epoch=True)
        self.log("val/code_usage_rate", info["code_usage_rate"], on_step=False, on_epoch=True)
        
        return info
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        
        # 学习率调度器（可选）
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=100000,  # 可根据需要调整
                eta_min=1e-5,
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像到离散 code indices"""
        z_e = self.encoder(x)
        _, _, info = self.quantizer(z_e)
        return info["encoding_indices"]
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """从 code indices 解码图像"""
        # indices: (B, H*W)
        B, HW = indices.shape
        H = W = int(HW ** 0.5)
        
        # 从 codebook 查找对应的 embedding
        quantized = self.quantizer.embedding[:, indices].permute(0, 2, 1)
        quantized = quantized.reshape(B, self.quantizer.embedding_dim, H, W)
        
        # 解码
        x_recon = self.decoder(quantized)
        return x_recon


if __name__ == "__main__":
    # 测试代码
    model = VQVAEModel(
        latent_channels=256,
        num_embeddings=1024,
        embedding_dim=256,
    )
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    x_recon, info = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"重建形状: {x_recon.shape}")
    print(f"重建损失: {info['recon_loss']:.4f}")
    print(f"VQ 损失: {info['vq_loss']:.4f}")
    print(f"困惑度: {info['perplexity']:.2f}")
    print(f"Code 使用率: {info['code_usage_rate']:.2%}")
    
    # 测试编码/解码
    indices = model.encode(x)
    print(f"Code indices 形状: {indices.shape}")
    
    x_recon2 = model.decode(indices)
    print(f"从 indices 解码的形状: {x_recon2.shape}")
    
    print("\n模型测试通过！")

