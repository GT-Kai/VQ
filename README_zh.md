## README（中文）— 动漫头像 VQ-VAE / VQ-GAN 技术方案

---

## 0. 项目环境搭建

```bash
conda create -n VQ python=3.10
pip install -r requirements.txt
```

## 1. 项目概述

本说明文档给出在约 **1 万张 256×256 动漫头像**上训练 **VQ-VAE**（可扩展为 **VQ-GAN**）的完整技术方案。实现基于 **PyTorch Lightning**，硬件为单机 **RTX 4090**。

**主要目标**：

- 学习适合动漫头像的高质量 **离散潜空间（codebook）**  
- 获得较好的 **重建质量** 与 **随机生成多样性**  
- 保持架构 **简洁、稳定且可扩展**（方便后续加 GAN loss 或 latent prior）

---

## 2. 推荐目录结构

```text
project/
  configs/
    vqvae_anime.yaml          # 超参数配置
  data/
    raw/                      # 原始图片
    processed/                # 预处理后图片（可选）
  src/
    datamodules/
      anime_datamodule.py     # LightningDataModule
    models/
      vqvae_module.py         # LightningModule (VQ-VAE / VQ-GAN)
      modules/
        encoder.py
        decoder.py
        quantizer_ema.py
        discriminator.py      # 可选：VQ-GAN 判别器
    utils/
      losses.py               # 重建损失、感知损失、GAN 损失等
      metrics.py              # FID、LPIPS 等评估
  scripts/
    train_vqvae.sh
  README_zh.md
  README_en.md
```

---

## 3. 模型结构设计

### 3.1 Encoder 设计

- **骨干网络**：ResNet 风格卷积 Encoder  
- **输入**：`3 × 256 × 256` RGB 图像，归一化到 `[-1, 1]`  
- **下采样路径**（总下采样倍数 = 16）：
  - `256 → 128 → 64 → 32 → 16`（4 个 stage，每个 stage 下采样 2×）
- **通道数设计（示例）**：
  - Stage1（256→128）：128 通道
  - Stage2（128→64）：256 通道
  - Stage3（64→32）：256 通道
  - Stage4（32→16）：256 通道
- **每个 stage 模块**：
  - `Conv (stride=2, kernel=3 或 4, padding=1)` 做下采样
  - 后接 2–3 个 ResBlock：
    - `Conv 3×3 → Norm(GroupNorm/BatchNorm) → 激活(ReLU/SiLU)`  
    - 残差连接
  - 可选：在 32×32 或 16×16 分辨率加入轻量自注意力块（可提升细节表达能力）
- **潜变量通道数**：
  - 推荐 `C_latent = 256`（平衡性能和显存）
  - 若需求更高保真且显存允许，可尝试 `C_latent = 384`
- **Encoder 输出**：
  - 特征图形状：`C_latent × 16 × 16`

### 3.2 Decoder 设计

- 对 Encoder **近似对称**，但使用上采样：
  - `16 → 32 → 64 → 128 → 256`
- 每个上采样 stage：
  - 最近邻或双线性上采样 2×
  - Conv 3×3 + 若干 ResBlock（与 Encoder 呼应）
- 通道设计（与 Encoder 反向）：
  - 从 `C_latent`（256 或 384）逐步降到 64
- 输出层：
  - 最后一层：Conv 3×3 → 3 通道 → `tanh`，输出范围 `[-1, 1]`

### 3.3 向量量化器 / Codebook

- 使用 **EMA 版 Vector Quantizer**（类似 VQ-VAE v2 / VQ-GAN）以提升稳定性：
  - EMA 衰减系数：`ema_decay = 0.99 – 0.999`
  - cluster size 平滑项：`eps = 1e-5`
- 潜空间网格：
  - 尺寸：`H_lat × W_lat = 16 × 16`
  - 每个空间位置的向量映射到最近的 code。
- 量化器接口（抽象）：
  - 输入：`z_e ∈ ℝ^{B × C_latent × 16 × 16}`
  - 输出：`z_q`（量化后的特征，形状相同）以及 `indices ∈ ℕ^{B × 16 × 16}`

---

## 4. Codebook 设计建议

### 4.1 Code 数量与 Embedding 维度

已知条件：

- 数据集约 1 万张  
- 分辨率 256×256  
- 潜空间分辨率 16×16（每张图 256 个 token）

**推荐配置**：

- **Embedding 维度** `D_e`：
  - 建议 `D_e = 256`，与 `C_latent` 保持一致
  - 若使用 `C_latent = 384`，可以：
    - 先用 1×1 Conv 将 384 映射到 256，再送入量化器；或
    - 直接使用 `D_e = 384` 的 codebook（参数更多）
- **Code 数量** `K`：
  - 初始推荐 **`K = 1024`**
  - 若重建偏模糊且 code 使用率非常高，可增大为 **`K = 2048`**
  - 不建议在该数据规模下盲目用过大的 K（如 8k+），否则容易产生大量“死 code”

### 4.2 初始化策略

- **Codebook 权重**：
  - 可使用标准 PyTorch 初始化（如 Kaiming uniform）
  - 或显式设为 `N(0, 0.1)` 正态分布
- **可选 k-means 初始化（进阶）**：
  - 关闭量化，先用 Encoder 跑若干 batch（例如 5k 图像）
  - 收集大量 `z_e` 向量，跑 k-means 聚类（K 个簇）
  - 用聚类中心初始化 codebook
- **EMA 统计量**：
  - 初始 count 设为一个小正数，避免除零

### 4.3 使用情况监控与正则化

- 建议实时监控：
  - 各 code 的使用直方图
  - code perplexity（困惑度）
- 若大量 code 基本不被使用：
  - 尝试减小 `K`（如 1024→512 或 768）
  - 或实现 **codebook reset**：
    - 定期将极少使用的 code 重置为高重建误差样本对应的 `z_e`

---

## 5. 损失函数组合与权重

总体损失形式：

\\[
L = \\lambda_{rec} L_{rec} + \\lambda_{vq} L_{vq} + \\lambda_{commit} L_{commit} + \\lambda_{perc} L_{perc} + \\lambda_{gan} L_{gan}
\\]

其中 VQ-VAE 基础部分只用前四项，VQ-GAN 扩展时再加入 `L_gan`。

### 5.1 重建损失 `L_rec`

- 使用像素级 `L1`（一般比 `L2` 锐利）：
  - `L_rec = mean(|x - x_recon|)`，在 `[-1, 1]` 归一化空间下计算
- 可选：`L1 + 少量 L2`，提升数值稳定性
- **权重**：
  - `λ_rec = 1.0`（作为主尺度）

### 5.2 VQ codebook 损失与 commitment 损失

- 经典 VQ-VAE 形式：
  - `L_vq = ||sg[z_e] - e||_2^2`（更新 codebook）
  - `L_commit = ||z_e - sg[e]||_2^2`（约束 Encoder 输出）
- **权重**：
  - `λ_vq = 1.0`
  - `λ_commit = 0.25 – 0.5`（默认 0.25；若 code 使用不足可调大）

### 5.3 感知损失 `L_perc`（LPIPS）

- 建议加入 LPIPS（VGG 版）：
  - `L_perc = LPIPS(x, x_recon)`
- 对动漫线条、眼睛和发丝等细节非常有帮助
- **权重**：
  - 初始 `λ_perc = 0.5`
  - 若重建仍偏糊，可增大到 `1.0`

### 5.4 可选 GAN 损失 `L_gan`（VQ-GAN 扩展）

若要扩展为 VQ-GAN：

- 引入 PatchGAN 判别器，输入 256×256 图像  
- 损失形式可选：
  - hinge GAN 或非饱和 GAN（带 logit）
- **权重与 schedule**：
  - 先用纯 VQ-VAE 训练一段时间（如 20k step，`λ_gan = 0`）
  - 再线性 warmup `λ_gan : 0 → 0.1`（如 20k step）
  - 之后根据稳定性和视觉效果在 `0.05 – 0.2` 内微调

---

## 6. 训练策略

### 6.1 数据与预处理

- **数据集**：
  - 约 1 万张动漫头像，建议保证构图大致一致（头肩照）
- **预处理**：
  - 转换为 RGB
  - center-crop 或 pad 成正方形，再 resize 到 256×256
  - 归一化到 `[-1, 1]`：`x = (img / 127.5) - 1.0`
- **数据增强**：
  - 水平翻转（p=0.5）
  - 轻微亮度/对比度/饱和度扰动（0.1–0.2）
  - 避免大角度旋转及过重几何变换，以免破坏人脸结构

### 6.2 batch size、训练步数 / epoch

- 硬件：单张 RTX 4090（24 GB）
- **batch size**：
  - 目标 `batch_size = 32`（256×256 VQ-VAE 一般可以承受）
  - 若显存不足：
    - 使用 `batch_size = 16`，并用梯度累积（accumulate_grad_batches=2）实现等效 32
- **训练时长**：
  - 建议总步数在 `100k – 200k` 之间
  - 对 1 万张图、有效 batch=32，相当于约 320–640 个“epoch”  
  - 因数据量不大，需重视验证集与早停策略，避免明显过拟合

### 6.3 优化器与学习率

- 优化器：`Adam` 或 `AdamW`
  - `lr = 2e-4`（Encoder / Decoder / Quantizer）
  - `betas = (0.9, 0.99)`
  - `weight_decay = 0 – 1e-4`（可以先设 0）
- 若使用 VQ-GAN：
  - 判别器单独一个优化器：
    - `lr = 2e-4`，`betas = (0.5, 0.99)` 或 `(0.9, 0.99)`

### 6.4 学习率调度与 warmup

- **warmup**：
  - 前 `2k – 5k` step 使用线性 warmup：`1e-6 → 2e-4`
- **warmup 之后**：
  - 使用 cosine decay：从 `2e-4` 逐渐衰减到 `1e-5`  
  - 或使用 `ReduceLROnPlateau` 监控验证重建损失

### 6.5 PyTorch Lightning 配置

- **精度**：
  - 推荐 `precision=16`（混合精度），节省显存并加速
- **梯度裁剪**：
  - `gradient_clip_val = 1.0`
- **checkpoint 策略**：
  - 按验证集重建损失 / LPIPS 保存最佳模型
  - 每 N step（如 10k）保存一个周期性 checkpoint 方便回溯
- **日志**：
  - 记录：
    - 各项损失：`L_rec, L_vq, L_commit, L_perc, L_gan`
    - code 使用直方图 / perplexity
    - 重建可视化（每几百 step 存一批 image grid）

---

## 7. 可能问题与解决思路

### 7.1 codebook 使用不足 / 潜变量塌陷

- **表现**：
  - 只有少数 code 被频繁使用
  - 重建图像较模糊，多样性差
- **解决方案**：
  - 增大 `λ_commit`（如 0.25→0.5）
  - 减小 `K`（如 1024→512）
  - 检查学习率是否过大（影响 EMA 稳定）
  - 定期对“死 code”做 reset

### 7.2 训练不稳定（尤其是加 GAN 后）

- 先用 **纯 VQ-VAE**（无 GAN）训练到重建效果可接受
- 开启 GAN 时：
  - 使用 `λ_gan` 的线性 warmup（避免突然冲击）
  - 适当调低判别器学习率
  - 视情况加入 spectral norm 或 gradient penalty

### 7.3 重建模糊 / 边缘不锐利

- 加大感知损失权重 `λ_perc`（如 0.5→1.0）
- 提升模型容量：
  - 增加少量 ResBlock
  - 稍微提高 `C_latent`（如 256→384）
- 适度增大 codebook：
  - `K: 1024 → 2048`（同时监控 code 使用率）

### 7.4 过拟合（数据只有 1 万张）

- 使用适度增强（翻转 + 轻微色彩抖动）
- 指定训练 / 验证划分（如 8k/2k）
- 启用早停（EarlyStopping），监控验证 L_rec / LPIPS

---

## 8. 任务清单（To-do）

### 8.1 数据与配置

- [ ] 收集并清洗约 1 万张 256×256 动漫头像
- [ ] 实现预处理与增强 pipeline（crop / resize / normalize / flip / jitter）
- [ ] 编写 `configs/vqvae_anime.yaml`，包括：
  - 模型结构参数（通道数、K、D_e、下采样层数等）
  - 各损失权重（λ_rec, λ_vq, λ_commit, λ_perc, λ_gan）
  - 优化器与学习率调度
  - 训练配置（batch size、总 step、precision 等）

### 8.2 核心模型（Lightning）

- [ ] 实现 `AnimeDataModule`（`anime_datamodule.py`）：
  - 训练 / 验证划分
  - 数据预处理与增强
  - DataLoader 配置
- [ ] 实现 `Encoder`（`encoder.py`）与 `Decoder`（`decoder.py`）
- [ ] 实现 EMA 版 `VectorQuantizerEMA`（`quantizer_ema.py`）
- [ ] 实现 `VQVAEModel`（`vqvae_module.py`，LightningModule）：
  - 前向流程：encode → quantize → decode
  - 计算并记录各项损失
  - 记录 code 使用统计与重建图像
- [ ]（可选）实现 `Discriminator`（`discriminator.py`）与 VQ-GAN 的 GAN 损失

### 8.3 训练与评估

- [ ] 编写训练脚本 `scripts/train_vqvae.sh`，基于 config 启动训练
- [ ] 在 `metrics.py` 中实现 FID / LPIPS 等评估指标
- [ ] 实现采样工具：
  - 从 codebook 随机采样并解码生成图像
  - 对给定图像做 encode → manipulate codes → decode

### 8.4 实验管理

- [ ] 集成日志系统（TensorBoard 或 WandB）
- [ ] 配置 checkpoint 与早停策略
- [ ] 记录主要实验设置和结果（如不同 K、不同 λ_perc / λ_gan 的效果）

---

## 9. 推荐的执行顺序

1. **Phase 1：纯 VQ-VAE**  
   - 只用重建损失 + VQ 损失 + commitment + LPIPS  
   - 确保训练收敛、重建质量和 code 使用情况正常  
2. **Phase 2：VQ-GAN 扩展（可选）**  
   - 加入 PatchGAN 判别器与 GAN 损失  
   - 使用 `λ_gan` warmup，观察锐利度与稳定性  
3. **Phase 3：建模离散潜空间先验（可选）**  
   - 在离散 code 序列（16×16=256 tokens）上训练自回归/Transformer prior  
   - 实现完全基于 code 的生成、插值与编辑


