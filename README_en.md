## README (English) — Anime Avatar VQ-VAE / VQ-GAN Plan

---

## 0. Project environment setup

```bash
conda create -n VQ python=3.10
pip install -r requirements.txt
```

## 1. Overview

This document describes a **technical plan** for training a VQ-VAE (optionally extendable to VQ-GAN) on ~10,000 anime avatar images with resolution **256×256** using a single **RTX 4090** GPU and **PyTorch Lightning**.

**Main goals**:

- Learn a high-quality **discrete latent space** (codebook) specialized for anime faces  
- Achieve strong **reconstruction quality** and **diverse generation**  
- Keep the design **simple, stable, and extensible** (easy to add GAN loss or a prior over codes later)

---

## 2. Recommended Project Structure

```text
project/
  configs/
    vqvae_anime.yaml          # hyper-parameters
  data/
    raw/                      # raw images
    processed/                # optional preprocessed images
  src/
    datamodules/
      anime_datamodule.py     # LightningDataModule
    models/
      vqvae_module.py         # LightningModule (VQ-VAE / VQ-GAN)
      modules/
        encoder.py
        decoder.py
        quantizer_ema.py
        discriminator.py      # optional, for VQ-GAN
    utils/
      losses.py               # reconstruction, perceptual, GAN losses
      metrics.py              # FID, LPIPS evaluation
  scripts/
    train_vqvae.sh
  README_zh.md
  README_en.md
```

---

## 3. Model Architecture Design

### 3.1 Encoder

- **Backbone**: ResNet-style convolutional encoder.
- **Input**: `3 × 256 × 256` RGB image normalized to `[-1, 1]`.
- **Downsampling schedule** (total factor = 16):
  - `256 → 128 → 64 → 32 → 16` (4 stages, each 2×)
- **Channel plan** (example):
  - Level 1 (256→128): 128 channels
  - Level 2 (128→64): 256 channels
  - Level 3 (64→32): 256 channels
  - Level 4 (32→16): 256 channels
- **Block design per level**:
  - `Conv (stride 2, kernel 4 or 3, padding 1)` for downsampling
  - 2–3 residual blocks with:
    - Conv 3×3 → GroupNorm / BatchNorm → nonlinearity (ReLU/SiLU)
    - Optional attention block at 32×32 or 16×16 (if you want more capacity)
- **Latent channels**:
  - `C_latent = 256` (good default)
  - For higher fidelity (larger model + more VRAM): `C_latent = 384`
- **Encoder output**:
  - Feature map: `C_latent × 16 × 16`

### 3.2 Decoder

- **Symmetric structure** to encoder, but with upsampling:
  - `16 → 32 → 64 → 128 → 256`
- Each upsampling stage:
  - Nearest-neighbor or bilinear upsample ×2
  - Conv 3×3 + residual blocks (mirroring encoder)
- Channels (reverse of encoder):
  - Start from `C_latent` (256 or 384)
  - Gradually reduce to 64 near 256×256 resolution
- Final layer:
  - Conv 3×3 → `3` channels → `tanh` activation → output in `[-1, 1]`

### 3.3 Vector Quantizer / Codebook

- Use **EMA-based vector quantization** (as in VQ-VAE v2 / VQ-GAN) for stability:
  - EMA decay: `ema_decay = 0.99 – 0.999`
  - Small epsilon for cluster size: `eps = 1e-5`
- Latent grid:
  - Size: `H_lat × W_lat = 16 × 16`
  - At each spatial position, we map encoder output vector to closest code vector.
- Quantizer interface (conceptual):
  - Input: `z_e ∈ ℝ^{B × C_latent × 16 × 16}`
  - Output: `z_q` (quantized features, same shape) and `indices ∈ ℕ^{B × 16 × 16}`

---

## 4. Codebook Design

### 4.1 Codebook Size & Embedding Dimension

Given:

- Dataset: ~10k images  
- Resolution: 256×256  
- Latent resolution: 16×16 (256 tokens per image)

**Recommended configuration**:

- **Embedding dimension** `D_e`:
  - `D_e = 256`, matching `C_latent`
  - If `C_latent = 384`, either:
    - Project `384 → 256` before quantizer, or
    - Use `D_e = 384` and keep codebook that size (more expensive)
- **Number of codes** `K`:
  - Start with **`K = 1024`**
  - If reconstructions are smooth and code usage appears saturated, consider **`K = 2048`**
  - Avoid excessively large `K` (e.g. 8k+) for this data size to prevent many dead codes.

### 4.2 Initialization Strategy

- **Codebook weights**:
  - Use standard PyTorch initialization (e.g. Kaiming uniform) or:
  - Explicitly set `N(0, 0.1)` for code vectors.
- **Optional k-means init** (advanced, optional):
  - Run encoder on a subset (e.g. 5k images) without quantization.
  - Collect a large number of latent vectors (`z_e`) and run k-means with `K` clusters.
  - Initialize codebook with these centroids.
- **EMA stats**:
  - Initialize counts to small positives to avoid division by zero.

### 4.3 Monitoring & Regularization

- Track:
  - Code usage histogram over dataset
  - Per-batch and moving-average perplexity
- If many codes are essentially unused:
  - Try **smaller K** (e.g. 512 or 768), or
  - Implement **codebook reset**:
    - Periodically reinitialize rarely used vectors to new encoder latents with high reconstruction error.

---

## 5. Loss Functions and Weights

We combine reconstruction, codebook, commitment, and perceptual losses; GAN loss is optional.

Overall loss:

\\[
L = \\lambda_{rec} L_{rec} + \\lambda_{vq} L_{vq} + \\lambda_{commit} L_{commit} + \\lambda_{perc} L_{perc} + \\lambda_{gan} L_{gan}
\\]

### 5.1 Reconstruction Loss

- Use pixel-wise `L1` (often sharper than `L2`):
  - `L_rec = mean(|x - x_recon|)`
- Optionally mix with `L2` (small weight) if optimization is noisy.
- **Weight**:
  - `λ_rec = 1.0` (base scale; others relative to this)

### 5.2 VQ Codebook & Commitment Loss

- Standard VQ-VAE form:
  - `L_vq = ||sg[z_e] - e||_2^2` (move embeddings towards encoder outputs)
  - `L_commit = ||z_e - sg[e]||_2^2` (prevent encoder outputs from drifting too far)
- **Weights**:
  - `λ_vq = 1.0`
  - `λ_commit = 0.25 – 0.5` (start with `0.25`; if codes underused, increase)

### 5.3 Perceptual Loss (LPIPS)

- Use **LPIPS** with a VGG-based backbone.
- Compute between input and reconstruction:
  - `L_perc = LPIPS(x, x_recon)`
- **Weight**:
  - Start with `λ_perc = 0.5`
  - If reconstructions are too blurry, increase to `1.0`

### 5.4 Optional GAN Loss (VQ-GAN Extension)

When extending to VQ-GAN:

- Add a **PatchGAN** discriminator operating on 256×256 images.
- Loss choice:
  - Hinge GAN or non-saturating GAN with logits.
- **Weights**:
  - Start small: `λ_gan = 0.1`
  - Warm up: first train with `λ_gan = 0` for some steps (e.g. 20k), then linearly ramp `0 → 0.1` over another 20k steps.

---

## 6. Training Strategy

### 6.1 Data & Preprocessing

- **Dataset**:
  - ~10k anime avatars, ensure consistent framing (head/shoulders).
- **Preprocessing**:
  - Convert to RGB, pad or center-crop to square, then resize to 256×256.
  - Normalize to `[-1, 1]` (i.e. `(x / 127.5) - 1.0`).
- **Augmentations**:
  - Horizontal flip (p=0.5).
  - Light color jitter (brightness/contrast/saturation small, e.g. 0.1–0.2).
  - Avoid strong rotations or large crops so facial structure remains stable.

### 6.2 Batch Size, Steps, and Epochs

- Hardware: single RTX 4090 (24 GB).
- **Batch size**:
  - Target `batch_size = 32` for 256×256; if OOM, use:
    - `batch_size = 16` with gradient accumulation 2 steps (effective 32).
- **Training length**:
  - ~`100k – 200k` optimizer steps is a good starting range.
  - With batch 32 and 10k images, this is ~320–640 epochs equivalent; monitor validation to avoid severe overfitting.

### 6.3 Optimizer and Learning Rate

- Optimizer: `Adam` or `AdamW`.
  - `lr = 2e-4` for encoder, decoder, and quantizer.
  - `betas = (0.9, 0.99)`
  - `weight_decay = 0.0 – 1e-4` (0 when starting is fine).
- If using VQ-GAN:
  - Separate optimizer for discriminator:
    - `lr = 2e-4`, `betas = (0.5, 0.99)` or `(0.9, 0.99)`.

### 6.4 LR Schedule & Warmup

- **Warmup**:
  - Linear warmup from `1e-6` to `2e-4` over first `2k – 5k` steps.
- **After warmup**:
  - Cosine decay from `2e-4` to `1e-5` over remaining steps; or
  - Step LR: reduce ×0.5 after plateau on validation loss.
- Implement schedule via PyTorch Lightning `lr_scheduler` interface.

### 6.5 PyTorch Lightning Configuration

- **Precision**: use `precision=16` (mixed precision) to save memory and speed up.
- **Gradient clipping**:
  - `gradient_clip_val = 1.0`.
- **Checkpointing**:
  - Save:
    - Best model by validation reconstruction loss and/or LPIPS.
    - Periodic checkpoints every N steps (e.g. 10k) for inspection.
- **Logging**:
  - Log:
    - Loss breakdown (`L_rec`, `L_vq`, `L_commit`, `L_perc`, `L_gan`).
    - Code usage histogram/perplexity.
    - Example reconstructions every few hundred steps.

---

## 7. Common Issues & Mitigation

### 7.1 Codebook Underuse / Posterior Collapse

- **Symptoms**:
  - Only a few codes dominate usage.
  - Reconstructions look overly smooth, diversity limited.
- **Solutions**:
  - Increase `λ_commit` (e.g. from 0.25 to 0.5).
  - Reduce `K` (e.g. 1024 → 512).
  - Check LR (too high LR can destabilize EMA updates).
  - Apply codebook reset on dead codes.

### 7.2 Training Instability (Especially with GAN)

- Start without GAN loss:
  - Train pure VQ-VAE until reconstructions are acceptable.
- When enabling GAN:
  - Warm up `λ_gan` slowly.
  - Optionally lower discriminator LR compared to generator.
  - Use spectral normalization or gradient penalty if discriminator collapses.

### 7.3 Blurry or Overly Smooth Reconstructions

- Increase `λ_perc` (e.g. 0.5 → 1.0).
- Increase model capacity:
  - More residual blocks, slightly larger `C_latent`.
- Increase codebook size:
  - `K: 1024 → 2048` (watch code usage).

### 7.4 Overfitting on 10k Images

- Use moderate augmentations (flip, small color jitter).
- Keep an explicit validation split (e.g. 8k train / 2k val).
- Add early stopping patience on validation reconstruction loss or LPIPS.

---

## 8. Implementation Tasks (Checklist)

### 8.1 Data & Config

- [ ] Prepare anime avatar dataset (~10k images, 256×256).
- [ ] Implement preprocessing and augmentation pipeline.
- [ ] Create `configs/vqvae_anime.yaml` including:
  - Model architecture (channels, K, D_e, resolutions).
  - Loss weights (λ_rec, λ_vq, λ_commit, λ_perc, λ_gan).
  - Optimizer and LR schedule.
  - Training hyperparameters (batch size, steps, precision).

### 8.2 Core Model (PyTorch Lightning)

- [ ] Implement `AnimeDataModule` (`anime_datamodule.py`):
  - Train/val splits, transforms, DataLoader settings.
- [ ] Implement `Encoder` (`encoder.py`) and `Decoder` (`decoder.py`).
- [ ] Implement EMA-based `VectorQuantizerEMA` (`quantizer_ema.py`).
- [ ] Implement `VQVAEModel` (`vqvae_module.py`) as `LightningModule`:
  - Forward pass (encode → quantize → decode).
  - Compute all losses and log them.
  - Log reconstructions and code usage stats.
- [ ] (Optional) Implement `Discriminator` (`discriminator.py`) and GAN loss for VQ-GAN.

### 8.3 Training & Evaluation

- [ ] Write `scripts/train_vqvae.sh` to launch training from config.
- [ ] Implement metrics in `metrics.py` (FID, LPIPS).
- [ ] Implement sampling utilities:
  - Random sampling from codebook (optional on top of simple uniform codes).
  - Reconstruction of given images.

### 8.4 Experiment Management

- [ ] Set up logging (TensorBoard or WandB).
- [ ] Configure model checkpointing and early stopping.
- [ ] Document main experiments and hyperparameter choices.

---

## 9. Suggested Next Steps

- **Phase 1**: Train a stable VQ-VAE with reconstruction + LPIPS only.
- **Phase 2**: Add GAN discriminator to move towards VQ-GAN, if sharper outputs are needed.
- **Phase 3** (optional): Train an autoregressive or transformer prior over discrete codes for unconditional or conditional generation.


