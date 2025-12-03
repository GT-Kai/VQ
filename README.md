# VQ-VAE Anime Avatar Training Project

This project implements a VQ-VAE / VQ-GAN model using PyTorch Lightning for learning discrete latent space representations of anime avatars.

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Project Structure](#-project-structure)
- [Configuration File](#-configuration-file)
- [Training Steps](#-training-steps)
- [Validation and Testing](#-validation-and-testing)
- [Common Issues](#-common-issues)

---

## 🚀 Environment Setup

### 1. Create Conda Environment

```bash
# Create Python 3.10 environment
conda create -n VQ python=3.10
conda activate VQ
```

### 2. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt
```

Main dependencies include:
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- SwanLab (experiment logging)
- Other dependencies see `requirements.txt`

---

## 📦 Dataset Preparation

### Dataset Format

The project supports loading images directly from a directory with the following requirements:

- **Image format**: `.png` or `.jpg`
- **Resolution**: Recommended 256×256 (the program will automatically resize)
- **Directory structure**:

```
datas/
  your_dataset/
    image1.png
    image2.jpg
    ...
```

### Dataset Configuration

Modify the dataset path in `conf/config.yaml`:

```yaml
data:
  class_path: dataloader.dataModule.AnimeDataModule
  init_args:
    data_dir: /path/to/your/dataset  # Change to your dataset path
    image_size: 256
    batch_size: 32
    num_workers: 4
    val_split: 0.2  # Validation set ratio (20%)
    train_augment: true  # Enable data augmentation
```

### Example Dataset

Example dataset path used in the project configuration (Kaggle):
```yaml
data_dir: /home/lick/project/VQ/datas/soumikrakshit/anime-faces/versions/1/data
```

---

## 📁 Project Structure

```
VQ/
├── main.py                 # Training main program (LightningCLI)
├── conf/
│   └── config.yaml        # Configuration file (model, training, data parameters)
├── model/
│   └── modelModule.py     # VQ-VAE model definition
├── dataloader/
│   └── dataModule.py      # Data loading module
├── callback/
│   └── MyCallback.py      # Custom callbacks (logging, visualization)
├── bash/
│   └── run.sh             # Training script example
├── checkpoints/           # Model checkpoint save directory
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## ⚙️ Configuration File

The configuration file is located at `conf/config.yaml` and mainly contains three parts:

### 1. Trainer Configuration

```yaml
trainer:
  accelerator: gpu         # Use GPU
  devices: 1              # Number of GPUs
  max_epochs: 100         # Maximum training epochs
  precision: 16-mixed     # Mixed precision training
  gradient_clip_val: 1.0  # Gradient clipping
  accumulate_grad_batches: 1  # Gradient accumulation steps
```

**Multi-GPU Training**:
- Set `devices: 2` to use 2 GPUs
- Uncomment `strategy: ddp` to enable distributed training

### 2. Model Configuration

```yaml
model:
  class_path: model.modelModule.VQVAEModel
  init_args:
    # Model structure
    latent_channels: 256           # Latent variable channels
    encoder_channels: [128, 256, 256, 256]
    decoder_channels: [256, 256, 256, 128]
    
    # Quantizer parameters
    num_embeddings: 1024          # Codebook size
    embedding_dim: 256            # Embedding dimension
    
    # Loss weights
    lambda_rec: 1.0               # Reconstruction loss
    lambda_vq: 1.0                # VQ loss
    lambda_commit: 0.25           # Commitment loss
    lambda_perc: 0.5              # Perceptual loss (LPIPS)
    lambda_gan: 0.0               # GAN loss (VQ-GAN extension)
    
    # Optimizer
    learning_rate: 2.0e-4
    betas: [0.9, 0.99]
```

### 3. Data Configuration

```yaml
data:
  class_path: dataloader.dataModule.AnimeDataModule
  init_args:
    data_dir: /path/to/dataset
    batch_size: 32
    image_size: 256
    val_split: 0.2
    train_augment: true
```

---

## 🏃 Training Steps

### 1. Verify Configuration

Before starting training, verify that the configuration file is correct:

```bash
python main.py fit --config conf/config.yaml --print_config
```

This will print the complete configuration information, including all parameters.

### 2. Quick Test Run

Use `fast_dev_run` mode to quickly verify that the code can run normally:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py fit --config conf/config.yaml --trainer.fast_dev_run=true
```

### 3. Start Training

Using the provided script:

```bash
cd bash
bash run.sh
```

Or run directly:

```bash
python main.py fit --config conf/config.yaml
```

### 4. Resume Training from Checkpoint

If training is interrupted, you can resume from the last saved checkpoint:

```bash
python main.py fit --config conf/config.yaml --ckpt_path checkpoints/last.ckpt
```

### 5. Use Specific GPU

Specify GPU via environment variable:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py fit --config conf/config.yaml
```

---

## 🔍 Validation and Testing

### Validation Mode

Evaluate the model using the validation set:

```bash
python main.py validate --config conf/config.yaml --ckpt_path checkpoints/best.ckpt
```

### Test Mode

```bash
python main.py test --config conf/config.yaml --ckpt_path checkpoints/best.ckpt
```

---

## 📊 Monitoring and Logging

### SwanLab Logging

The project integrates SwanLab for experiment tracking, including:

- Training/validation loss curves
- Reconstructed image visualization
- Codebook usage statistics
- Hyperparameter records

After training starts, SwanLab will automatically log. You can modify the project name and experiment name in the configuration file:

```yaml
trainer:
  logger:
    - class_path: swanlab.integration.pytorch_lightning.SwanLabLogger
      init_args:
        project: vq-vae-anime
        experiment_name: vqvae-baseline
```

### Checkpoint Saving

Model checkpoints are saved in the `checkpoints/` directory:

- `vqvae-{epoch}-{loss}.ckpt`: Best model saved by validation loss
- `last.ckpt`: Last training checkpoint

Configuration notes:
- `save_top_k: 3`: Keep the top 3 best models
- `every_n_train_steps: 10000`: Save every 10000 steps

---

## ❓ Common Issues

### 1. Out of Memory (OOM)

**Solution**:
- Reduce `batch_size` (e.g., from 32 to 16)
- Use gradient accumulation: `accumulate_grad_batches: 2`
- Ensure mixed precision is used: `precision: 16-mixed`

### 2. Dataset Path Error

**Error message**: `No image files found in {data_dir}`

**Solution**:
- Check if the `data_dir` path in `conf/config.yaml` is correct
- Ensure the directory contains `.png` or `.jpg` files

### 3. Codebook Underuse

**Symptoms**: Only a few codes are used, reconstructed images are blurry

**Solution**:
- Increase `lambda_commit` (e.g., from 0.25 to 0.5)
- Reduce `num_embeddings` (e.g., from 1024 to 512)
- Check if the learning rate is too high

### 4. Blurry Reconstructed Images

**Solution**:
- Increase perceptual loss weight: `lambda_perc: 1.0`
- Increase codebook size: `num_embeddings: 2048`
- Increase model capacity (more ResBlocks or larger channel numbers)

### 5. Training Instability

**Solution**:
- Lower learning rate: `learning_rate: 1.0e-4`
- Enable gradient clipping: `gradient_clip_val: 1.0`
- Check if data augmentation is too aggressive

### 6. Checkpoint Loading Failure

**Solution**:
- Ensure the checkpoint file path is correct
- Check if the model configuration matches the training configuration
- Use the `--ckpt_path` parameter to specify the full path

---

## 🔧 Advanced Configuration

### Multi-GPU Training

Configure in `conf/config.yaml`:

```yaml
trainer:
  accelerator: gpu
  devices: 2              # Use 2 GPUs
  strategy: ddp           # Enable distributed training
```

Note: `batch_size` is the batch size per GPU, total batch size = `batch_size × num_gpus`.

### Custom Data Augmentation

Modify the `_get_train_transform()` method in `dataloader/dataModule.py` to customize data augmentation.

### Adjust Loss Weights

Adjust loss weights in `conf/config.yaml` based on training results:

- `lambda_rec`: Reconstruction loss (default 1.0)
- `lambda_vq`: VQ loss (default 1.0)
- `lambda_commit`: Commitment loss (default 0.25)
- `lambda_perc`: Perceptual loss (default 0.5)

---

## 📝 Training Recommendations

### Recommended Training Process

1. **Phase 1: Basic Training**
   - Start training with default configuration
   - Monitor validation loss and reconstruction quality
   - Train until reconstruction results stabilize

2. **Phase 2: Fine-tuning**
   - Adjust loss weights based on reconstruction quality
   - If images are blurry, increase `lambda_perc`
   - If code usage is insufficient, adjust `lambda_commit`

3. **Phase 3: Extension (Optional)**
   - Enable GAN loss to extend to VQ-GAN
   - Gradually increase `lambda_gan` (from 0 to 0.1)

### Training Time Estimation

- Dataset: ~10k images
- Batch size: 32
- Steps per epoch: ~320
- Recommended training: 100-200k steps (approximately 320-640 epochs)

---

## 📚 Related Documentation

- `conf/README.md`: Detailed configuration file documentation
- `model/README.md`: Model architecture documentation
- `dataloader/README.md`: Data loading documentation
- `callback/README.md`: Callback function documentation

---

## 📄 License

This project is for learning and research purposes only.

---

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!
