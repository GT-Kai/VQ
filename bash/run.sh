export CUDA_VISIBLE_DEVICES=1
export SWANLAB_API_KEY=g8ff0AxRFdoWGvlvEYAoe
# 验证配置（不实际训练）
# python main.py fit --config conf/config.yaml --print_config

# # 测试运行（快速验证）
# python main.py fit --config conf/config.yaml --trainer.fast_dev_run=true

# # 开始训练
python main.py fit --config conf/config.yaml

# # 从 checkpoint 继续训练
# python main.py fit --config conf/config.yaml --ckpt_path checkpoints/last.ckpt

# # 仅验证
# python main.py validate --config conf/config.yaml --ckpt_path checkpoints/best.ckpt

# # 测试
# python main.py test --config conf/config.yaml --ckpt_path checkpoints/best.ckpt