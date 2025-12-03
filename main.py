"""
VQ-VAE 训练主程序
使用 LightningCLI 从配置文件加载参数
"""

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule


class MyLightningCLI(LightningCLI):
    """自定义 LightningCLI，支持从配置文件加载 class_path"""
    
    def add_arguments_to_parser(self, parser):
        """添加自定义参数链接（如果需要）"""
        # 可以在这里链接参数，例如：
        # parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        pass


cli = MyLightningCLI(
    LightningModule,
    LightningDataModule,
    save_config_callback=None,
    subclass_mode_model=True, 
    subclass_mode_data=True,
    )


