"""
VQ-VAE 训练主程序
使用 LightningCLI 从配置文件加载参数
"""

from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    """自定义 LightningCLI，支持从配置文件加载 class_path"""
    
    def add_arguments_to_parser(self, parser):
        """添加自定义参数链接（如果需要）"""
        # 可以在这里链接参数，例如：
        # parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        pass


def cli_main():
    """使用 LightningCLI 创建命令行接口"""
    # 不指定 model_class 和 datamodule_class，让配置文件中的 class_path 生效
    cli = MyLightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()


