"""
实验运行入口脚本 / Experiment Runner Entry Script

中文：简单的主程序入口，调用 heart_cdss.cli 执行实验。
English: Simple main entry point calling heart_cdss.cli to run experiments.
"""

def main() -> None:
    from heart_cdss.cli import main as _main

    _main()


if __name__ == "__main__":
    main()
