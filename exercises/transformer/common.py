"""共用工具:matplotlib 中文配置 + --plot 参数解析 + 保存到 plots/ 子目录。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti TC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

PLOTS_DIR = Path(__file__).parent / "plots"


def parse_args(extra=None):
    """解析命令行参数。extra 可以是一个回调, 接受 parser 添加额外参数。"""
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true",
                   help="生成图片到 plots/ 子目录 (默认不画图)")
    if extra is not None:
        extra(p)
    return p.parse_args()


def save_fig(name: str, dpi: int = 120, bbox_inches: str | None = None) -> Path:
    """保存当前 figure 到 plots/<name>.png, 自动建目录。"""
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    kwargs = {"dpi": dpi}
    if bbox_inches:
        kwargs["bbox_inches"] = bbox_inches
    plt.savefig(path, **kwargs)
    return path
