"""共用工具:matplotlib 中文配置 + --plot/--save 参数解析 + 保存到 plots/ 子目录。

参数语义:
  --plot   生成并显示图 (plt.show())
  --save   生成并保存图到 plots/ (不显示)
  两个可以同时用; 都不加时跳过画图。
"""

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


def parse_args():
    """args.plot = 显示, args.save = 保存, args.draw = 任一开启就需要画。"""
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true",
                   help="生成并显示图 (plt.show)")
    p.add_argument("--save", action="store_true",
                   help="生成并保存图到 plots/ 子目录")
    args = p.parse_args()
    args.draw = args.plot or args.save
    return args


def save_fig(name: str, dpi: int = 120, bbox_inches: str | None = None) -> Path:
    """保存当前 figure 到 plots/<name>.png, 自动建目录。"""
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    kwargs = {"dpi": dpi}
    if bbox_inches:
        kwargs["bbox_inches"] = bbox_inches
    plt.savefig(path, **kwargs)
    return path


def finalize(args, name: str, bbox_inches: str | None = None):
    """根据 args 处理 figure 的最后一步: save (if --save), show (if --plot)。"""
    if args.save:
        path = save_fig(name, bbox_inches=bbox_inches)
        print(f"图已保存到 {path}")
    if args.plot:
        plt.show()
