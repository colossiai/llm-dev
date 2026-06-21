"""
07 - 读取 checkpoint:查看内容 + 加载续写

================ 给零基础读者的 5 分钟讲解 ================

【前情提要】
  06 训练完 Mini GPT 后, 用 `--save_model` 把成果存成了一个 .pt 文件:
      checkpoints/06_minigpt.pt
  这一步演示**怎么把它读回来** — 既能"看里面有什么", 又能"直接拿来续写",
  不用每次重新训练。

【.pt 文件到底是什么?】
  就是 torch.save 用 pickle 存的**一个普通 Python dict**。
  我们当初存了 4 样东西:
      model_state  → 学到的权重 (OrderedDict: 名字 -> tensor)   ← 核心
      config       → 模型骨架超参 (重建模型用)
      vocab        → 字符↔id 映射 (encode/decode 用)
      final_loss   → 最后一步的 loss (一个记录)

【读它分两种目的】
  ① inspect (看内容): torch.load 拿到 dict, 按 key 打印。
  ② use (用模型续写): 光有数据还不够, 要三步把它变回能跑的模型:
        a. 按 config 搭一个一模一样的"空壳"   MiniGPT(**config)
        b. 把权重灌进去                       load_state_dict(model_state)
        c. 配上 vocab 做 encode/decode

【一个坑: weights_only】
  PyTorch 2.6 起 torch.load 默认 weights_only=True, 只允许加载纯张量。
  我们的 checkpoint 里有 config/vocab 这种普通 dict, 所以要显式传
  weights_only=False。**只对自己生成的、可信的文件这么做**(别人给的 .pt
  可能藏恶意代码, pickle 会执行它)。
"""

from importlib import import_module
from pathlib import Path

import torch

import common

# 06 的文件名以数字开头, 不能用普通 import, 借 importlib 按字符串导入。
# 复用它定义的 MiniGPT 类(以及 CKPT_DIR 路径)。
_m06 = import_module("06_train_and_generate")
MiniGPT = _m06.MiniGPT
CKPT_DIR = _m06.CKPT_DIR


def inspect(ckpt):
    """① 看 checkpoint 里有什么。"""
    print("=== ① Inspect: checkpoint 里有什么 ===\n")
    print(f"顶层 keys: {list(ckpt.keys())}")
    print(f"config   : {ckpt['config']}")
    print(f"final_loss: {ckpt['final_loss']:.4f}")

    char_to_id = ckpt["vocab"]["char_to_id"]
    preview = dict(list(char_to_id.items())[:5])
    print(f"vocab    : {len(char_to_id)} 个字符, 例如 {preview} ...")

    print("\n权重张量 (名字 -> 形状):")
    state = ckpt["model_state"]
    for name, tensor in state.items():
        print(f"  {name:35s} {tuple(tensor.shape)}")
    total = sum(t.numel() for t in state.values())
    print(f"\n总参数量: {total} ({total/1e3:.1f}K)")


def load_model(ckpt):
    """② 把 checkpoint 变回能跑的模型 — 三步走。"""
    print("\n=== ② Load: 把权重灌回模型 ===\n")
    # a. 按 config 搭一个一模一样的空壳
    model = MiniGPT(**ckpt["config"])
    # b. 把学到的权重灌进去
    model.load_state_dict(ckpt["model_state"])
    # c. 切到 eval 模式(本模型没 dropout/BN, 但这是好习惯)
    model.eval()
    print("模型已重建并加载权重 ✓")
    return model


def main():
    def add_args(p):
        p.add_argument(
            "--ckpt",
            type=str,
            default=str(CKPT_DIR / "06_minigpt.pt"),
            help="checkpoint 文件路径",
        )

    args = common.parse_args(add_args)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"找不到 checkpoint: {ckpt_path}")
        print("请先运行: uv run 06_train_and_generate.py --save_model")
        return

    # torch.load 拿到的就是当初 save 的那个 dict。
    # weights_only=False: 允许加载 config/vocab 这种非张量对象(仅限可信文件)。
    print(f"读取 {ckpt_path}\n")
    ckpt = torch.load(ckpt_path, weights_only=False)

    # ① 看内容
    inspect(ckpt)

    # ② 加载成模型
    model = load_model(ckpt)

    # ③ 用 vocab 做 encode/decode, 然后续写
    char_to_id = ckpt["vocab"]["char_to_id"]
    id_to_char = ckpt["vocab"]["id_to_char"]
    encode = lambda s: [char_to_id[c] for c in s]
    decode = lambda ids: "".join(id_to_char[i] for i in ids)

    print("\n=== ③ Generate: 用加载的模型续写 ===")
    prompts = ["the q", "pack ", "how v"]
    for prompt in prompts:
        ids = torch.tensor([encode(prompt)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=40, temperature=0.8)
        print(f"\nPrompt: '{prompt}'")
        print(f"续写: '{decode(out[0].tolist())}'")


if __name__ == "__main__":
    main()
