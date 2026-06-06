"""
Manual softmax implementation in pure Python (no torch / numpy).

定义:
    softmax(x)_i = exp(x_i) / sum_j exp(x_j)

数值稳定性技巧:
    softmax(x) == softmax(x - max(x))    (恒等式)
    先减最大值再 exp，避免 exp(1000) 这种溢出。

Run:
    uv run python manual-softmax.py
"""

import math


# ============================================================
# Implementation
# ============================================================

def manual_softmax(xs: list[float]) -> list[float]:
    """1D softmax，含数值稳定性 (减最大值)."""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def softmax_batch(rows: list[list[float]]) -> list[list[float]]:
    """对 2D 矩阵的每一行独立做 softmax."""
    return [manual_softmax(row) for row in rows]


# ============================================================
# Test helpers
# ============================================================

def approx_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def lists_close(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    return len(a) == len(b) and all(approx_eq(x, y, tol) for x, y in zip(a, b))


# ============================================================
# Tests
# ============================================================

def test_basic_correctness():
    """对照手算公式 exp(x_i) / sum(exp(x_j))."""
    xs = [2.0, 1.0, 0.1]
    expected_denom = math.exp(2.0) + math.exp(1.0) + math.exp(0.1)
    expected = [
        math.exp(2.0) / expected_denom,
        math.exp(1.0) / expected_denom,
        math.exp(0.1) / expected_denom,
    ]
    got = manual_softmax(xs)
    assert lists_close(got, expected), f"got {got}, expected {expected}"
    print(f"test_basic_correctness ✓  softmax({xs}) = {[round(v, 4) for v in got]}")


def test_sum_to_one():
    """每个输出必须加起来等于 1."""
    cases = [
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0],
        [0.0, 0.0, 0.0],
        [1.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ]
    for xs in cases:
        p = manual_softmax(xs)
        assert approx_eq(sum(p), 1.0), f"sum != 1 for {xs}: got {sum(p)}"
    print(f"test_sum_to_one ✓  ({len(cases)} cases)")


def test_all_in_unit_interval():
    """每个输出应该落在 (0, 1) 区间内."""
    for xs in [[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0], [0.5, 0.5]]:
        p = manual_softmax(xs)
        for v in p:
            assert 0.0 < v < 1.0, f"概率越界: {v} from {xs}"
    print("test_all_in_unit_interval ✓")


def test_order_preserved():
    """softmax 不改变大小顺序: argmax 不变."""
    xs = [0.1, 5.0, 2.0, 3.0, 0.5, 4.0]
    p = manual_softmax(xs)
    assert p.index(max(p)) == xs.index(max(xs))
    # 完整顺序也应该一致
    rank_x = sorted(range(len(xs)), key=lambda i: -xs[i])
    rank_p = sorted(range(len(p)), key=lambda i: -p[i])
    assert rank_x == rank_p
    print(f"test_order_preserved ✓  argmax={p.index(max(p))}")


def test_translation_invariance():
    """softmax(x) == softmax(x + c)，平移不变性 → 也是稳定性技巧的依据."""
    xs = [1.0, 2.0, 3.0]
    p1 = manual_softmax(xs)
    p2 = manual_softmax([x + 100.0 for x in xs])
    p3 = manual_softmax([x - 50.0 for x in xs])
    assert lists_close(p1, p2)
    assert lists_close(p1, p3)
    print("test_translation_invariance ✓  softmax(x) == softmax(x + 100) == softmax(x - 50)")


def test_numerical_stability():
    """大数输入不应溢出 (没有减最大值的朴素实现会得到 nan)."""
    xs = [1000.0, 999.0, 998.0]
    p = manual_softmax(xs)
    assert approx_eq(sum(p), 1.0), f"sum != 1: {sum(p)}"
    assert all(0.0 < v < 1.0 for v in p), f"溢出: {p}"
    assert p.index(max(p)) == 0  # 最大输入对应最大概率

    # 朴素实现对比 (会溢出)
    try:
        naive = [math.exp(x) for x in xs]
        naive_probs = [e / sum(naive) for e in naive]
        print(f"  朴素实现意外没崩: {naive_probs}")
    except OverflowError:
        print("  朴素实现 OverflowError (预期内) ✓")

    print(f"test_numerical_stability ✓  正确处理 exp(1000)")


def test_uniform_input():
    """所有 logit 相等 → 均匀分布."""
    p = manual_softmax([7.0, 7.0, 7.0, 7.0])
    expected = [0.25] * 4
    assert lists_close(p, expected)
    print("test_uniform_input ✓  相同 logit → 均匀分布")


def test_temperature():
    """高温 → 分布更平; 低温 → 分布更尖."""
    xs = [3.0, 2.0, 1.0, 0.0]
    p_sharp = manual_softmax([x / 0.5 for x in xs])   # T = 0.5
    p_normal = manual_softmax(xs)                      # T = 1.0
    p_flat = manual_softmax([x / 5.0 for x in xs])    # T = 5.0

    # 尖度: 最大概率从大到小: sharp > normal > flat
    assert max(p_sharp) > max(p_normal) > max(p_flat)
    print(
        f"test_temperature ✓  "
        f"max(T=0.5)={max(p_sharp):.3f} > "
        f"max(T=1)={max(p_normal):.3f} > "
        f"max(T=5)={max(p_flat):.3f}"
    )


def test_negative_values():
    """负值也应该工作正常."""
    xs = [-10.0, -5.0, 0.0]
    p = manual_softmax(xs)
    assert approx_eq(sum(p), 1.0)
    assert p[2] > p[1] > p[0]  # 0 > -5 > -10，概率单调
    print("test_negative_values ✓")


def test_two_element():
    """二元 softmax 等价于 sigmoid 的形式."""
    # softmax([a, b])[1] = exp(b)/(exp(a)+exp(b)) = 1/(1+exp(a-b)) = sigmoid(b-a)
    a, b = 1.5, 0.3
    p = manual_softmax([a, b])
    sigmoid_eq = 1.0 / (1.0 + math.exp(a - b))
    assert approx_eq(p[1], sigmoid_eq)
    print(f"test_two_element ✓  softmax([{a},{b}])[1] = sigmoid({b - a}) = {p[1]:.4f}")


def test_batched():
    """对 2D 矩阵每一行分别做 softmax."""
    mat = [
        [1.0, 2.0, 3.0],
        [10.0, 10.0, 10.0],
        [-1.0, 0.0, 1.0],
    ]
    out = softmax_batch(mat)
    for row in out:
        assert approx_eq(sum(row), 1.0)
    # 第二行全相等 → 均匀
    assert lists_close(out[1], [1 / 3, 1 / 3, 1 / 3])
    print(f"test_batched ✓  ({len(mat)} rows，每行和为 1)")


# ============================================================
# Optional: cross-check against torch (only if available)
# ============================================================

def test_matches_torch():
    """如果安装了 torch，对比 PyTorch 的 F.softmax 结果."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("test_matches_torch ⊘  (torch 未安装，跳过)")
        return

    cases = [
        [2.0, 1.0, 0.1],
        [1000.0, 999.0, 998.0],
        [0.0, 0.0, 0.0, 0.0],
        [-3.0, -2.0, -1.0, 0.0, 1.0],
    ]
    for xs in cases:
        mine = manual_softmax(xs)
        theirs = F.softmax(torch.tensor(xs), dim=-1).tolist()
        assert lists_close(mine, theirs, tol=1e-6), f"mismatch on {xs}: {mine} vs {theirs}"
    print(f"test_matches_torch ✓  ({len(cases)} cases 都和 F.softmax 一致)")


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Manual softmax — tests")
    print("=" * 60)

    test_basic_correctness()
    test_sum_to_one()
    test_all_in_unit_interval()
    test_order_preserved()
    test_translation_invariance()
    test_numerical_stability()
    test_uniform_input()
    test_temperature()
    test_negative_values()
    test_two_element()
    test_batched()
    test_matches_torch()

    print("=" * 60)
    print("All tests passed ✓")
    print("=" * 60)

    # Quick demo
    print("\nDemo:")
    for logits in ([2.0, 1.0, 0.1], [0.0, 0.0, 0.0], [5.0, 2.0, -3.0]):
        probs = manual_softmax(logits)
        print(f"  softmax({logits}) = {[round(p, 4) for p in probs]}  sum={sum(probs):.6f}")
