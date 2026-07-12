# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///

import numpy as np

# ============================================================
# 真实应用: PageRank
#   5 个网页, 谁链接谁 (from -> to)。这是一张真实的有向网络。
# ============================================================
PAGES = ["A", "B", "C", "D", "E"]
# 邻接: LINKS[i] = i 指向的页面下标列表 
LINKS = {
    0: [1, 2, 3],   # A -> B, C, D
    1: [0, 3],      # B -> A, D
    2: [0],         # C -> A          (C 把全部"投票"都投给了 A)
    3: [1, 2],      # D -> B, C
    4: [0, 3],      # E -> A, D       (没有任何人链接到 E)
}
DAMP = 0.85         # Google 的阻尼系数 (随机上网者有 15% 概率随机跳转)


def build_google_matrix():
    """构造列随机的 Google 矩阵 G, 其【特征值=1】的特征向量就是 PageRank。"""
    n = len(PAGES)
    M = np.zeros((n, n))
    for src, outs in LINKS.items():
        print(f"src={src}, outs={outs}")

        if outs:
            for dst in outs:
                M[dst, src] = 1.0 / len(outs)   # 列 src 平均分给它指向的页面
        else:
            M[:, src] = 1.0 / n                 # 没有出链的页面 -> 均匀跳转
    print("M =\n", M)

    G = DAMP * M + (1 - DAMP) / n * np.ones((n, n))
    return G


def compute_pagerank():
    """用两种办法算 PageRank, 互相印证: (1) 直接特征分解  (2) 幂迭代。"""
    G = build_google_matrix()

    # 办法1: 特征分解, 取特征值最接近 1 的那个
    vals, vecs = np.linalg.eig(G)
    print("全部特征值 vals =", vals)
    print("特征值的模 |λ| =", np.abs(vals))
    idx = np.argmin(np.abs(vals - 1.0))
    lambda_vec = vals[idx].real
    r_eig = np.abs(vecs[:, idx].real)
    r_eig = r_eig / r_eig.sum()

    # 办法2: 幂迭代 (真实工程里就是这么算的, 因为网页矩阵太大)
    n = len(PAGES)
    r = np.ones(n) / n
    for _ in range(100):
        r = G @ r
    r_power = r / r.sum()

    print("G =\n", G)
    print("特征分解法: λ = ", lambda_vec)
    print("特征分解法: r = ", r_eig)
    print("幂迭代法: r = ", r_power)


def main():
    compute_pagerank()

if __name__ == "__main__":
    main()
