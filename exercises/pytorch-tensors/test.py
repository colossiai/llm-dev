import numpy as np
import torch

v1 = torch.tensor([10., 20., 30., 40.])
print(v1)
print(v1.shape)
print(tuple(v1.shape))


v2 = v1.reshape(4, 1)
print(v2)
print(v2.shape)


v3 = torch.tensor([[100.], [200.], [300.]])  # (3, 1)
print(v3)
print(v3.shape)


imgs = torch.randn(5, 3, 4, 4)
# print(imgs)
print(imgs.shape)


a = torch.zeros(3, 1, 4)
b = torch.zeros(2, 4)
print((a + b).shape)


print('argmax() example')

logits = torch.tensor([
    [0.1, 0.5, 0.2, 0.8],   # 样本 0，4 个类别
    [0.9, 0.1, 0.3, 0.4],   # 样本 1
    [0.2, 0.7, 0.6, 0.3],   # 样本 2
])
# shape = (3, 4)

print(logits.argmax(dim=-1))
# → tensor([3, 0, 1])
#
#   样本 0 → 类别 3 (0.8 最大)
#   样本 1 → 类别 0 (0.9 最大)
#   样本 2 → 类别 1 (0.7 最大)

print(logits.argmin(dim=-1))