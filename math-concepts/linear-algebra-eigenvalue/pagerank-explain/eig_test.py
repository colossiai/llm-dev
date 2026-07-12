# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
# ]
# ///


import numpy as np

A = np.array([
    [2, 1],
    [1, 2]
])

values, vectors = np.linalg.eig(A)

print(values)
print(vectors)


v = vectors[:,0]

print('eigen-vector0 v=\n', v)
print('A @ v =\n', A @ v)
print('λ (values[0]) * v =\n', values[0] * v)

print('verify Av = λv')
print('Av - λv =\n', A @ v - values[0] * v)
