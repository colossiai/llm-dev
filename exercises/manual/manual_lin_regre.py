data = [
    (1, 3),
    (2, 5),
    (3, 7),
    (4, 9)
]

w = 0.0
b = 0.0

lr = 0.01

n = len(data)

for epoch in range(3000):
    dw = 0.0
    db = 0.0
    loss = 0.0

    for x, y in data:
        y_hat = w * x + b
        error = y_hat - y
        loss += error ** 2

        dw += 2 * error * x
        db += 2 * error
    
    loss /= n
    dw /= n
    db /= n

    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss:.16f}, w={w:.16f}, b={b:.16f}")

print(f"Final parameters: w={w:.16f}, b={b:.16f}")