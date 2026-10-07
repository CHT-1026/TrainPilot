import torch

x = torch.tensor(2.0, device="cuda")
target = torch.tensor(1.0, device="cuda")
w = torch.tensor(3.0, device="cuda", requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)
r = 5
while r != 0:
    optimizer.zero_grad()
    prediction = w * x  # TODO：用 w 和 x 计算预测值
    loss = pow(prediction-target, 2)        # TODO：计算平方误差
    print("prediction:", prediction.item())
    print("loss:", loss.item())
    loss.backward()
    learning_rate = 0.1
    optimizer.step()
    print(w.grad)
    print(w)
    r -= 1


