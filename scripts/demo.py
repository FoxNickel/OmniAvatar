import torch

# 创建3个形状相同的2D张量 (2x2)
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.tensor([[9, 10], [11, 12]])

# 在第0维堆叠 (新增维度在最前面)
stacked_0 = torch.stack([a, b, c], dim=0)
print(stacked_0.shape)  # 输出: torch.Size([3, 2, 2])
print(stacked_0)

# 在第1维堆叠 (新增维度在中间)
stacked_1 = torch.stack([a, b, c], dim=1)
print(stacked_1.shape)  # 输出: torch.Size([2, 3, 2])
print(stacked_1)

# 在第2维堆叠 (新增维度在最后面)
stacked_2 = torch.stack([a, b, c], dim=2)
print(stacked_2.shape)  # 输出: torch.Size([2, 2, 3])
print(stacked_2)