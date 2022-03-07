import torch
import numpy as np

a = torch.tensor([[-1], [-2]])
print(a.shape)
b = a.squeeze(-1)
print(b, b.shape)