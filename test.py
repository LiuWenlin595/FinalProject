import torch
import numpy as np

a = torch.tensor([[-1], [-2]])
b = torch.maximum(a, -a)
print(b)

print((2/5)**2)