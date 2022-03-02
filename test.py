import numpy as np

a = np.array([1, 2, 3])
b = np.sum(a)
c = a / b
print(c)
print(np.random.choice(3, 2, False, p=c))
