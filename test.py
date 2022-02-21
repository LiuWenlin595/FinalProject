import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]
c = []
c.append(a)
c.append(b)

d = [7, 8, 9]
e = [10, 11, 12]
f = []
f.append(d)
f.append(e)

g = []
g.append(c)
g.append(f)

print(g)
print(np.array(g).shape)
print(g[:1])