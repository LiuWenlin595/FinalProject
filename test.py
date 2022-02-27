import numpy as np
 
a = np.array([[[1],[2],[3]], [[4],[5],[6]],[[7],[8],[9]]])
print(a.shape)
c = a[[0, 0], [[1, 1], [2, 2], [0, 0]]]
print(c)
print(c.shape)
