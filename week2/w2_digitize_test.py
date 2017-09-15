import numpy as np

m = 1
n = 2
p = [0.1, 0.2, 0.3, 0.4]
p = np.array(p)
edges = np.r_[0, p.cumsum()]
print edges
print np.digitize(np.random.rand(m, n), edges)


