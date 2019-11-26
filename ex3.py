import numpy as np
ar = np.array([
    [0,2,3,0,0,0,0,0,0]
])


tmp = np.concatenate((ar,ar,ar))

tmp = tmp.reshape(-1,3,3,3)
print(tmp)