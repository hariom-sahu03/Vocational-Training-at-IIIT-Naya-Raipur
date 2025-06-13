import numpy as np
arr=np.array([4,6,7,9,8,1])
x=np.searchsorted(arr,7,'right')
print(x)