import numpy as np
arr=np.array([1,2,3,4,5,6,7,8,9])
newarr=np.array_split(arr,3)
print(newarr,"\n")


arr1=np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18]])
newarr1=np.array_split(arr1,3,axis=1)
print(newarr1)