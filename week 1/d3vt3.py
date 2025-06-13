import numpy as np
arr=np.array([1,2,3,4,5])
print("1st",arr[1])
print("2nd",arr[2]+arr[3])

arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print("3rd",arr[1,4])


arr2=np.array([[[1,2,3],
                [4,5,6],
                [7,8,9]]])
print("4th",arr2[0,1,2])


arr3=np.array([1,2,3,4,5,6,7])
print("5th",arr3[0:5])
print("6th",arr3[4:])
print("7th",arr3[:4])
print("8th",arr3[-3:-1])
print("9th",arr3[0:5:2])

arr4=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print("10th",arr[1,1:4])
print(arr[0:2,2])
