mylist=[1,2,1,2,3,3,3,3,3,4,5,6,7,8,8,8,9,9,10]
target=3

for i in range(len(mylist)):
    if mylist[i] == target :
        print("Target found at index",i)

x=mylist.count(3)
print("THE NO OF REPEATION OF 3 IS ",x)

