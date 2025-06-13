my_list=['data',2,3]
my_list.append(4)
print(my_list)


import array
my_array=array.array('u',["a","b","c"])
my_array.append("d")
print (my_array)
for i in range(0,len(my_array)):
    print(my_array[i],end="")
    print("\n")