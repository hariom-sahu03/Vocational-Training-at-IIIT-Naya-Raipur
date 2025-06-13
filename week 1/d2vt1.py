thislist = ['apples' ,'banana','cherry']
print(thislist)
thistuple = ('apples','banana',1,2,3,3)
print(thistuple)
thisset={'apples','banana',1,2,3,}
print(thisset)
thisdict = {
    "name":"HARIOM",
    "age":20
}
print(thisdict)
print(len(thislist))
print(type(thislist))
print(thislist[1:2])
print(thislist[1])


mylist=["apple","Banana","chery","orange","kivi"]
if "apple" in mylist:
    print("YES!,apple is present in mylist")


mylist.append("Mango")
print (mylist)
mylist.remove("kivi")
print(mylist)
mylist.sort()
print(mylist)

numlist=[15,50,5,78,90,909,5015,222,741]
numlist.sort(reverse=true)
print (numlist)