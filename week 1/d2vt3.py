number=list(range(1,21))
for num in number[:]:
    if num %2 or num %4 != 0:
        number.remove(num)
    print ("Numbers from 1 to 20 that is divisible by either 2 or 4 are",number)
