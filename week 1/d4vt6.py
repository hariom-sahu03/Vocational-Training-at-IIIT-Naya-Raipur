from matplotlib import pyplot as plt
x=[1,3,5,7]
y=[5,7,4,6]
x2=[2,4,6,8]
y2=[10,14,12,13]
plt.bar(x,y,label="Loss per month",color='r')
plt.bar(x2,y2,label='Profit per month',color='b')
plt.xlabel('Month',fontsize='xx-large')
plt.ylabel('Lakhs in rupees',fontsize='xx-large')
plt.title("XYZ comapany \n Loss Profit",fontweight='bold')
plt.legend()
plt.show()