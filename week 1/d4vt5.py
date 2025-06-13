import numpy as np
from matplotlib import pyplot as plt
x=[1,2,3,4]
y=[5,7,4,6]
x2=[1,2,3,4]
y2=[10,14,12,13]
plt.plot(x,y,label='Loss Per Month',color='r')
plt.plot(x2,y2,label='Profit Per Month',color='b')
plt.xlabel('Month',fontsize='xx-large')
plt.ylabel('Lakhs in Rupees',fontsize='xx-large')
plt.title('XYZ Company \n Loss-Profit',fontweight='bold')
plt.legend()
plt.show()