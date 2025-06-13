import numpy as np
from matplotlib import pyplot as plt
x=np.arange(1,11)
y=2*x+5
plt.title('MatplotLib demo')
plt.xlabel('X axis caption')
plt.ylabel('Y axis caption')
# plt.plot(x,y)
# plt.plot(x,y,'ob')
plt.plot(y,'r+')
plt.show()
