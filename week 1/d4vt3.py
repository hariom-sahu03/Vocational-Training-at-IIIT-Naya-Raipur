import numpy as np
from matplotlib import pyplot as plt
x=np.arange(0,3*np.pi,0.1)
y_sin=np.sin(x)
y_cos=np.cos(x)
plt.title('Sine  Wave form')
plt.plot(x,y_sin)
plt.figure()
plt.title('cos wave')
plt.plot(x,y_cos,"--k")
plt.figure
plt.show()
