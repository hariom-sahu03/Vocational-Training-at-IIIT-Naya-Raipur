from matplotlib import pyplot as plt
population_ages=[20,30,25,35,38,4,40,44,65,45,32,55,12,32,18,21,65,85,96,89,87,58,54,33,22,44,35,39,68,67,67,66,65,5]
bins=[0,10,20,30,40,50,60,70,80,90,100]
plt.hist(population_ages,bins,histtype='bar',rwidth=0.8,label='Count')
plt.xlabel("Ages in Bins of 10 Years")
plt.ylabel('Number of person')
plt.legend()
plt.show()