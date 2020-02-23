import pandas
from matplotlib import pyplot as plt

x= [1,2,3,4,5,6,7]
y= [1.2, 3.5, 2.4, 3.4, 4.2, 5.5, 6]
plt.plot(x,y)
plt.title("Plot chart", fontsize=8 , color='g')
plt.xlabel('age')
plt.ylabel('height')
plt.show()

