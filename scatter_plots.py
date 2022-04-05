import matplotlib.pyplot as plt
import numpy as np

# Generate scatter plot of independent vs Dependent variable
plt.style.use('ggplot')
fig = plt.figure(figsize = (18, 18))

data = np.loadtxt("example.data", dtype = float)
 
for index in range(0, data.shape[1]-1):
    ax = fig.add_subplot(3, 2, index + 1)
    ax.scatter(data[:, index], data[:, 6])
    ax.set_ylabel('y', size = 10)
    ax.set_xlabel('x' + str(index), size = 10)
 
plt.show()
