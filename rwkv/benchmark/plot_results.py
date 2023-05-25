#%%
import numpy as np
import matplotlib.pyplot as plt

result = np.loadtxt("parallel_scan.txt")[::-1]
x = [10, 20 , 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

# %%
plt.plot(x, result, 'o-')
plt.xscale('log')
plt.xlabel('Number of tokens')
plt.ylabel('Time (s)')
# %%
