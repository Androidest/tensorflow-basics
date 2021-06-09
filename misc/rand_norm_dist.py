#%%
# Random with normal distribution 
import matplotlib.pyplot as plt
import numpy as np

# 身高 170 为均值，68.26% 左右的身高在 170 ± 10	之间，
# 身高取值范围基本在：170 ± 3*10 （即半径为 3*Sigma） 
# mean=170， sigma=10， arrayCount=1000000
gausses = np.random.normal(170, 10, 1000000)  

plt.figure('fig1')          #添加一个窗口
plt.hist(gausses, bins=100, normed=False, histtype='bar', facecolor='red', alpha=0.2)

plt.figure('fig2')
plt.plot(gausses)

plt.show()
# %%
