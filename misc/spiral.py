# %%
import numpy as np
import matplotlib.pyplot as plt

m_dim = 1000        # m_dim 为螺旋图坐标的数据的数量，数值越大螺旋图越光滑
n_dim = 3          # 矩阵 T 的行数 n, 即 paper 中的 n

# 创建螺旋图，二维数据
def create_spiral(m_dim):
    x = []
    y = []
    for theta in np.linspace(0, 20 * np.pi, m_dim):
        r = ((theta)**2)
        # 计算螺旋图的坐标
        x.append(r*np.cos(theta)) 
        y.append(r*np.sin(theta))
    return np.array([x,y])

# ReLU 激活函数
def relu(input):
    return np.maximum(input, 0)

# 显示二维数据
def show_image(img, title):
    [x, y] = img
    plt.title(title)
    plt.plot(x, y) 
    plt.show()


input = create_spiral(m_dim)         # shape=(2, m_dim)
show_image(input, 'Input')           #显示螺旋图，其中隐藏着兴趣流 （mainfold of interest）

# 随机一个 n x 2 矩阵T
T = np.random.rand(n_dim, 2) - 0.5   # shape=(n_dim, 2), 随机值范围（-0.5，0.5）
layer_output = np.dot(T, input)      # y = Tx

# ReLU 激活
layer_output = relu(layer_output) 

# 计算 T 的广义逆转矩阵 T^(−1)
T_inv = np.linalg.pinv(T)
output = np.dot(T_inv, layer_output) # 映射回二维数据：x` = T^(−1)y 

show_image(output, 'Output/dim='+str(n_dim))




# %%
