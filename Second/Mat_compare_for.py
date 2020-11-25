import time as tm
import numpy as np

# 数据长度(包含的元素个数)
dim = 300
x = np.ones(dim)
y = np.ones(dim)

z_for = 0.0
z_mat = 0.0

# for循环运算，开始计时
tStart = tm.clock()
# for循环解算x1*x2(对应元素相乘)
for i in range(dim):
    z_for += x[i]*y[i]
# 停止计时
tEnd=tm.clock()
# for循环计算用时
tFor = tEnd-tStart

# 矩阵运算，开始计时
tStart = tm.clock()
# 矩阵计算x1*x2(对应元素相乘)
z_mat = np.dot(x, y.T)
# 停止计时
tEnd = tm.clock()
# 计算numpy矩阵运算用时
tMatrix = tEnd-tStart

print(f'for循环用时tFor = {tFor}')
print(f'矩阵运算用时tMatrix = {tMatrix}')
print(f'运算用时tFor-tMatrix = {tFor-tMatrix}\ntFor与tMatrix的比值：{tFor/tMatrix}')
print('运算结果的差(所有元素累积和)z-z_mat = ',z_for-z_mat)