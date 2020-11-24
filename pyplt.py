import matplotlib.pyplot as plt
from File2Matrix import file2Matrix
import numpy as np

file_path = "C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\datingTestSet2.txt"
count, data_list, label_list = file2Matrix(file_path)


fig = plt.figure()
ax = fig.add_subplot(111)


# ----------------------------------------------------------------------------------------
# 错误1。。。。。。。。。。没有转换成numpy.array，所以切片(slice)不能切
# data_list.type = list
# label_list.type = list

# ax.scatter(data_list[:,0], data_list[:,1],
#            15.0*label_list,15.0*label_list)

# error:
# Traceback (most recent call last):
#   File "C:/Users/hello/PycharmProjects/pythonProject/pyplt.py", line 13, in <module>
#     ax.scatter(data_list[:,0], data_list[:,1],
# TypeError: list indices must be integers or slices, not tuple
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# 错误2。。。。。。。。。。。每有在转换成numpy.array时把dtype转成"float32"或者"float64"
# data_list_np = np.array(data_list)
# label_list_np = np.array(label_list)
# ax.scatter(data_list_np[:,0], data_list_np[:,1],
#           15.0*label_list_np,15.0*label_list_np)
#
# Traceback (most recent call last):
#   File "C:/Users/hello/PycharmProjects/pythonProject/pyplt.py", line 32, in <module>
#     15.0*label_list_np,15.0*label_list_np)
# TypeError: ufunc 'multiply' did not contain a loop with signature matching types dtype('<U32') dtype('<U32') dtype('<U32')
#
# ----------------------------------------------------------------------------------------

#can run test
# data_list_np = np.array(data_list, dtype="float32")
# label_list_np = np.array(label_list, dtype="float32")
# ax.scatter(data_list_np[:,1], data_list_np[:,2],
#           15.0*label_list_np,15.0*label_list_np)
# plt.show()