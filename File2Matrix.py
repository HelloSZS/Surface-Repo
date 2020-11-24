import numpy as np
import gc

import random


# def file2Matrix(file_path):
#     egin_list = []
#     label_list = []
#     count = 0
#
#     with open(file_path) as inline:
#         #in this data("dataTestSet2"), there are 3 egins and 1 label in a line
#         for line in inline:
#             #文档里面用制表符作为每行里两个数据之间的分隔符
#             a,b,c,d= line.split('\t')
#
#             #因为d = '数字\n'
#             #所以d[0] = '数字'
#             egin_list.append([a, b, c])
#             label_list.append(d[0])
#             count += 1
#     #垃圾回收，解引用。并且启用垃圾回收函数回收没有被引用的对象，释放内存。
#     del a,b,c,d
#     gc.collect()
#
#     return count, np.array(egin_list),np.array(label_list)


def file2Matrix(file_path, shuffle=False):
    data_list = []
    label_list = []
    count = 0

    with open(file_path) as inline:
        for line in inline:
            # strip()，用于移除字符串头尾指定的字符
            # 可移除末尾的"\n"换行符
            line = line.strip()

            # 分割之后a1,a2,a3,label都是string类型
            # 所以要转换成float(label要转换成int)，再放进列表
            # 不然到后面运算错误
            a1, a2, a3, label = line.split("\t")
            data_list.append([float(a1), float(a2), float(a3),label])
            count += 1

    if shuffle == True:
        random.shuffle(data_list)

    data_list = np.array(data_list)

    label_list = data_list[:, 3]
    data_list = data_list[:, :3]

    data_list = data_list.tolist()
    label_list = label_list.tolist()

    return count, data_list, label_list

# #list转换成nparray
# count, data, label = file2Matrix("C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\datingTestSet2.txt")
# print("data_list：\n",np.array(data))
# print("label_list：\n",np.array(label))
# # print("count：\n",count)
