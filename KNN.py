import numpy as np


def KNN(inputX, dataSet, labels, k):
    dataset_num = dataSet.shape[0]
    # 这里 input.shape = (dataset_num, eign_num), 对应的是算法流程的矩阵B
    input = np.tile(inputX, (dataset_num, 1))

    # 对应算法流程第二步
    distance = input - dataSet

    # 对应算法流程第三步的矩阵
    sqdistance = distance ** 2
    # 各行分别对所有列元素求和成一列
    sqdisSum = sqdistance.sum(axis=1)
    # 距离大小从小到大排序 并且列出相应位置的index号
    argsortlist = sqdisSum.argsort()

    # 用字典存放
    classCount = {}
    for i in range(k):
        this_index = argsortlist[i]

        # thiskey: thisvalue + 1
        # classCount.get(key,defaultreturn=0)
        # defaultreturn: 如果找不到这个value，那返回0并且+1
        # 变成 thiskey:0+1
        classCount[labels[this_index]] = classCount.get(labels[this_index], 0) + 1
        print(classCount)
    # 原来书上代码：python2.7可用，python 3不可用
    # sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)

    # python3方法，iteritems()方法不再适用，改成items()
    # dict.items():key:value变成[(key(1),value(1)), ... ,(key(n),value(n))]
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)

    # sortedClassCount[keyindex][valueindex]
    return sortedClassCount[0][0]


# # 测试部分
# # 50个样本，有50个特征
# dataset = np.array([np.random.randint(1, 100, 50) for n in range(50)])
# # 50个样本标签，标签有3类，1/2/3
# label = np.random.randint(1, 4, 50)
#
# # 打印查看data
# print(f'dataset Matrix\n:{dataset}\n')
# print(f'label Martix\n:{label}\n')
#
# test_data = np.random.randint(1, 100, 50)
# print(f"this is test data martix:\n{test_data}")
#
# # 参数`10`：代表取前10个最近距离的数据的label来统计，帮test_data分类
# KNN(test_data, dataset, label, 10)
#
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataset[:,0],dataset[:,1],
#            15.0*label, 15.0*label)
# plt.show()