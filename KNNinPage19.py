# import numpy as np
# import operator
#
# def classfly0(inX, dataSet, labels, k):
#     dataSetSize = dataSet.shape[0]
#     print(dataSetSize)
#     # 代表inX这个List，重复dataSetSize行
#     # 比如说inX.shape = (50,), dataSet.shape=(50, 50)，而dataSetSize = 50
#     # 那么np.tile(inX, (dataSetSize, 1))，即相当于inX.shape会变成(50, 50)，和dataSet.shape同步
#     # np.tile(inX, (dataSetSize, 1)) = [inX , ,,, inX],即总共有50个inX
#     # 或许dataSet.shape=(50,50)
#
#     # 计算50个特征距离
#     # [
#     # [x1-y1_1, x2-y1_2 ..., x50-y1_50],
#     # ........
#     # [x1-y50_1, x2-y50_2 ..., x50-y50_50]
#     # ]
#     diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
#     #
#     # [
#     # [(x1-y1_1)^2, (x2-y1_2)^2 ..., (x50-y1_50)^2],
#     # ........
#     # [(x1-y50_1)^2, (x2-y50_2)^2 ..., (x50-y50_50)^2]
#     # ]
#     sqDiffMat = diffMat ** 2
#     print(sqDiffMat)
#     # [
#     # [(x1-y1_1)^2+(x2-y1_2)^2+...+(x50-y1_50)^2],
#     # ........
#     # [(x1-y50_1)^2+(x2-y50_2)^2+...+(x50-y50_50)^2]
#     # ]
#     # （axis=1）表示同一行里面不同列的元素相加
#     # （axix=0）表示同一列里面不同行元素相加
#     sqDistance = sqDiffMat.sum(axis=1)
#     print(sqDistance)
#     # [
#     # [((x1-y1_1)^2+(x2-y1_2)^2+...+(x50-y1_50)^2)^0.5]
#     # ........
#     # [((x1-y50_1)^2+(x2-y50_2)^2+...+(x50-y50_50)^2)^0.5]
#     # ]
#     distance = sqDistance ** 0.5
#     print(distance)
#     # 把distance从小到大的dataSet_index从前往后排，并生成一个list放置indices
#     sortedDisIndices = distance.argsort()
#     print(sortedDisIndices)
#
#     # 字典，key:value对应
#     classCount = {}
#
#     for i in range(k):
#         # 找出对应index的label
#         VoteIlabel = labels[sortedDisIndices[i]]
#         # 找出label之后放进classCount里面
#         # 说明：classCount{key=labelname：,value=this_label_count}
#         # 1.如果classCount里面get到有该key，那么value+=1，如果找不到有，那么返回0，然后+1
#         classCount[VoteIlabel] = classCount.get(VoteIlabel, 0) + 1
#
#     # for循环处理完成后，classCount里面会存放前k个距离最小的，它们的label(key)以及对应统计出现的次数(value)
#     # 比如说：{'苹果':4, '橙子':7, '梨':3}
#
#
#     # 然后用一个sorted函数来对classCount这个字典的value大小，从大到小进行排序
#     # 最后输出一个key，作为该算法对输入数据的特征作出"判断"后预测的类别
#     # 原来代码：python2.7可用，python 3不可用
#     # sortedclassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
#
#     # python3方法，iteritems()方法不再适用，改成items()
#     # dict.items():key:value变成[(key(1),value(1)), ... ,(key(n),value(n))]
#     sortedclassCount = sorted(classCount.items(),key = lambda item:item[1],reverse=True)
#     print(f'sortedclassCount is {sortedclassCount}')
#     print(f'sortedclassCount[0][0] is {sortedclassCount[0][0]}')
#     return sortedclassCount[0][0]
#
# # 测试classfly0后半部分，classCount
# # a = {'苹果':4, '橙子':7, '梨':3}
# # # python2.7版本可用
# # # sortedclassCount = sorted(a.iteritems(), key=operator.itemgetter(1), reverse=True)
# # # python3，自己的方法
# # sortedclassCount = sorted(a.items(),key = lambda item:item[1],reverse=True)
# # print(f'sortedclassCount is {sortedclassCount[0][0]}')
#
# # # 测试classfly0函数前半部分
# # # 50个样本，有50个特征
# # # [50......]
# # # [12312...]
# # data_50 = np.array([np.random.randint(1, 100, 50) for n in range(51)])
# # # 50个样本标签，标签有3类，1/2/3
# # label_50 = np.random.randint(1, 4, 50)
# #
# # print(data_50)
# # print(label_50)
# #
# # test_data_1 = np.random.randint(1, 100, 50)
# # print(f"this is test data martix:\n{test_data_1}")
# #
# # classfly0(test_data_1, data_50, label_50, 40)
# # # 测试classfly0函数前半部分----END
#
#
# # # 测试classfly0函数前半部分
# # 50个样本，有50个特征
# # [50......]
# # [12312...]
# # data_50 = np.array([np.random.randint(1, 100, 50) for n in range(50)])
# # # 50个样本标签，标签有3类，1/2/3
# # label_50 = np.random.randint(1, 4, 50)
# #
# # print(data_50)
# # print(label_50)
# #
# # test_data_1 = np.random.randint(1, 100, 50)
# # print(f"this is test data martix:\n{test_data_1}")
# #
# # classfly0(test_data_1, data_50, label_50, 10)
# # # 测试classfly0函数前半部分----END
#
# #np.array参数不加,dtype = 'float32'，会发生
# # Python：ufunc ‘subtract‘ did not contain a loop with signature matching types dtype(u11)
# a = np.array([25607,8.680527,0.086955])
# print(a)
# from File2Matrix import file2Matrix
# file_path = "C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\datingTestSet2.txt"
# count, data_list, label_list = file2Matrix(file_path)
# #np.array参数不加,dtype = 'float32'，会发生
# # Python：ufunc ‘subtract‘ did not contain a loop with signature matching types dtype(u11)
# data_list_np = np.array(data_list)
# label_list_np = np.array(label_list)
#
# classs = classfly0(a,data_list_np,label_list_np,10)
# print(classs)

#
# # 关于tile函数
# # 先创建一个向量和一个矩阵来进行使用
# a = np.array([1, 2, 3, 4])  # 经过测试（Python3下1.13.0版本numpy），这里使用python中的列表形式也可以，最终也会被tile()函数转化成numpy中的array形式。
# # 第一种使用方法
# x1 = np.tile(a, 2)
# #    ->  array([1, 2, 3, 4, 1, 2, 3, 4])
# # 第二种方法
# # (2,1)第一行里面重复1次，然后有2行这样的第一行矩阵
# x2 = np.tile(a, (2, 1))
# #    -> array([[1, 2, 3, 4], [1, 2, 3, 4]])
# # (2,2,1)第一行里面重复1次（参数里面第三个数字“1”），”这个矩阵“里面然后有2行这样的第一行矩阵(参数里面第二个“2”)，”这个矩阵“有两个(参数里面第一个"2")
# x2 = np.tile(a, (2, 2, 1))
# #    -> array([
# #              [[1, 2, 3, 4], [1, 2, 3, 4]],
# #              [[1, 2, 3, 4], [1, 2, 3, 4]]
# #             ])


#尝试用pandas来读取数据
data = "C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\datingTestSet2.txt"
import pandas as pd

data2 = pd.read_csv(data,sep='\t')

print(data2.head())
