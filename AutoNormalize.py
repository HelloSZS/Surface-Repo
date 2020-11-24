import numpy as np
from File2Matrix import file2Matrix

# 数值归一化
def auto_norm(dataSet):
    # max() and min() are numpy array's function
    #min(0)找出每列最小值
    #比如a = [[1, 4, 5], [2, 3, 4]]
    #a.min(0) = [1 3 4]

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    #dataSet.shape = (1000, 3)
    #不加这个，下面出来的normDataSet = dataSet - np.tile(minVals, (m, 1))
    #会变成array类型，而不是np.array
    normDataSet = np.zeros(np.shape(dataSet))

    #dataSet.shape[0] = 1000
    m = dataSet.shape[0]

    #minVals.shape = (1, 3)
    #maxVals.shape = (1, 3)
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


# # #test
# count, data_list, label_list = file2Matrix("C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\datingTestSet2.txt")
#
#
# normDataSet, ranges, minVals = auto_norm(np.array(data_list))
