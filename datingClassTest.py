import numpy as np
from File2Matrix import file2Matrix
from KNN import KNN
from AutoNormalize import auto_norm

import random

def datingClassTest():
    DataRatio = 0.1
    dtype = "float32"

    # count, data_list, label_list = file2Matrix("C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\datingTestSet2.txt",shuffle=False)
    count2, data_list2, label_list2 = file2Matrix("C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\datingTestSet.txt",shuffle=True)
    #data_list_np = np.array(data_list,dtype="float32")
    data_list2_np = np.array(data_list2,dtype=dtype)

    try:
        #label_list_np = np.array(label_list,dtype="float32")
        label_list2_np = np.array(label_list2,dtype=dtype)
    except:
        #label_list_np = np.array(label_list)
        label_list2_np = np.array(label_list2)
        print(f"error, label_list_np cannot convert to np.array as dtype={dtype}")

    #data_list_np_norm, ranges, minVals = auto_norm(data_list_np)
    data_list2_np_norm, ranges, minVals = auto_norm(data_list2_np)


    m = count2

    num_test_vec = int(m * DataRatio)

    errorCount = 0.0

    #print(data_list_np_norm)

    dic = {'didntLike':1, 'smallDoses':2, 'largeDoses':3}

    for i in range(num_test_vec):
        classiflyResult = KNN(data_list2_np_norm[i,:],data_list2_np_norm[num_test_vec:m,:],label_list2_np,20)
        # print(str(classiflyResult))
        # print(str(label_list2_np[i]))
        print(f"the classiflier predict class:{classiflyResult}, and the real result:{label_list2_np[i]}")
        if classiflyResult != label_list2_np[i]:
            errorCount += 1.0
            print("error")

    print(f"total error rate:{errorCount/float(num_test_vec)}")


if __name__ == '__main__':
    datingClassTest()