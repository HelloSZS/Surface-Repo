import numpy as np
import os
from img2mat import img2mat
from KNN import KNN

def count(dir_list):
    alldata_count = 0
    count_class = {}

    # 从文件名中读取label与对应数据个数
    for dir in dir_list:
        tmp = dir.split('.')
        classs, _ = tmp[0].split('_')
        count_class[classs] = count_class.get(classs,0) + 1
        alldata_count +=1

    return alldata_count, count_class

def file_path_list_concat(dir,dict):
    path_list = []
    for i in range(10):
        classi_filenumber = dict.get(str(i))
        for j in range(classi_filenumber):
            filename = str(i) + "_" + str(j) + ".txt"
            path_list.append((os.path.join(dir,filename), i))

    return path_list

def file2_mat_label(path_list):
    count = len(path_list)
    count_from0 = 0
    rnum = 0
    mat = np.zeros((count,1024))
    labels = np.zeros((1,count))

    for path, label in path_list:
        labels[0,count_from0] = int(label)
        #一个数据读入mat
        with open(path) as lines:
            rnum = 0
            for line in lines:
                if rnum == 32 : break
                line = line.strip()
                for j in range(32):
                    mat[count_from0,rnum*32+j] = int(line[j])
                rnum += 1
        count_from0 += 1
    #print(f"aaaaaaaaaaaaaaaaaaa{labels[0]}")
    return mat, labels
    # print(mat[0,0:32])
    # print(labels[0,180:300])

# dir = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits'
# alldata_count, count_class = count(os.listdir(dir))
# path_list = file_path_list_concat(dir,count_class)
# file2_mat_label(path_list)



def readdata(dir = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits'):
    # 第一步解析数据文件路径

    # 读取数据的第二种方式
    dir_root = dir
    dir_list = os.listdir(dir)

    alldata_count, label_num_dic = count(dir_list)

    # 从字典label_num_dic中，读取label以及序号，拼接成文件路径
    # pathlist：[(filepath1, class1),......(filepathn, classn)]
    path_list = file_path_list_concat(dir_root,label_num_dic)

    # data_mat.shape=(data_count, 1024)
    # label_list.shape(1,data_count)
    data_mat, label_list = file2_mat_label(path_list)
#    print(label_list.tolist())

    # print(training_mat.tolist())

    # label_list = []
    # # 读取这么多文件的egin_list
    # for path, label in path_list:
    #     linenum = 0
    #     with open(path) as inline:
    #         egin_vect = np.zeros((1024))
    #         for line in inline:
    #             if linenum == 32: break
    #             for i in range(32):
    #                 egin_vect[linenum * 32 + i] = int(line[i])
    #             linenum += 1
    #
    #     label_list.append(label)
    #     egin_vect = egin_vect.tolist()
    #     egin_list.append(egin_vect)

    #[(data_path, label)]
    # egin_list = np.array(egin_list)
    # label_list = np.array(label_list)

    return alldata_count, data_mat, label_list


#test test
train_dir = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits'
alltraindata_count, train_mat, label_list = readdata(train_dir)

test_dir = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\testDigits'
alltestdata_count, test_mat, test_label_list = readdata(test_dir)

# print(test_mat[0,:])
# print(f"dddddddddddddddddd{label_list}")

# test_index = 300
# testdata = test_mat[test_index,:]
# testlabel = test_label_list[0,test_index]

# sortedClassCount = KNN(testdata,train_mat,label_list,100)
# print(f"class_pred:{sortedClassCount}, real class: {int(testlabel)}")
error = 0
for i in range(alltestdata_count):
    test_index = i
    testdata = test_mat[test_index, :]
    testlabel = test_label_list[0, test_index]
    sortedClassCount = KNN(testdata, train_mat, label_list, 20)
    print(f"class_pred:{sortedClassCount}, real class: {int(testlabel)}")
    if int(sortedClassCount) != int(testlabel): error += 1

    print(f"error rate：{error/float(i+1)}")


# print(egin_list)
# print(egin_list)
# print(label_list)
#
# test_data = np.zeros((1,1024))
# print(test_data)
# test_data_path = "C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\testDigits\\8_1.txt"
#
# line_num = 0
# with open(test_data_path) as inline:
#     for line in inline:
#         if line_num == 32:break
#         line.strip()
#         for i in range(32):
#             test_data[0,line_num*32+i] = int(line[i])
#
# real_label = 8
# # print(label_list)
# sortedClassCount = KNN(test_data[0,:],egin_list,label_list,20)
# print(f"predicted:{sortedClassCount}，real:{real_label}")

# dir_root = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits'
# dir_list = os.listdir('C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits')
# print(dir_list)
# path_list =[]
# dic = {}
# for dir in dir_list:
#     tmp = dir.split('.')
#     classs, num = tmp[0].split('_')
#     dic[classs] = dic.get(classs, 0) + 1
#
# #print(dic)
#
# # 从字典中读取文件，并且写入list
# for i in range(10):
#     for j in range(dic[str(i)]):
#         # 开始拼接
#         file_name = str(i) + "_" + str(j) + ".txt"
#         data_path = os.path.join(dir_root, file_name)
#         #print(path_list)
#         # (data,path, label)
#         path_list.append((data_path, i))
#
# #print(path_list)
# label_list = []
# egin_vect = np.zeros((1024))
# linenum = 0
# for path, label in path_list:
#     with open('C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits\\9_198.txt') as inline:
#         for line in inline:
#             if linenum == 32:break
#             for i in range(32):
#                 egin_vect[linenum * 32 + i] = int(line[i])
#             linenum += 1
#     label_list.append(label)
#
# egin_vect = egin_vect.tolist()
# print(egin_vect)
# a = []
# label_list = np.array(label_list)
# print(label_list)
# a.append(egin_vect)
# a = np.array(a)
# print(a)
#
# b = [(1,2),(3,4)]
# print(b[1][1])