import numpy as np
import os
# from img2mat import img2mat
# from KNN import KNN
import paddle.fluid as fluid

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

def file_path_list_concat(dir,dict,shuffle = True):
    path_list = []
    for i in range(10):
        classi_filenumber = dict.get(str(i))
        for j in range(classi_filenumber):
            filename = str(i) + "_" + str(j) + ".txt"
            path_list.append((os.path.join(dir,filename), i))
    import random
    if shuffle == True:
        random.shuffle(path_list)
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



def readdata(dir = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits',shuffle = True):
    # 第一步解析数据文件路径
    # 读取数据的第二种方式
    dir_root = dir
    dir_list = os.listdir(dir)

    alldata_count, label_num_dic = count(dir_list)

    # 从字典label_num_dic中，读取label以及序号，拼接成文件路径
    # pathlist：[(filepath1, class1),......(filepathn, classn)]
    path_list = file_path_list_concat(dir_root,label_num_dic,shuffle = True)

    data_mat, label_list = file2_mat_label(path_list)

    return alldata_count, data_mat, label_list

#以上是读取数据的部分

alldata_count, data_mat, label_list = readdata()
#print(data_mat.shape)
#shape = (1934, 1024) = (alldata_count, 1024)
#1664为128的整数倍，1个epoch可以运行13个iter

def to_onehot(label_list):
    num_class = 10

    a = list(label_list)[0]

    list_int = []
    for i in a:
        list_int.append(int(i))
    #print(list_int)
    #array = [i for i in range(10)]
    one_hot_array = np.eye(10)[list_int]

    #print(one_hot_array)
    return one_hot_array

onehot_label_list = np.array(to_onehot(label_list))

#数据集划分，1664个训练数据，270个测试数据
train_mat = data_mat[:1664,:]
test_mat = data_mat[1664:,:]

train_label_list = label_list[0,:1664]
test_label_list = label_list[0,1664:]

def data_loader(data_list, label_list,buffer_size = 128,label_size=270):
    def reader():
        step_label = 0
        while True:
            if step_label >= label_size:
                break

            for i in range(step_label,step_label+buffer_size):
                yield data_list[i,:], int(label_list[i])

            # print(data_list[i].shape, label_list[i].shape)
            step_label += buffer_size
    return reader


reader = data_loader(train_mat,train_label_list)
train_reader = fluid.io.batch(reader,batch_size=32)

reader2 = data_loader(test_mat,test_label_list,buffer_size=10,label_size=270)
test_reader = fluid.io.batch(reader2,batch_size=10)
# a = next(train_reader())
# print(a)

#以上是数据处理部分
def mul_perception(input):
    layer1 = fluid.layers.fc(input,size=100,act="relu")
    layer2 = fluid.layers.fc(layer1,size=100,act="relu")
    output = fluid.layers.fc(layer2,size=10,act="softmax")
    return output

data = fluid.layers.data(name='data', shape=[1,1024],dtype='float32')
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

model = mul_perception(data)
cost = fluid.layers.cross_entropy(input=model,label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model,label=label)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder(place=place,feed_list=[data,label])

# 开始训练和测试
for pass_id in range(10):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader

        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 给模型喂入数据
                                        fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        # 每100个batch打印一次信息  误差、准确率
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

    # 保存模型
    model_save_dir = "./models/"
    # 如果保存路径不存在就创建
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('save models to %s' % (model_save_dir))
    fluid.io.save_params(executor=exe,dirname=model_save_dir)
    # fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
    #                               ['image'],  # 推理（inference）需要 feed 的数据
    #                               [model],  # 保存推理（inference）结果的 Variables
    #                               exe)  # executor 保存 inference model


# #test test
# train_dir = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits'
# alltraindata_count, train_mat, label_list = readdata(train_dir)
#
# test_dir = 'C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\testDigits'
# alltestdata_count, test_mat, test_label_list = readdata(test_dir)
#
# # print(test_mat[0,:])
# # print(f"dddddddddddddddddd{label_list}")
#
# # test_index = 300
# # testdata = test_mat[test_index,:]
# # testlabel = test_label_list[0,test_index]
#
# # sortedClassCount = KNN(testdata,train_mat,label_list,100)
# # print(f"class_pred:{sortedClassCount}, real class: {int(testlabel)}")
# error = 0
# for i in range(alltestdata_count):
#     test_index = i
#     testdata = test_mat[test_index, :]
#     testlabel = test_label_list[0, test_index]
#     sortedClassCount = KNN(testdata, train_mat, label_list, 20)
#     print(f"class_pred:{sortedClassCount}, real class: {int(testlabel)}")
#     if int(sortedClassCount) != int(testlabel): error += 1

    # print(f"error rate：{error/float(i+1)}")
