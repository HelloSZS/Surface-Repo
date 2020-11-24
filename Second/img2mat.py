import numpy as np

def img2mat(file_name):
    return_vect = np.zeros((1,1024))

    count = 0
    with open(file_name) as inline:
        for line in inline:
            for i in range(32):
                return_vect[0,32*count+i] = int(line[i])
            count += 1

    return return_vect

#test
print(img2mat("C:\\Users\\hello\\Desktop\\MLBook\\Ch02\\digits\\trainingDigits\\0_0.txt"))