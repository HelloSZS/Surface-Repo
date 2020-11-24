# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

    u = np.array([3,5,2])
    v = np.array([1,2,3])

    print(np.cross(u,v))

    u = np.array([[3,5,2]])
    #v = np.array([[1,2,3]]).T    #error
    v = np.array([[1, 2, 3]])     #ok
    print(np.cross(u,v))

    u = np.array([3,5,2])
    v = np.array([1,2,3])

    print(np.dot(u,v))

    u = np.array([[3,5,2]])
    v = np.array([[1,2,3]]).T

    print(np.dot(u,v))

    #列向量线性相加
    u=np.array([[3,4,5]]).T
    v=np.array([[4,5,6]]).T
    w=np.array([[5,6,7]]).T

    print(3*u+4*v+5*w)

    m = np.array([[1.0,2,3],[4,5,6]])

    print(m)
    print(m.T)
    print(m.T.shape)

    S = np.array([[1, 2, 3],
                  [2, 5, 6],
                  [3, 6, 9]])

    print(S)
    print(S.T)

    m = np.array([[1,2,3],
                  [4,5,6]])
    print(m)
    print(m.T)

    a = np.zeros([4,5])
    print(a)

    b = np.diag([1,2,3])
    print(b)

    c = np.eye(5)
    print(c)

    d = np.array([[1],[2],[3]])
    #or
    e = np.array([[1,2,3]]).T
    print(e)
    print(d)
    print(d.T)

    A_1 = np.array([[1,1,0],[1,0,1]])
    A_2 = np.array([[1,2,-1],[2,4,-2]])
    A_3 = np.array([[1,0],[0,1],[0,-1]])
    A_4 = np.array([[1,2],[1,2],[-1,-2]])
    A_5 = np.array([[1,1,1],[1,1,2],[1,2,3]])

    print(np.linalg.matrix_rank(A_1))
    print(np.linalg.matrix_rank(A_2))
    print(np.linalg.matrix_rank(A_3))
    print(np.linalg.matrix_rank(A_4))
    print(np.linalg.matrix_rank(A_5))

    from scipy import linalg

    C = np.eye(3)
    print(C)
    C_n = linalg.inv(C)
    print(C_n)

    B = np.array([[0, 2, 3], [0, 2, 3], [0, 0, 4]])
    B_n = None
    try:
        B_n = linalg.inv(B)
        print(B_n)
    except:
        print("不可逆")


    # eng, mat, phy = np.loadtxt('c:\\Users\\hello\\Desktop\\MLofM\\a.csv',delimiter=',',usecols=(0,1,2),unpack=True)
    # print(eng)
    # print(eng.mean(),mat.mean(),phy.mean())
    # print(np.cov(eng),np.cov(mat),np.cov(phy))
    #
    # S = np.vstack((eng,mat,phy))
    # print(np.cov(S))


    # A_1 = np.array([[1,1,0],[1,0,1]])
    # A_2 = np.array([[1,2,-1],[2,4,-2]])
    # A_3 = np.array([[1,0],[0,1],[0,-1]])
    # A_4 = np.array([[1,2],[1,2],[-1,-2]])
    A_6 = np.array([[1,1,5],[3,1,2],[1,2,3]])


    A = np.array([[1,2,3],[1,-1,4],[1,2,-1]])

    y = np.array([14,11,5])

    x = linalg.solve(A,y)
    print(x)

    A = np.array([[2,1],[1,2]])

    eval, evect = linalg.eig(A)

    print("eval:")
    print(eval)
    print("evect:")
    print(evect)

    B = np.array([[1,0,0],[0,2,0],[0,0,5]])
    eval, evect = linalg.eig(B)
    print(eval)
    print(evect)

    A_6T = A_6.T
    print(A_6)
    print(A_6T)
    A_62 = np.dot(A_6,A_6T)
    print(A_62)
    eigval, eigvect = linalg.eig(A_62)
    print(eigval)

    A_4T = A_4.T
    A_42 = np.dot(A_4,A_4T)
    print('A42：\n',A_42)

    def ifinv(A):
        try:
            A_42_inv = linalg.inv(A_42)
            print("表示",A_42_inv)
        except:
            print("不可逆")



    A42_eigval, A42_eigvec = linalg.eig(A_42)
    print(A42_eigvec)

    # try:
    #     B_n = linalg.inv(B)
    #     print(B_n)
    # except:
    #     print("Error，不可逆")


    import matplotlib.pyplot as plt

    x = [2,2,4,8,4]
    y = [2,6,6,8,8]

    S = np.vstack((x,y))
    #np.cov(Matrix)
    #相当于计算 E((X-u)(Y-v))
    #
    #
    #
    print(np.cov(S))


    import numpy as np
    import matplotlib.pyplot as plt


#验证零均值化前后的协方差矩阵是否发生变化
    x = [2,2,4,8,4]
    y = [2,6,6,8,8]

    S = np.vstack((x,y))
    print("尚未均值化")
    print(S)
    print(np.cov(S))

##############################################
    x = x - np.mean(x)
    y = y - np.mean(y)

    S = np.vstack((x,y))
    print("已经均值化")
    print(S)
    print(np.cov(S))