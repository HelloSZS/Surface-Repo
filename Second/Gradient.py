# w, b 为需要调整的参数
# 先初始化权重为 w = 3, b = 4
w = 3
b = 4

x = 2 * w + b
y = b + 1

#目标函数
z = x * y

#目标z值：60
target_z = 60
now_z = z

# z对b求偏导，根据链式求导法则
# 也就是z对x求偏导*x对b求偏导，再加上z对y求偏导*y对b求偏导
# z对x求偏导 = y，z对y求偏导 = x
# x对b求偏导 = 1，y对b求偏导 = 1
z_b = y*1 + x*1

def GradientDescent(step):
    global target_z, z_b, w,b, now_z

    # 调整参数
    delta_z = target_z - now_z
    # 因为 deltaz/deltab(即z对b求偏导) = z_b
    # 所以 deltab = deltaz/z_b
    # 知道deltaz和z_b的值，就可以求出deltab的值
    delta_b = delta_z/z_b

    b += delta_b

    #然后 参数b已经更新，可以重新计算值
    now_x = 2 * w +b
    now_y = b + 1
    now_z = now_x * now_y

    if(step%2 == 0):
        # print(f'step：{step}，现在b={"%.17f"%b}')
        print("step：",step,"，现在b=","%.17f"%b)
        # print(f'step：{step}，现在z={"%.17f"%now_z}')
        print("step：",step,"，现在z=","%.17f"%now_z)
        print("-----------------------------------------")

for i in range(14):
    GradientDescent(i+1)