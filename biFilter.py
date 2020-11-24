import cv2
def bi_demo(image):
    dst = cv2.bilateralFilter(image,0,100,15)  #第二个参数d是distinct，我们若是输入了d,会根据其去算第3或4个参数，我们最好是使用第3或4个参数反算d,先设为0
    cv2.imshow("bi_demo",dst)

src = cv2.imread("./234.jpg")  #读取图片
cv2.namedWindow("input image",cv2.WINDOW_AUTOSIZE)    #创建GUI窗口,形式为自适应
cv2.imshow("input image",src)    #通过名字将图像和窗口联系
bi_demo(src)
cv2.waitKey(0)   #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
cv2.destroyAllWindows()  #销毁所有窗口