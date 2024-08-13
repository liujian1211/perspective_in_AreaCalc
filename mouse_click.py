import cv2

# 鼠标点击事件的回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在图像上打印坐标点
        print(f"Clicked at ({x}, {y})")

# 读取图像
image = cv2.imread("D:/test_perspective/crop_imgs/larger_test.jpg")

# 创建窗口并绑定鼠标事件
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", mouse_callback)

while True:
    # 在窗口上显示图像
    # cv2.resizeWindow('image',image.shape[1],image.shape[0])
    cv2.imshow("image", image)

    # 等待按键事件，按下ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cv2.destroyAllWindows()