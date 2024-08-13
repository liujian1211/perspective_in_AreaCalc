import cv2
import numpy as np

# 读取棋盘格图像
chessboard = cv2.imread('D:/test_pytorch/chessboard_calc.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)

M= np.array([[ -1.02667479e+00 ,-7.12372457e-01 ,1.06639518e+03],
    [ 3.96903353e-02, -1.90241099e+00,  1.97882057e+03],
    [3.56884200e-05 ,-2.10318713e-03  ,1.00000000e+00]])

square_size=100

rows = 10
cols = 7

# 进行透视变换
# warped_image = cv2.warpPerspective(chessboard, M, (square_size * (cols - 2), square_size * (rows - 2)))
warped_image = cv2.warpPerspective(chessboard, M, (chessboard.shape[1],chessboard.shape[0]))

# 显示原始图像和鸟瞰图
cv2.namedWindow('Original img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original img',chessboard.shape[1],chessboard.shape[0])
cv2.imshow('Original img', chessboard)
# cv2.imshow('Original Chessboard', chessboard)
cv2.namedWindow('Warped img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Warped img',warped_image.shape[1],warped_image.shape[0])
cv2.imshow('Warped img',warped_image)
# cv2.imshow('Warped Chessboard', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
