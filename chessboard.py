import cv2
import numpy as np
import math

#海龙公式求4个点围成的区域面积
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def area_of_quadrilateral(p1, p2, p3, p4):
    a = distance(p1, p2)
    b = distance(p2, p3)
    c = distance(p3, p4)
    d = distance(p4, p1)
    s = (a + b + c + d) / 2
    return round(math.sqrt((s-a)*(s-b)*(s-c)*(s-d)),2)

def transformed_cor(M,p1,p2,p3,p4):
    p1 = p1+(1,)
    p2 = p2+(1,)
    p3 = p3+(1,)
    p4 = p4+(1,)

    ret1 = np.dot(M,p1)
    ret2 = np.dot(M,p2)
    ret3 = np.dot(M,p3)
    ret4 = np.dot(M,p4)

    p1_x = round(ret1[0] / ret1[2],2)
    p1_y = round(ret1[1] / ret1[2],2)

    p2_x = round(ret2[0] / ret2[2],2)
    p2_y = round(ret2[1] / ret2[2],2)

    p3_x = round(ret3[0] / ret3[2],2)
    p3_y = round(ret3[1] / ret3[2],2)

    p4_x = round(ret4[0] / ret4[2],2)
    p4_y = round(ret4[1] / ret4[2],2)

    point1=()
    point2=()
    point3=()
    point4=()

    point1 = point1 + (p1_x,p1_y)
    point2 = point2 + (p2_x,p2_y)
    point3 = point3 + (p3_x,p3_y)
    point4 = point4 + (p4_x,p4_y)
    return point1,point2,point3,point4,

# 读取棋盘格图像
chessboard = cv2.imread('D:/test_perspective/flv_video_13515277111_34.png')

# print(f'原图尺寸为：{chessboard.shape}')

# 定义棋盘格的行数和列数
rows = 10
cols = 14

# 棋盘格的交叉点数目
num_corners = (cols - 1) * (rows - 1)

# 设置棋盘格方块的大小
square_size = 20 #可以控制缩放比例

# 计算棋盘格的交叉点坐标
obj_points = np.zeros((num_corners, 3), np.float32)
obj_points[:, :2] = np.mgrid[0:cols - 1, 0:rows - 1].T.reshape(-1, 2) * square_size

# 转换为灰度图像
gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)

# 寻找棋盘格的角点
ret, corners = cv2.findChessboardCorners(gray, (cols - 1, rows - 1), None)

# 如果找到角点
if ret:
    # 亚像素级角点检测
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


    #打印4个角点，顺序为：右上、右下、左上、左下
    # print(corners2[0][0][0],corners2[0][0][1])
    # print(corners2[cols-2][0][0],corners2[cols-2][0][1])
    # print(corners2[corners2.shape[0]-cols][0][0], corners2[corners2.shape[0]-cols][0][1])
    # print(corners2[corners2.shape[0]-1][0][0], corners2[corners2.shape[0]-1][0][1])

    #4个角点的顺序为：右上、右下、左上、左下
    cv2.circle(chessboard, (int(corners2[0][0][0]), int(corners2[0][0][1])), 10, (0, 0, 255), -1) #原图右上角 #原图左下角
    cv2.circle(chessboard, (int(corners2[cols-2][0][0]), int(corners2[cols-2][0][1])), 10, (0, 0, 255), -1)     #原图右下角 #原图左上角
    cv2.circle(chessboard, (int(corners2[corners2.shape[0]-cols+1][0][0]),int(corners2[corners2.shape[0]-cols+1][0][1])), 10, (0, 0, 255), -1) #原图左上角 #原图右下角
    cv2.circle(chessboard, (int(corners2[corners2.shape[0]-1][0][0]), int(corners2[corners2.shape[0]-1][0][1])), 10, (0, 0, 255), -1)  #原图左下角 #原图右上角

    # chessboard1.jpg的外围矩形
    obj_p1 = (1792, 2012)
    obj_p2 = (2153, 1998)
    obj_p3 = (2162, 2229)
    obj_p4 = (1805, 2238)

    #绘制目标的外接矩形
    cv2.circle(chessboard,obj_p1,10,(0,255,0),-1)
    cv2.circle(chessboard, obj_p2, 10, (0, 255, 0), -1)
    cv2.circle(chessboard, obj_p3, 10, (0, 255, 0), -1)
    cv2.circle(chessboard, obj_p4, 10, (0, 255, 0), -1)

    cv2.line(chessboard,obj_p1,obj_p2,(255,0,0),2)
    cv2.line(chessboard, obj_p2, obj_p3, (255, 0, 0), 2)
    cv2.line(chessboard, obj_p3, obj_p4, (255, 0, 0), 2)
    cv2.line(chessboard, obj_p4, obj_p1, (255, 0, 0), 2)

    area_chessboard = area_of_quadrilateral( obj_p1, obj_p2,obj_p3,obj_p4)
    # print(f'原图识别目标的像素面积为{area_chessboard}') #255174.49，手工计算面积为262135，因此计算正确，实际应为 255174.49
    print(f'原图识别目标的4个顶点的坐标为{obj_p1}、{obj_p2}、{obj_p3}、{obj_p4}')

    #目标外围矩形顶点
    # obj_p1=(827, 955)
    # obj_p2=(1164, 932)
    # obj_p3=(1172, 1241)
    # obj_p4=(833, 1250)
    #
    # area_obj = area_of_quadrilateral(obj_p1,obj_p2,obj_p3,obj_p4)
    # print(f'原图目标的像素面积为{area_obj}')

    # 绘制角点
    cv2.drawChessboardCorners(chessboard, (cols - 1, rows - 1), corners2, ret)

    # 定义棋盘格的目标点和图像点
    src_points = np.float32([corners2[0],corners2[cols-2],corners2[corners2.shape[0]-cols+1],corners2[corners2.shape[0]-1]])

    # dst_points = np.float32(
    #     [[0, square_size * (rows - 2)],[0, 0],  [square_size * (cols - 2), square_size * (rows - 2)],[square_size * (cols - 2), 0]
    #      ])

    #左下、左上、右下、右上
    # dst_points = np.float32([
    #                         [140,0],
    #                         [140,140],
    #                         [0,0],
    #                         [0,140]
    #                         ])

    dst_points = np.float32([
        [0, 0],
        [140, 0],
        [140, 140],
        [0, 140]
    ])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # M为：[[-1.31138588e+00  -4.22548695e-01  5.14157276e+02]
    #       [-6.26406913e-03  -2.22809369e+00  1.52850518e+03]
    #       [1.39037842e-04   -4.88144353e-03  1.00000000e+00]]

    # print(f'M为：{M}')

    p1_res,p2_res,p3_res,p4_res = transformed_cor(M,obj_p1,obj_p2,obj_p3,obj_p4)
    print(f'转换后p1坐标为:{p1_res},p2坐标为：{p2_res},p3坐标为：{p3_res},p4坐标为：{p4_res}')

    area_obj_trans = area_of_quadrilateral(p1_res,p2_res,p3_res,p4_res)
    print(f'转换后的目标的像素面积为{area_obj_trans}')

    # 进行透视变换
    warped_image = cv2.warpPerspective(chessboard, M, (square_size * (cols - 2), square_size * (rows - 2)))
    # warped_image = cv2.warpPerspective(chessboard, M, (chessboard.shape[1], chessboard.shape[0]))
    print(f'warped_image的尺寸为：{warped_image.shape}')

    cv2.circle(warped_image,(int(p1_res[0]),int(p1_res[1])),5,(0,0,255),-1)
    cv2.circle(warped_image, (int(p2_res[0]),int(p2_res[1])), 5, (0, 0, 255), -1)
    cv2.circle(warped_image, (int(p3_res[0]),int(p3_res[1])), 5, (0, 0, 255), -1)
    cv2.circle(warped_image, (int(p4_res[0]),int(p4_res[1])), 5, (0, 0, 255), -1)

    # 显示原始图像和鸟瞰图
    cv2.namedWindow('Original img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original img',chessboard.shape[1],chessboard.shape[0])
    cv2.imshow('Original img', chessboard)

    cv2.namedWindow('Warped img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Warped img',warped_image.shape[1],warped_image.shape[0])
    cv2.imshow('Warped img',warped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('No chessboard pattern found!')