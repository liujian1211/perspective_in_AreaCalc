import cv2
import numpy as np

vidcap = cv2.VideoCapture('./test_video.mp4')
success,image = vidcap.read()

while success:
    success,image = vidcap.read()
    frame = cv2.resize(image,(640,480))

    #选择4个点
    tl = (222,300) #左上角
    bl = (20,472) #左下角
    tr = (400,300) #右上角
    br = (600,472) #右下角

    #绘制4个点
    cv2.circle(frame, tl, 5,(0,0,255),-1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_frame = cv2.warpPerspective(frame,matrix,(640,480))

    cv2.imshow('Frame',frame)
    cv2.imshow('transformed_frame',transformed_frame)

    if cv2.waitKey(45)==27: #45表示延迟，控制视频播放速度
        break