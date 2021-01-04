# -*- coding: utf-8 -*-
"""
腐蚀
cv2.erode(src,                     # 输入图像
	  kernel,                  # 卷积核
	  dst=None, 
	  anchor=None,
	  iterations=None,         # 迭代次数，默认1
	  borderType=None,
	  borderValue=None) 

膨胀
cv2.dilate(src,                    # 输入图像
           kernel,                 # 卷积核
           dst=None, 
           anchor=None, 
           iterations=None,        # 迭代次数，默认1
           borderType=None, 
           borderValue=None)
"""
import cv2
import numpy as np
original_img = cv2.imread("./numberdata/data/test_output.jpg", 1)
res = cv2.resize(original_img,None,fx=0.6, fy=0.6,
                 interpolation = cv2.INTER_CUBIC) #图形太大了缩小一点
B, G, R = cv2.split(res)                    #获取红色通道
img = R
_,RedThresh = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
#OpenCV定义的结构矩形元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
eroded = cv2.erode(RedThresh,kernel)        #腐蚀图像
dilated = cv2.dilate(RedThresh,kernel)      #膨胀图像

cv2.imshow("original_img", res)             #原图像
cv2.imshow("R_channel_img", img)            #红色通道图
cv2.imshow("RedThresh", RedThresh)          #红色阈值图像
cv2.imshow("Eroded Image",eroded)           #显示腐蚀后的图像
cv2.imshow("Dilated Image",dilated)         #显示膨胀后的图像

#NumPy定义的结构元素
NpKernel = np.uint8(np.ones((3,3)))
Nperoded = cv2.erode(RedThresh,NpKernel)       #腐蚀图像
cv2.imshow("Eroded by NumPy kernel",Nperoded)  #显示腐蚀后的图像
cv2.waitKey(0)
cv2.destroyAllWindows()