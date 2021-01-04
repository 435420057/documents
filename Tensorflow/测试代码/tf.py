# -*- coding: UTF-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./data/timg.jpg', 0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show() 