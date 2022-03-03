import os
import cv2



one = cv2.imread("waterfall_res.bmp")
cv2.imwrite("waterfall_input.bmp", cv2.cvtColor(one, cv2.COLOR_BGR2GRAY))