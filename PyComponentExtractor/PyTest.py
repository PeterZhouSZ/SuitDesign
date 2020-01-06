ModuleDir = r'C:\Code\Projects\OpenCV\PyComponentExtractor\Release'

import sys
sys.path.insert(0, ModuleDir)

import ComponentExtractor

import numpy as np

img = np.zeros((100, 100), np.uint8)
img[10, 10] = 255
img[50, 50] = 255
img[51, 50] = 255
img[50, 51] = 255
output = ComponentExtractor.componentsExtractor (img, 255)
print(output)