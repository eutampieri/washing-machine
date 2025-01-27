import cv2 as cv
import os
from glob import glob
region = [(568, 276), (568 + 27, 276 + 87)]

def is_on(image_path):
    image = cv.imread(image_path)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_region = image_grey[region[0][1]:region[1][1], region[0][0]:region[1][0]]
    _, binarized_region = cv.threshold(image_region, 220, 255, cv.THRESH_BINARY)
    sum = 0
    count = 0
    shape = binarized_region.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            sum += binarized_region[i, j]
            count += 1
    avg = sum / count
    return avg > 0.85

for i in glob("raw/*.jpg"):
    print(i)
    path = i.split(os.sep)[1].split('.')[0]
    try:
        status = is_on(i)
    except:
        print(i)
        raise OSError
    if status:
        path = f"labeled{os.sep}on{os.sep}" + path + ".jpg"
    else:
        path = f"labeled{os.sep}off{os.sep}" + path + ".jpg"
    os.rename(i, path)
