import cv2
import numpy as np
import os

#img = cv2.imread('../test2/Te-no_0010_mask.png')
#img[img != 255] = 0 # change everything to white where pixel is not black
#cv2.imwrite('../test2/Te-no_0010_mask.png', img)

path="../results/"
dirs = os.listdir( path )
dirs.sort()

for item in dirs:
    if os.path.isfile(path+item):
        if not "input" in item:
            img = cv2.imread(path+item)
            img[img != 255] = 0 # change everything to white where pixel is not black
            cv2.imwrite(path+item, img)
            print(path+item)
            #os.rename(path+item,path+os.path.splitext(item)[0]+"_input.png")
            #np.save(path+os.path.splitext(item)[0]+"_label.npy",im)