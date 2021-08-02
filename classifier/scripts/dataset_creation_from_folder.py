import xml.etree.cElementTree as ET
import os
import os.path
import cv2
import random
import base64
import configparser
import os
from os import listdir
from tqdm import tqdm 
from data_augmentation import *

basepath = "/media/ajithbalakrishnan/external/Dataset/freelancer/Dataset with EXIF GPS-20210717T181130Z-001/Dataset with EXIF GPS"

img_list = [file for file in listdir(basepath) if file.endswith('.jpg')]

print("len(img_source)",len(img_list))

img_error = []
k= open(str(basepath + "/total.txt"),"a")
for i in tqdm (range (len(img_list)),desc="Loading..."):
    img_name = img_list[i]
    img = img_name.split('.')

    path = os.path.join(basepath, img[0])
    os.mkdir(path)

    try:
        img_path = os.path.join(basepath,img_name)
        img_read =cv2.imread(img_path)
        
        img_dest = path + "/" + str(img_name)
        print("img_dest : ",img_dest)
        cv2.imwrite(img_dest,img_read)

        flip_img = vertical_flip(img_read)
        img_dest = path + "/v_flip_" + str(img_name)
        print("img_dest : ",img_dest)
        cv2.imwrite(img_dest,flip_img)

        flip_img = horizontal_flip(img_read)
        img_dest = path + "/h_flip_" + str(img_name)
        print("img_dest : ",img_dest)
        cv2.imwrite(img_dest,flip_img)

        transpose_img = img_transpose(img_read)
        img_dest = path + "/transpose_" + str(img_name)
        print("img_dest : ",img_dest)
        cv2.imwrite(img_dest,transpose_img)

        # rotated_img = rotate_image(img_read,angle=90)
        # img_dest = path + "/rotate_" + str(img_name)
        # print("img_dest : ",img_dest)
        # cv2.imwrite(img_dest,rotated_img)
        

    except:
        print("img read/write error")
        img_error.append(img_name)

    k.writelines(str(img[0]))
    k.write("\n")
k.close()

print("img error list ", img_error)




