import os
import random
from scipy import ndarray
import os.path
import cv2
import time
import random
import base64
import configparser
# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from os import listdir
from tqdm import tqdm 
import numpy as np
from matplotlib import pyplot as plt

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img
def horizontal_flip(img):
  
    return cv2.flip(img, 1)
    
def vertical_flip(img):

    return cv2.flip(img, 0)


def img_transpose(img):
    image = cv2.transpose(img)
    return image
    
def rotation(img, angle):
#    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]
def img_blur(img):
    blur_img = cv2.blur(img,(5,5))
    return blur_img

def img_resize(img_path,dest_path):
    try:
        img_read =cv2.imread(img_path)
        bigger = cv2.resize(img_read, (512, 512))

        image_name = dest_path + "image/"+ img_name
        cv2.imwrite(image_name,bigger)
        time.sleep(.1)
        
        label_read =cv2.imread(label_path)
        bigger = cv2.resize(label_read, (512, 512))

        labl_name = dest_path + "label/"+ label_name
        cv2.imwrite(labl_name,bigger)
        time.sleep(.1)
        
    except Exception as e:
        print("error : ",e)
        img_error.append(img_name)

def add_noise(img):
    # Getting the dimensions of the image
    print("0")
    row , col = img.shape
    print("1")
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_coord][x_coord] = 0
          
    return img

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# # dictionary of the transformations we defined earlier
# available_transformations = ["brightness","channel_shift","horizontal_flip","rotation","vertical_flip","img_blur"]

# img_folder_path = '/home/ajithbalakrishnan/vijnalabs/Assignments/Sensor_Physics/dataset/Annotated_Dataset_V2/image/'
# annotation_folder_path = "/home/ajithbalakrishnan/vijnalabs/Assignments/Sensor_Physics/dataset/Annotated_Dataset_V2/label/"  #derived_aug_images
# dest_path =  "/home/ajithbalakrishnan/vijnalabs/Assignments/Sensor_Physics/dataset/Annotated_Dataset_V2/final/"
# augmentation_type = "channel_shift"

# img_list = [file for file in listdir(img_folder_path) if file.endswith('.jpg')]
# img_list = img_list+ [file for file in listdir(img_folder_path) if file.endswith('.png')]+[file for file in listdir(img_folder_path) if file.endswith('.jpeg')]

# label_list = [file for file in listdir(annotation_folder_path) if file.endswith('.png')]


# print("len(img_source)",len(img_list))
# print("len(xml_source)",len(label_list))

# xml_error =  []
# img_error = []


# count = 1392

#######horizontal_flip    # Derived dataset
#######vertical_flip      # Derived dataset
#channel_shift(img_read,60)
#img_blur
#add noise 
########img_transpose         # Derived dataset
#grayscale
########rotation 60           # Derived dataset
########rotation 120          # Derived dataset
########rotation 220          # Derived dataset
#brightness

#img_blur
#gray scale
#channel_shift(img_read,60)
#brightness

# for i in tqdm (range (len(img_list)),desc="Loading..."):
   
#         img_name = img_list[i]
#         img = img_name.split('.')
#         label_name = str(img[0]) + ".png"
#         img_path = img_folder_path + img_name
#         label_path = annotation_folder_path + label_name

#         try:
#             print("img_path : ",img_path)
#             img_read =cv2.imread(img_path)
#             count = count +1

#         #     #aug_img = horizontal_flip(img_read)
#         #     aug_img = vertical_flip(img_read,True)
#         #     #aug_img = rotation(img, 30)
#             # angle = 220
#             # angle = int(random.uniform(-angle, angle))

#             #aug_img =add_noise(img_read)
            
#             img_dest = dest_path + "image/" + str(count)+"."+str(img[1])
#             cv2.imwrite(img_dest,img_read) 

#         #     label = cv2.imread(label_path)
#         #    img = vertical_flip(label,True)
#             label_read =cv2.imread(label_path)
#         #    aug_label = img_transpose(label_read)
#             label_dest = dest_path + "label/" + str(count)+".png"
#             cv2.imwrite(label_dest, label_read)
#         except Exception as e:
#             print("img read/write error : ",e)
#             img_error.append(img_name)

# print("XML Error list ", xml_error)
# print("img error list ", img_error)


# exit()

# for k in tqdm (range (len(available_transformations)),desc="Loading..."):

#     operation = available_transformations[k]
#     print("available_transformations :", operation)
    
    
#     for i in tqdm (range (len(img_list)),desc="Loading..."):
   
#         img_name = img_list[i]
#         img = img_name.split('.')
#         label_name = str(img[0]) + ".png"
#         img_path = img_folder_path + img_name
#         label_path = annotation_folder_path + label_name

        

#         try:

#             img_read =cv2.imread(img_path)
#             print("img_path ",img_path)
#             count = count +1
#             # if operation == "brightness":
#             #     bright_img = brightness(img_read, 0.8, 1.2)
#             #     img_dest = dest_path  +"image/" + str(count)+"."+str(img[1])
#             #     print("img_dest : ",img_dest)
#             #     cv2.imwrite(img_dest,bright_img)

#             #     label = cv2.imread(label_path)
#             #     label_dest = dest_path + "label/" + str(count)+".png"
#             #     cv2.imwrite(label_dest, label)

#             # if operation == "img_blur":
#             #     blur_img = img_blur(img_read)
#             #     img_dest = dest_path  +"image/" +str(count)+"."+str(img[1])
#             #     print("img_dest : ",img_dest)
#             #     cv2.imwrite(img_dest,blur_img)

#             #     label = cv2.imread(label_path)
#             #     label_dest = dest_path + "label/" + str(count)+".png"
#             #     cv2.imwrite(label_dest, label)

#             if operation == "channel_shift":
#                 channel_shift = channel_shift(img_read, 50)
#                 img_dest = dest_path  +"image/" +str(count)+"."+str(img[1])
#                 print("img_dest : ",img_dest)
#                 cv2.imwrite(img_dest,channel_shift)
#                 time.sleep(.1)

#                 label = cv2.imread(label_path)
#                 label_dest = dest_path + "label/" + str(count)+".png"
#                 cv2.imwrite(label_dest, label)
#                 time.sleep(.1)
#         #     #channel_shift_img = channel_shift(img_read, 80)
#         #     #img = horizontal_flip(img_read)
#         #     img = vertical_flip(img_read,True)
#         #     #img = rotation(img, 30)

#         #     img_dest = dest_path + "/" + augmentation_type + img_name
#         #     cv2.imwrite(img_dest,img) 
#         #     label = cv2.imread(label_path)

#         #     img = vertical_flip(label,True)
#         #     label_dest = dest_path + "/" + augmentation_type+ "label" + img_name
#         #     cv2.imwrite(label_dest, img)

            

#         except Exception as e:
#             print("img read/write error : ",e)
#             img_error.append(img_name)

# print("XML Error list ", xml_error)
# print("img error list ", img_error)
