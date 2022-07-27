import numpy as np
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import glob
import cv2
import argparse
from tf_models import *

def get_parser():
    parser = argparse.ArgumentParser(
        description='Folder Cleanup for redundant images and mask without any objects')
    
    parser.add_argument('--data-path',
                        default="/home/fjannat/Documents/EarthVision/data/phase_2/dataset_2/",
                        help='path to large image')
    parser.add_argument('--save-results',
                    default="/home/fjannat/Documents/EarthVision/data/phase_2/dataset_2/",
                    help='path to mage')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()  

    #set_name=arg.set_name
    
    set_name=["train", "val", "test"]
    for i in set_name:
        print("Removing items from " +str(i))
        print("-------------------------------")
        file_dir_list = sorted(glob.glob(arg.data_path+"/"+i +"/masks/*"))
        print("Length of mask directory: "+str(len(file_dir_list)))

        filtered_mask_dir_list=[]
        for i in file_dir_list:
            mask = cv2.imread(i)
            a= np.unique(mask)
            if len(a)<=1:
                filtered_mask_dir_list.append(i)    
        print("Length of filtered mask directory: "+str(len(filtered_mask_dir_list)))
        print("")

        for i in filtered_mask_dir_list:
            if os.path.exists(i):
                os.remove(i)

        print("Removed redundant masks")     



        filtered_list_copy=filtered_mask_dir_list
        filtered_img_dir_list=[]
        for i in filtered_list_copy:
            i = i.replace("mask","image")
            filtered_img_dir_list.append(i)
        print("Length of filtered image directory: "+str(len(filtered_img_dir_list)))
        print("")



        for i in filtered_img_dir_list:
            if os.path.exists(i):
                os.remove(i)

        print("Removed redundant images") 
    
    
    image_dataset = []
    for filename in sorted(glob.glob(arg.data_path+'/test/images/image*.png')): #assuming png
        
        im=Image.open(filename)
        a = np.asarray(im)
        #a = np.expand_dims(a, axis=2)
        image_dataset.append(a)

    print("New Length of Images after removing redundant images: "+str(len(image_dataset)))

    mask_dataset = []
    for filename in sorted(glob.glob(arg.data_path+'/test/masks/mask*.png')): #assuming png
        im=Image.open(filename)
        a = np.asarray(im)
        mask_dataset.append(a[:,:])
    
    print("New Length of Masks after removing redundant Masks: "+str(len(mask_dataset)))
    
    if not os.path.exists(arg.save_results):
        os.makedirs(arg.save_results)   
    mask_dataset= np.expand_dims(mask_dataset, axis=3)
    visualize_image_mask_sample(image_dataset, mask_dataset, arg.save_results, name="visualize_image_mask_sample_after_data_cleanup" ,n=5)
 
  
    