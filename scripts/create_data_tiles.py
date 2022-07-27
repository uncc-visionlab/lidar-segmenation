#create patches from image and mask
import numpy as np
from patchify import patchify, unpatchify
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import glob
import albumentations as A
import random
from tf_models import *

def get_parser():
    parser = argparse.ArgumentParser(
        description='Creating images and masks folder by augmenting a large image and mask')
    parser.add_argument('--tile-size',
                        type=int,
                        default=128,) 
    parser.add_argument('--split-num',
                        type=int,
                        default=2,) 
    
    parser.add_argument('--tile-number',
                        type=int,
                        default=35000,) 
    parser.add_argument('--img-dir',
                        #default="/home/fjannat/Documents/EarthVision/test_data_1/testimg.png",
                        default="/home/fjannat/Documents/EarthVision/data_resource/kom_ucb_mls/singleband_hillshade/",
                        help='path to large image')
    parser.add_argument('--new-dir',
                        default="/home/fjannat/Documents/EarthVision/data/hill_single/dataset_2/",
                        help='path to the directory where needs to create the new folders')

    parser.add_argument('--save-results',
                    default="/home/fjannat/Documents/EarthVision/data/hill_single/dataset_2/",
                    help='path to mage')
    return parser


def get_datalist(data_path):   
    image_dataset = []
    mask_dataset = []
    hill_dataset = []

    for filename in sorted(glob.glob(data_path+'/img*.png')): #assuming png
        im = cv2.imread(filename)
        image_dataset.append(im)

    for filename in sorted(glob.glob(data_path+'/mask*.png')): #assuming png
        im = cv2.imread(filename)
        mask_dataset.append(im[:,:])
        
    for filename in sorted(glob.glob(data_path+'/hillshade*.png')): #assuming png
        im = cv2.imread(filename)
        hill_dataset.append(im[:,:])
    return image_dataset, mask_dataset, hill_dataset
 
def get_datalist_from_dir(data_path):   
    image_dataset = []
    mask_dataset = []
    hill_dataset = []
    for filename in sorted(glob.glob(data_path+'/images/image*.png')): #assuming png
        im = cv2.imread(filename)
        image_dataset.append(im)

    for filename in sorted(glob.glob(data_path+'/masks/mask*.png')): #assuming png
        im = cv2.imread(filename)
        mask_dataset.append(im[:,:])
        
    for filename in sorted(glob.glob(data_path+'/hills/hill*.png')): #assuming png
        im = cv2.imread(filename)
        hill_dataset.append(im[:,:])
    return image_dataset, mask_dataset, hill_dataset
 
def visualize_image_mask_hill_sample(image_dataset, mask_dataset, hill_dataset, save_dir, name,  n=5):
    rows=n
    cols=3
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(4,n*4) )
    c=0
    j=1
    for i in range(n):
        image_number = random.randint(0, (len(image_dataset)-1))
        ax.ravel()[c].imshow(hill_dataset[image_number])
        ax.ravel()[c].set_title("Hillshade: "+str(image_number))        
        ax.ravel()[c+1].imshow(image_dataset[image_number])
        ax.ravel()[c+1].set_title("Image: "+str(image_number))
        ax.ravel()[c+2].imshow(mask_dataset[image_number][:,:,0])
        ax.ravel()[c+2].set_title("Mask: "+str(image_number))
        
        c=c+3
        j=j+1
    plt.tight_layout()
    plt.savefig(save_dir+"/"+str(name)+".png")
    
def get_augmented_data(img, mask, hill, set_name, tile_size, n):
    
    train_transform = A.Compose([
        A.RandomCrop(height=tile_size, width = tile_size, p=1),
        A.VerticalFlip(p=.5),              
        A.Rotate(p=.8),
    ],
    additional_targets={'image0': 'image', 'image1': 'image'}
    )

    val_transform = A.Compose([
        A.RandomCrop(height=tile_size, width = tile_size, p=1),
    ],
    additional_targets={'image0': 'image', 'image1': 'image'}
    )
    
    X_aug=[]
    y_aug=[]
    hill_aug=[]
    print(len(img))
    print(len(mask))
    print(len(hill))
    for i in range(n):
        if set_name=="train":
            for j in range(len(img)):
                transformed = train_transform(image=img[j], image0=mask[j], image1=hill[j])
                X_aug.append(transformed['image'])
                y_aug.append(transformed['image0'])
                hill_aug.append(transformed['image1'])
        if set_name=="val":
            for j in range(len(img)):
                transformed = val_transform(image=img[j], image0=mask[j], image1=hill[j])            
                X_aug.append(transformed['image'])
                y_aug.append(transformed['image0'])
                hill_aug.append(transformed['image1'])
        
    print(len(X_aug)) 
    return X_aug, y_aug, hill_aug
    
if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()        
    new_dir = arg.new_dir
    
    image_dataset, mask_dataset, hill_dataset = get_datalist(arg.img_dir)
    print("Length of Images: "+str(len(image_dataset)))
    print("Length of Masks: "+str(len(mask_dataset)))  
    print("Length of Hillshades: "+str(len(hill_dataset)))
    
    if not os.path.exists(arg.save_results):
        os.makedirs(arg.save_results)   
        
    visualize_image_mask_hill_sample(image_dataset, mask_dataset, hill_dataset, arg.save_results, name="visualize_image_mask_hill_sample" ,n=2)
    
    

    print("Generating Labels from masks:")
    labels = get_labels_from_list(mask_dataset)
    print("Length of Labels: "+str(len(labels)))
    print("")
    

    n=arg.tile_number 
    a = arg.split_num
    l = len(labels)

#     train= get_augmented(image_dataset[0:2], labels[0:2], "train", arg.tile_size, n)
#     val= get_augmented(image_dataset[2:4], labels[2:4], "val",arg.tile_size, n) 

    train= get_augmented_data(image_dataset[0:a], labels[0:a], hill_dataset[0:a], "train", arg.tile_size, n)
    val= get_augmented_data(image_dataset[a:l], labels[a:l], hill_dataset[a:l], "val", arg.tile_size, n) 
    
    if not os.path.exists(arg.new_dir):
        os.makedirs(arg.new_dir)      
    subfolder_names = ['train', 'val']   
    for subfolder_name in subfolder_names:
        os.makedirs(os.path.join(arg.new_dir, subfolder_name), exist_ok=True)         
    subfolder_names = ['images', 'masks', 'hills']
    for subfolder_name in subfolder_names:
        os.makedirs(os.path.join(new_dir, "train", subfolder_name), exist_ok=True) 
        os.makedirs(os.path.join(new_dir, "val", subfolder_name), exist_ok=True) 
          
    for i in range(len(train[0])):    
        im = Image.fromarray(train[0][i])
        mask = Image.fromarray(train[1][i]*255)
        hill = Image.fromarray(train[2][i])
        k=f"{i:02}"
        im.save(arg.new_dir+"/"+"train"+"/images/image"+str(k)+".png")
        mask.save(arg.new_dir+"/"+"train"+"/masks/mask"+str(k)+".png")
        hill.save(arg.new_dir+"/"+"train"+"/hills/hill"+str(k)+".png")
        
    for i in range(len(val[0])):    
        im = Image.fromarray(val[0][i])
        mask = Image.fromarray(val[1][i]*255)
        hill = Image.fromarray(val[2][i])
        k=f"{i:02}"
        im.save(arg.new_dir+"/"+"val"+"/images/image"+str(k)+".png")
        mask.save(arg.new_dir+"/"+"val"+"/masks/mask"+str(k)+".png")
        hill.save(arg.new_dir+"/"+"val"+"/hills/hill"+str(k)+".png")
        
        
    print("Image saving completed!")
    print("")
          
          
    im_path = arg.new_dir+'train/'  
    
    new_train_img, new_train_mask, new_train_hill = get_datalist_from_dir(im_path)
    
    
    new_train_img=np.array(new_train_img)
    new_train_mask=np.array(new_train_mask)
    new_train_hill = np.array(new_train_hill)
    
    print("Unique values in labels:")
    print(np.unique(new_train_mask))
    print("")

    visualize_image_mask_hill_sample(new_train_img, new_train_mask, new_train_hill, arg.save_results, name="visualize_image_mask_hill_sample_after_augmentation", n=5)
    
    print("")
    print("Now removing redundant images:")
    
    set_name=["train", "val", "test"]
          
    for i in set_name:
        print("Removing items from " +str(i))
        print("-------------------------------")
        file_dir_list = sorted(glob.glob(arg.new_dir+"/"+i +"/masks/*"))
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
        #####
        
        filtered_list_copy=filtered_mask_dir_list
        filtered_hill_dir_list=[]
        for i in filtered_list_copy:
            i = i.replace("mask","hill")
            filtered_hill_dir_list.append(i)
        print("Length of filtered hill directory: "+str(len(filtered_hill_dir_list)))
        print("")



        for i in filtered_hill_dir_list:
            if os.path.exists(i):
                os.remove(i)

        print("Removed redundant hillshade images")
        
        
    im_path = arg.new_dir+'train/'      
    image_dataset, mask_dataset, hill_dataset = get_datalist_from_dir(im_path)
    
    if not os.path.exists(arg.save_results):
        os.makedirs(arg.save_results)  
          
    #mask_dataset= np.expand_dims(mask_dataset, axis=3)
    visualize_image_mask_hill_sample(image_dataset, mask_dataset, hill_dataset, arg.save_results, name="visualize_image_mask_hill_sample_after_data_cleanup" ,n=5)
       
          
    

    print("Finished!")
          
          
    
    
    

            