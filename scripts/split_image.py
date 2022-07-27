import os
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from PIL import Image
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Split image and mask to train and val tiles')
    
    parser.add_argument('--image-path',
                        default="/home/fjannat/Documents/EarthVision/data_resource/uchbenmul/img.png",
                        help='path to mage')
    parser.add_argument('--mask-path',
                        default="/home/fjannat/Documents/EarthVision/data_resource/uchbenmul/mask.png",
                        help='path to mask')
    parser.add_argument('--save-dir',
                        default="/home/fjannat/Documents/EarthVision/data/raw_2/",
                        help='path to mage')
   
    return parser

if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()   


    im =  cv2.imread(arg.image_path)
    msk =  cv2.imread(arg.mask_path)
    
    imgheight=im.shape[0]
    imgwidth=im.shape[1]

    y1 = 0
    M = int(imgheight//3)*2
    N = int(imgwidth)
    c=0
    
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir)  
        
        
    for y in range(0,imgheight,M):
        for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N
            im_tiles = im[y:y+M,x:x+N]
            msk_tiles = msk[y:y+M,x:x+N]
            #print(tiles.shape)
            #print(c)
            c=c+1

            cv2.rectangle(im, (x, y), (x1, y1), (0, 255, 0))
            cv2.imwrite(arg.save_dir + "img" + '_' + str(c)+".png",im_tiles)
            cv2.imwrite(arg.save_dir + "mask" + '_' + str(c)+".png",msk_tiles)


