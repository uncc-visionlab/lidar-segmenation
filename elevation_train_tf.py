import tensorflow
tensorflow.random.set_seed(9999)
from numpy.random import seed
seed(9999)

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
import os
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from PIL import Image
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
import albumentations as A
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import argparse
from scripts.tf_models import *
from keras.callbacks import ModelCheckpoint, EarlyStopping,  CSVLogger
#from tensorflow.keras.callbacks import CSVLogger

import tensorflow as tf
from keras.models import load_model
#tf.config.experimental.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


def get_parser():
    parser = argparse.ArgumentParser(description='Train the model (tensorflow) for LIDAR segmenation')
    
    parser.add_argument('--data-path',
                        default="/home/fjannat/Documents/EarthVision/data/hill_single/dataset_2/",
                        help='path to mage')
    parser.add_argument('--save-results',
                        default="/home/fjannat/Documents/EarthVision/Results/hill_single/trial_11/",
                        help='path to mage')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='number of batches')
    
    parser.add_argument('--img-size',
                        type=int,
                        default=128,
                        help='image size')
    parser.add_argument('--img-type',
                        type=str,
                        default="hillshade",
                        help='hillshade or elevation')
    
    parser.add_argument('--transform',
                        type=bool,
                        default=False,
                        help='True or False')
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='number of epochs')
    parser.add_argument('--model_name',
                        type=str,
                        default="new_model")
    return parser

def display_multiple_img(original, ground_truth, y_test_argmax, hill_dataset, save_dir, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=4,figsize=(12,n*5) )
    #figure = plt.figure(figsize=(30,20))
    #figure.set_size_inches(30, 20)
    c=0
    j=1
    for i in range(n):
        image_number = random.randint(0, len(ground_truth)-1)
        ax.ravel()[c].imshow(original[image_number])
        ax.ravel()[c].set_title("Elevation Image: "+str(image_number))
        ax.ravel()[c+1].imshow(ground_truth[image_number])
        ax.ravel()[c+1].set_title("Ground Truth: "+str(image_number))
        ax.ravel()[c+2].imshow(y_test_argmax[image_number])
        ax.ravel()[c+2].set_title("Predicted Image: "+str(image_number))
        
        ax.ravel()[c+3].imshow(hill_dataset[image_number])
        ax.ravel()[c+3].set_title("Hillshade Image: "+str(image_number))
        
        c=c+4
        j=j+1
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_dir+"/visualize_results.png")

def mean_iou(y_true, y_pred):

    axes = (1,2) # W,H axes of each image
    #intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_pred), axis=axes) + np.sum(np.abs(y_true), axis=axes)
    #union = mask_sum  - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth)/(mask_sum + smooth)

    iou = np.mean(iou)
    return iou

if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()   
    
    train_image_dir = os.path.join(arg.data_path, "train/images")
    train_mask_dir = os.path.join(arg.data_path, "train/masks")
    val_image_dir = os.path.join(arg.data_path, "val/images")
    val_mask_dir = os.path.join(arg.data_path, "val/masks")
    
    train_hill_dir = os.path.join(arg.data_path, "train/hills")
    val_hill_dir = os.path.join(arg.data_path, "val/hills")
    print("Training Image Directory: " + str(train_image_dir))
    print("Training Mask Directory: " + str(train_mask_dir))
    print("Validation Image Directory: " + str(val_image_dir))
    print("Validation Mask Directory: " + str(val_mask_dir))
    
    print("Training Hill Directory: " + str(train_hill_dir))
    print("Validation Hill Directory: " + str(val_hill_dir))
    print("")
    
    
    
    # Declare an augmentation pipeline
    transform = A.Compose([
        #A.RandomCrop(height=64, width = 64, p=1),
        #A.Resize(224, 224, always_apply=True),
        A.VerticalFlip(p=.5),              
        A.Blur(p=.5),
        A.Rotate(p=1),
        A.CoarseDropout (max_holes=3, max_height=2, max_width=2, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5) 
    ])

#     train_image_dataset = get_images(train_image_dir)
#     train_mask_dataset = get_images(train_mask_dir)
#     val_image_dataset = get_images(val_image_dir)
#     val_mask_dataset = get_images(val_mask_dir)
    
    #get_images_spec
    train_image_dataset = get_elevation_images_from_dir(train_image_dir, 10000, arg.img_size)
    train_mask_dataset = get_images_spec(train_mask_dir, 10000)
    val_image_dataset = get_elevation_images_from_dir(val_image_dir, 3000, arg.img_size)
    val_mask_dataset = get_images_spec(val_mask_dir, 3000)

    print("Total length of Training Images: " + str(len(train_image_dataset)))
    print("Total length of Training Masks: " + str(len(train_mask_dataset)))
    print("Total length of Validation Images: " + str(len(val_image_dataset)))
    print("Total length of Validation Masks: " + str(len(val_mask_dataset)))
    print("")
    
    train_hill_dataset = get_images_spec(train_hill_dir, 10000)
    val_hill_dataset = get_images_spec(val_hill_dir, 3000)
    
    print(" ") 
    print("Total length of Hill Training Images: " + str(len(train_hill_dataset)))
    print("Total length of Hill Validation Images: " + str(len(val_hill_dataset)))
    
    if not os.path.exists(arg.save_results):
        os.makedirs(arg.save_results)  

    #visualize_image_mask_sample(train_image_dataset, train_mask_dataset, arg.save_results, "image_mask_sample" , n=5)
    visualize_image_mask_hill_sample(train_image_dataset, train_mask_dataset, train_hill_dataset, arg.save_results, n=5)

    
    print(np.unique(val_mask_dataset))
    
    train_labels = train_mask_dataset/255
    val_labels = val_mask_dataset/255
    
    print(np.unique(val_mask_dataset))
    visualize_image_label_sample(val_image_dataset, val_labels[:,:,:,1], arg.save_results, n=5)

    n_classes = len(np.unique(train_labels))
    print("No of Total Classes: "+str(n_classes))
    

    train_labels_cat = to_categorical(train_labels[:,:,:,1], num_classes=n_classes)
    val_labels_cat = to_categorical(val_labels[:,:,:,1], num_classes=n_classes)
    
    if arg.img_type=="elevation":           
        X_train=train_image_dataset        
        X_test=val_image_dataset
        
    if arg.img_type=="hillshade": 
        X_train=train_hill_dataset        
        X_test=val_hill_dataset
        
        
    y_train=train_labels_cat
    y_test=val_labels_cat
    scale=1./255
    print(np.unique(X_train))

    X_train = X_train*scale
    X_test = X_test*scale

    print(np.unique(X_train))

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    #IMG_CHANNELS = 1
    print("Image Size: "+str(IMG_HEIGHT))
    


    #metrics=['accuracy', dice_coef]
    metrics=['accuracy', jacard_coef]
    #model = get_model()
    model = multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
    
    
    
    # save best model with maximum validation accuracy
    #checkpoint = ModelCheckpoint(arg.save_results, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    checkpoint = ModelCheckpoint(arg.save_results, monitor="val_mean_io_u", verbose=1, save_best_only=True, mode="max")

    # stop model training early if validation loss doesn't continue to decrease over 2 iterations
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="min")
    csv_logger = os.path.join(arg.save_results, "csv_logger.csv")
    # log training console output to csv
    csv_logger = CSVLogger(csv_logger, separator=",", append=False)

    # create list of callbacks
    callbacks_list = [checkpoint, csv_logger]  # early_stopping
    
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    #model.compile(optimizer='adam', loss="binary_focal_crossentropy", metrics=metrics)
    #model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=metrics)
    #model.compile(optimizer='adam', loss=binary_crossentropy, metrics=metrics)
    model.summary() 
    
    history = model.fit(X_train, y_train, 
                    batch_size = arg.batch_size, 
                    verbose=1, 
                    epochs=arg.epochs, 
                    validation_data=(X_test, y_test), 
                    shuffle=True,
                    callbacks=callbacks_list)
    
    model.save(arg.save_results+'/trained_model.hdf5')
    
    plot_result(history, "loss", "val_loss", "Loss", arg.save_results)
    plot_result(history, "mean_io_u", "val_mean_io_u", "MeanIOU", arg.save_results)
    
    model_path= arg.save_results+'/trained_model.hdf5'
    model = load_model(model_path, compile = False)
    
    
    y_pred=model.predict(X_test[0:500], batch_size=4)
    y_pred_argmax=np.argmax(y_pred, axis=3)
    original = X_test[0:500]
    #original = val_image_dataset
    y_test_argmax=np.argmax(y_test[0:500], axis=3)
    ground_truth=y_test_argmax
    
    #import pdb;pdb.set_trace()
    display_multiple_img(val_image_dataset, ground_truth, y_pred_argmax, val_hill_dataset, arg.save_results, n=20)
    print("Mean IOU:")
    iou= mean_iou(ground_truth, y_pred_argmax)
    print(iou)
    
    filedir = os.path.join(arg.save_results, "save_notes.txt")
    
    with open(filedir, 'w') as f:
        f.write('Mean IOU:')
        f.write(str(iou))
        f.write("")
        f.write("Image Type:")
        f.write(arg.img_type)
        f.write("Loss function:")
        f.write("tf.keras.losses.CosineSimilarity Loss")
        f.write("Metrics:")
        f.write("Mean IOU")
        f.write("Optimizer:")
        f.write("SGD")
    
    

