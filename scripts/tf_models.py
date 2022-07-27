from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
import os
import cv2
import numpy as np
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from PIL import Image
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import glob
import albumentations as A
import skimage.io as io
from skimage.transform import resize
from numpy import asarray


def multi_unet_model(n_classes=2, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    #outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model


def get_images(data_path):
    file_dataset = []
    for filename in sorted(glob.glob(data_path+'/*.png')): #assuming png
        im=Image.open(filename)
        a = np.asarray(im)
        #a = np.expand_dims(a, axis=2)
        file_dataset.append(a)
    file_dataset = np.array(file_dataset)
    return file_dataset
    

def get_images_spec(data_path, n):
    file_dataset = []
    c=0
    for filename in sorted(glob.glob(data_path+'/*.png')): #assuming png
        im=Image.open(filename)
        a = np.asarray(im)
        #a = np.expand_dims(a, axis=2)
        file_dataset.append(a)
        c=c+1
        if c>=n:
            break
    file_dataset = np.array(file_dataset)
    return file_dataset

def get_elevation_images_from_dir(data_path, n, im_size):
    size=(im_size,im_size)
    file_dataset = []
    c=0
    for filename in sorted(glob.glob(data_path+'/*.png')): #assuming png
        im = io.imread(filename)
        # im = rgb2gray(im)
        im = np.array(im, np.float32)
        # im = np.moveaxis(im, 2, -3)
        im = resize(im, size, anti_aliasing=True)
        pixels = asarray(im)
        pixels = pixels.astype('float32')
        mean, std = pixels.mean(), pixels.std()
        # print('Before normalization', 'Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
        # global standardization of pixels
        pixels = (pixels - mean) / std
        mean2, std2 = pixels.mean(), pixels.std()
        # assert [np.isclose([mean2, std2], [0, 1.0], atol=0.0001)] == [ True, True]
        #pixels = np.moveaxis(pixels, 2, -3) # move channels to last i.e: [C,W,H]
        # print('images', pixels.shape)
        file_dataset.append(pixels)
        c=c+1
        if c>=n:
            break
    file_dataset = np.array(file_dataset)
    return file_dataset 


def dice_loss(y_true, y_pred, smooth=0.01):
    # flatten
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # one-hot encoding y with 3 labels : 0=background, 1=label1, 2=label2
    y_true_f = K.one_hot(K.cast(y_true_f, np.uint8), 3)
    y_pred_f = K.one_hot(K.cast(y_pred_f, np.uint8), 3)
    # calculate intersection and union exluding background using y[:,1:]
    intersection = K.sum(y_true_f[:,1:]* y_pred_f[:,1:], axis=[-1])
    mask_sum = K.sum(y_true_f[:,1:], axis=[-1]) + K.sum(y_pred_f[:,1:], axis=[-1])
    union = mask_sum - 1
    # apply dice formula
    dice = K.mean((2. * intersection + smooth)/(mask_sum + smooth), axis=0)
    #iou = K.mean((intersection + smooth)/(mask_sum -1 + smooth), axis=0)
    dice_loss = 1-dice
    return dice_loss

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

    
def visualize_image_mask_sample(image_dataset, mask_dataset, save_dir, name,  n=5):
    rows=n
    cols=2
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(4,n*4) )
    c=0
    j=1
    for i in range(n):
        image_number = random.randint(0, (len(image_dataset)-1))
        ax.ravel()[c].imshow(image_dataset[image_number])
        ax.ravel()[c].set_title("Image: "+str(image_number))
        ax.ravel()[c+1].imshow(mask_dataset[image_number][:,:,0])
        ax.ravel()[c+1].set_title("Mask: "+str(image_number))
        
        c=c+2
        j=j+1
    plt.tight_layout()
    plt.savefig(save_dir+"/"+str(name)+".png")
    

    
def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    #label_seg = np.zeros(label.shape,dtype=np.uint8)
    # label_seg [np.all(label==anstr,axis=-1)] = 0
    # label_seg [np.all(label==Unlabeled,axis=-1)] = 1
    
    # label_seg = label_seg[:,:]  #Just take the first channel, no need for all 3 channels
    #label_seg = np.where((label == 255), 1, label)
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label >= 200,axis=-1)] = 0
    label_seg [np.all(label<200,axis=-1)] = 1
    #print("hi")
    
    return label_seg


def get_labels(mask_dataset):
    anstr = '#FFFFFF'.lstrip('#')
    anstr = np.array(tuple(int(anstr[i:i+1], 16) for i in (0, 2))) # 60, 16, 152

    Unlabeled = '#000000'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+1], 16) for i in (0, 2))) #155, 155, 155

    #label = mask_dataset[1]
    
    labels = []
    for i in range(mask_dataset.shape[0]):
        label = rgb_to_2D_label(mask_dataset[i])
        labels.append(label)    

    labels = np.array(labels)   
    #labels = np.expand_dims(labels, axis=3)
    print(labels.shape)
    print("Unique labels in label dataset are: ", np.unique(labels))
    return labels


def get_labels_from_list(mask_dataset):
    anstr = '#FFFFFF'.lstrip('#')
    anstr = np.array(tuple(int(anstr[i:i+1], 16) for i in (0, 2))) # 60, 16, 152

    Unlabeled = '#000000'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+1], 16) for i in (0, 2))) #155, 155, 155


    
    labels = []
    for i in range(len(mask_dataset)):
        label = rgb_to_2D_label(mask_dataset[i])
        labels.append(label)    
   
    return labels

def get_augmented(img, mask, set_name, tile_size, n):
    
    train_transform = A.Compose([
        A.RandomCrop(height=tile_size, width = tile_size, p=1),
        A.VerticalFlip(p=.5),              
        #A.Blur(p=.5),
        A.Rotate(p=.8),
#         A.VerticalFlip(p=.5),              
#         A.Blur(p=1),
#         A.RandomRotate90(p=1),
#         A.Rotate(limit=(-90,90),p=1),
#         A.HorizontalFlip(p=.5),
    ])

    val_transform = A.Compose([
        A.RandomCrop(height=tile_size, width = tile_size, p=1),
    ])
    
    X_aug=[]
    y_aug=[]
    for i in range(n):
        if set_name=="train":
            for j in range(len(img)):
                transformed = train_transform(image=img[j], mask=mask[j])
                X_aug.append(transformed['image'])
                y_aug.append(transformed['mask'])
        if set_name=="val":
            for j in range(len(img)):
                transformed = val_transform(image=img[j], mask=mask[j])            
                X_aug.append(transformed['image'])
                y_aug.append(transformed['mask'])
        
    print(len(X_aug)) 
    return X_aug, y_aug



    
def visualize_image_label_sample(image_dataset, label_dataset, save_dir, n=5):
    rows=n
    cols=2
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(4,n*4) )
    c=0
    j=1
    for i in range(n):
        image_number = random.randint(0, len(image_dataset))
        ax.ravel()[c].imshow(image_dataset[image_number])
        ax.ravel()[c].set_title("Image: "+str(image_number))
        ax.ravel()[c+1].imshow(label_dataset[image_number]*255)
        #ax.ravel()[c+1].imshow(label_dataset[i]*255)
        ax.ravel()[c+1].set_title("Mask: "+str(image_number))
        
        c=c+2
        j=j+1
    plt.tight_layout()
    plt.savefig(save_dir+"/visualize_image_label_sample.png")
        
def visualize_image_mask_hill_sample(image_dataset, label_dataset, hill_dataset, save_dir, n=5):
    rows=n
    cols=3
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(6,n*4) )
    c=0
    j=1
    for i in range(n):
        image_number = random.randint(0, (len(image_dataset)-1))
        ax.ravel()[c].imshow(image_dataset[image_number])
        ax.ravel()[c].set_title("Elevation Image: "+str(image_number))
        ax.ravel()[c+1].imshow(label_dataset[image_number][:,:,0])
        #ax.ravel()[c+1].imshow(label_dataset[i]*255)
        ax.ravel()[c+1].set_title("Mask: "+str(image_number))
        
        ax.ravel()[c+2].imshow(hill_dataset[image_number])
        #ax.ravel()[c+1].imshow(label_dataset[i]*255)
        ax.ravel()[c+2].set_title("Hillshade: "+str(image_number))
        
        
        c=c+3
        j=j+1
    plt.tight_layout()
    plt.savefig(save_dir+"/visualize_image_label_hill_sample.png")
        
        
        
        
def get_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


def plot_result(history, param_1, param_2, type_name, save_dir):
    fig = plt.figure(figsize =(4, 4))
    history = history
    result = history.history[param_1]
    val_result = history.history[param_2]
    epochs = range(1, len(result) + 1)
    plt.plot(epochs, result, 'y', label=param_1)
    plt.plot(epochs, val_result, 'r', label=param_2)
    plt.title('Training and validation '+ param_1)
    plt.xlabel('Epochs')
    plt.ylabel(type_name)
    plt.legend()
    #plt.show()
    #plt.savefig(os.path.join(save_dir, (str(type_name)+".png")))
    plt.savefig(save_dir+"/"+str(type_name)+".png")
    
    
    
    

