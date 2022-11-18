#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.utils import normalize
import os
import cv2
# from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
# import tifffile as tiff
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf
from keras.metrics import MeanIoU
# import segmentation_models as sm
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# from keras.layers import Input
# from keras.models import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

import segmentation_models as sm

# import albumentations as A

from tensorflow.keras.optimizers import Adam

from datetime import datetime

# from PIL import Image
# from keras import backend, optimizers

# from focal_loss import BinaryFocalLoss
from att_models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef


# In[3]:


def display_multiple_img(hill, original, ground_truth, y_test_argmax, plt_name, save_results, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=4, figsize=(12, n * 5))
    c = 0
    j = 1
    for i in range(n):
        image_number = random.randint(0, len(ground_truth) - 1)
        ax.ravel()[c].imshow(original[image_number], cmap='gray')
        ax.ravel()[c].set_title("Hillshade Image: " + str(image_number))
        ax.ravel()[c + 1].imshow(ground_truth[image_number], cmap='gray')
        ax.ravel()[c + 1].set_title("Ground Truth: " + str(image_number))
        ax.ravel()[c + 2].imshow(y_test_argmax[image_number], cmap='gray')
        ax.ravel()[c + 2].set_title("Predicted Image: " + str(image_number))

        ax.ravel()[c + 3].imshow(hill[image_number], cmap='gray')
        ax.ravel()[c + 3].set_title("Hillshade Image: " + str(image_number))

        c = c + 4
        j = j + 1
    plt.tight_layout()
    plt.savefig(save_results + "/" + str(plt_name) + ".png")
    plt.show()


def display_learning_curves(history):
    result = history.history
    param = []
    for key in result:
        param.append(key)
    l = int(len(result) / 2)
    print(l)
    n_epochs = range(1, len(history.history[param[0]]) + 1)
    print(n_epochs)

    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(n_epochs, history.history[param[0]], label=str(param[0]))
    plt.plot(n_epochs, history.history[param[0 + l]], label=str(param[0 + l]))
    plt.title(str(param[0]))
    plt.xlabel("Epoch")
    plt.ylabel(str(param[0]))
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 2)
    plt.plot(n_epochs, history.history[param[1]], label=str(param[1]))
    plt.plot(n_epochs, history.history[param[1 + l]], label=str(param[1 + l]))
    plt.title(str(param[1]))
    plt.xlabel("Epoch")
    plt.ylabel(str(param[1]))
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 3)
    plt.plot(n_epochs, history.history[param[2]], label=str(param[2]))
    plt.plot(n_epochs, history.history[param[2 + l]], label=str(param[2 + l]))
    plt.title(str(param[2]))
    plt.xlabel("Epoch")
    plt.ylabel(str(param[2]))
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def get_rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label = label

    label_seg = np.zeros(label.shape, dtype=np.float32)
    # when platforms
    #     label_seg [np.all(label>=150,axis=-1)] = 1
    #     label_seg [np.all(label<150,axis=-1)] = 0
    # when annular structure

    label_seg[np.all(label >= 254, axis=-1)] = 1
    label_seg[np.all(label < 254, axis=-1)] = 0

    return label_seg


def get_labels_from_mask(mask_dataset):
    labels = []
    for i in range(len(mask_dataset)):
        label = get_rgb_to_2D_label(mask_dataset[i])
        labels.append(label)
    labels = np.array(labels)
    return labels


def get_area_covered(img, th_area=2):
    height = img.shape[0]
    width = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
    count = cv2.countNonZero(thresh)
    area = (1 - count / (width * height)) * 100
    if area > th_area:
        return True
    else:
        return False


def get_column(mask, i):
    col = [row[i] for row in mask]
    col = np.array(col)
    col = np.hstack(col.flatten())
    if 0 in col:
        return True
    else:
        return False


def get_row(mask, i):
    row = np.array(mask[i])
    row = np.hstack(row.flatten())
    if 0 in row:
        return True
    else:
        return False


def check_if_obj_border(mask):
    x, y, z = mask.shape
    # checking first and last row
    row_f = get_row(mask, 0)
    row_l = get_row(mask, y - 1)
    # checking first and last column
    col_f = get_column(mask, 0)
    col_l = get_column(mask, y - 1)
    op = [row_f, row_l, col_f, col_l]
    if True in op:
        return True

    else:
        return False


def get_image_mask_patches(img_dir, mask_dir, hill_dir, img_size=128, step=20, th_area=2):
    large_image_stack = cv2.imread(img_dir)
    large_mask_stack = cv2.imread(mask_dir)[:, :, 0:1]
    large_hill_stack = cv2.imread(hill_dir)
    print(large_image_stack.shape)
    print(large_mask_stack.shape)
    print(large_hill_stack.shape)

    patches_img = patchify(large_image_stack, (img_size, img_size, 3),
                           step=step)  # Step=128 for 128 patches means no overlap

    patches_mask = patchify(large_mask_stack, (img_size, img_size, 1),
                            step=step)  # Step=128 for 128 patches means no overlap

    patches_hill = patchify(large_hill_stack, (img_size, img_size, 3),
                            step=step)  # Step=128 for 128 patches means no overlap

    all_img_patches = []
    all_mask_patches = []
    all_hill_patches = []

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):

            single_patch_mask = patches_mask[i, j, 0, :, :, :]

            single_patch_mask = (single_patch_mask.astype('float32'))
            # single_patch_mask = get_labels_from_mask(single_patch_mask)
            # area_thresh = get_area_covered(single_patch_mask, th_area)
            # if len(np.unique(single_patch_mask))>1 and area_thresh:
            if len(np.unique(single_patch_mask)) > 1:
                # check =check_if_obj_border(single_patch_mask[:,:,0:1])
                # if check!=True:
                all_mask_patches.append(single_patch_mask[:, :, 0:1])

                single_patch_img = patches_img[i, j, 0, :, :, :]
                single_patch_img = (single_patch_img.astype('float32')) / 255.
                all_img_patches.append(single_patch_img)

                single_patch_hill = patches_hill[i, j, 0, :, :, :]
                single_patch_hill = (single_patch_hill.astype('float32')) / 255.
                all_hill_patches.append(single_patch_hill)

    images = np.array(all_img_patches)
    masks = np.array(all_mask_patches)
    hills = np.array(all_hill_patches)

    return images, masks, hills


def get_sample_display_multiple_img(original, ground_truth, hillshade, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=3, figsize=(12, n * 5))
    c = 0
    j = 1

    for i in range(n):
        image_number = random.randint(0, len(ground_truth) - 1)
        ax.ravel()[c].imshow(original[image_number][:, :, 0:1], cmap='gray')
        ax.ravel()[c].set_title("Original Image: " + str(image_number))
        ax.ravel()[c + 1].imshow(ground_truth[image_number], cmap='gray')
        ax.ravel()[c + 1].set_title("Ground Truth: " + str(image_number))
        ax.ravel()[c + 2].imshow(hillshade[image_number][:, :, 0:1], cmap='gray')
        ax.ravel()[c + 2].set_title("Hillshade: " + str(image_number))
        c = c + 3
        j = j + 1
    plt.tight_layout()
    plt.show()


# In[4]:

if __name__ == "__main__":
    home_folder = '/home/fjannat/Documents/EarthVision/data_resource/'
    home_folder = '/home/arwillis/PyCharm/'
    results_folder = 'results/'

    gis_data_path = ['data/KOM/raw/', 'data/MLS/raw/', 'data/UCB/raw/']
    gis_input_filenames = ['kom_dsm_lidar.png',
                           'MLS_DEM.png',
                           'UCB_elev_adjusted.png']
    gis_input_gt_filenames = ['kom_dsm_lidar_gt.png',
                              'MLS_DEM_gt.png',
                              'UCB_elev_adjusted_gt.png']
    img_dir1 = home_folder + gis_data_path[0] + gis_input_filenames[0]
    mask_dir1 = home_folder + gis_data_path[0] + gis_input_gt_filenames[0]
    hill_dir1 = img_dir1

    img_dir2 = home_folder + gis_data_path[1] + gis_input_filenames[1]
    mask_dir2 = home_folder + gis_data_path[1] + gis_input_gt_filenames[1]
    hill_dir2 = img_dir2

    img_dir3 = home_folder + gis_data_path[2] + gis_input_filenames[2]
    mask_dir3 = home_folder + gis_data_path[2] + gis_input_gt_filenames[2]
    hill_dir3 = img_dir3

    img1, mask1, hill1 = get_image_mask_patches(img_dir1, mask_dir1, hill_dir1, img_size=128, step=40)
    print(len(img1))

    img2, mask2, hill2 = get_image_mask_patches(img_dir2, mask_dir2, hill_dir2, img_size=128, step=40)
    print(len(img2))

    img3, mask3, hill3 = get_image_mask_patches(img_dir3, mask_dir3, hill_dir3, img_size=128, step=20)
    print(len(img3))

    # split the data within each image test/train
    # test 20%
    # merge all the train and split the training into train/validate
    # train 65%
    # validate 15%
    train_img = np.concatenate((img2, img3), axis=0)
    train_mask = np.concatenate((mask2, mask3), axis=0)
    train_hill = np.concatenate((hill2, hill3), axis=0)
    # train_img = np.concatenate((img1, img2, img3), axis=0)
    # train_mask = np.concatenate((mask1, mask2, mask3), axis=0)
    # train_hill = np.concatenate((hill1, hill2, hill3), axis=0)

    val_img = img1
    val_mask = mask1
    val_hill = hill1

    print(train_img.shape)
    print(val_img.shape)

    get_sample_display_multiple_img(train_img, train_mask, train_hill, n=5)

    ###############################################
    # Encode labels... but multi dim array so need to flatten, encode and reshape
    from sklearn.preprocessing import LabelEncoder

    labelencoder = LabelEncoder()
    n, h, w = train_mask[:, :, :, 0].shape
    train_masks_reshaped = train_mask[:, :, :, 0].reshape(-1, 1)
    print(train_masks_reshaped.shape)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    print(np.unique(train_masks_encoded_original_shape))

    n, h, w = val_mask[:, :, :, 0].shape
    val_masks_reshaped = val_mask[:, :, :, 0].reshape(-1, 1)
    val_masks_reshaped_encoded = labelencoder.fit_transform(val_masks_reshaped)
    val_masks_encoded_original_shape = val_masks_reshaped_encoded.reshape(n, h, w)
    print(np.unique(val_masks_encoded_original_shape))

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
    val_masks_input = np.expand_dims(val_masks_encoded_original_shape, axis=3)
    print(train_masks_input.shape)

    X_train = train_img
    X_val = val_img

    y_train = train_masks_input
    y_val = val_masks_input
    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled

    from tensorflow.keras.utils import to_categorical

    n_classes = 2

    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

    val_masks_cat = to_categorical(y_val, num_classes=n_classes)
    y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))

    X_train = np.expand_dims(X_train[:, :, :, 1], axis=3)
    X_val = np.expand_dims(X_val[:, :, :, 1], axis=3)

    #######################################
    # Parameters for model

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    input_shape

    BATCH_SIZE = 40
    EPOCH = 500

    sm.set_framework('tf.keras')

    sm.framework()

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([.5, .5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [total_loss, sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    save_results = home_folder + results_folder + 'unet/trial'

    if not os.path.exists(save_results):
        os.makedirs(save_results)

    checkpoint = ModelCheckpoint(save_results, monitor="val_iou_score", verbose=1, save_best_only=True, mode="max")
    early_stopping = EarlyStopping(monitor="val_iou_score", patience=150, verbose=1, mode="max")

    # create list of callbacks
    callbacks_list = [checkpoint, early_stopping]  # early_stopping

    """
    UNet
    """
    unet_model = UNet(input_shape, NUM_CLASSES=2)
    unet_model.compile(optimizer=Adam(lr=1e-2), loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.1),
                       metrics=metrics)

    # print(unet_model.summary())

    start1 = datetime.now()
    unet_history = unet_model.fit(X_train, y_train_cat,
                                  verbose=1,
                                  batch_size=BATCH_SIZE,
                                  validation_data=(X_val, y_val_cat),
                                  shuffle=True,
                                  epochs=EPOCH,
                                  callbacks=callbacks_list)

    stop1 = datetime.now()
    # Execution time of the model
    execution_time_Unet = stop1 - start1
    print("UNet execution time is: ", execution_time_Unet)

    unet_model.save(save_results + '/as_unet.hdf5')

    display_learning_curves(unet_history)

    unet_model.load_weights(save_results)
    y_pred1 = unet_model.predict(X_val)
    y_pred1_argmax = np.argmax(y_pred1, axis=3)

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    print(IOU_keras)
    IOU_keras.update_state(val_mask[:, :, :, 0], y_pred1_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)

    test_img = X_val
    ground_truth = y_val_cat[:, :, :, 1]
    test_pred1 = unet_model.predict(test_img)
    test_prediction1 = np.argmax(test_pred1, axis=3)

    display_multiple_img(val_hill, test_img, ground_truth, test_prediction1, 'unet_plt_1', save_results, n=5)
