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
import pickle
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


def display_multiple_img(original, ground_truth, y_test_argmax, plt_name, save_results, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=3, figsize=(12, n * 2))
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
        c = c + 3
        j = j + 1
    plt.tight_layout()
    plt.savefig(save_results + "/" + str(plt_name) + ".png")
    plt.show()


def getRigidImagePatch(img, height, width, center_y, center_x, angle):
    theta = angle / 180 * np.pi
    xy_center = (width / 2, height / 2)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    bound_w = int(height * np.abs(sin_t) + width * np.abs(cos_t))
    bound_h = int(height * np.abs(cos_t) + width * np.abs(sin_t))
    xy_start = np.array([np.floor(center_x - bound_w / 2), np.floor(center_y - bound_h / 2)], dtype=np.int32)
    xy_end = np.array([np.ceil(center_x + bound_w / 2), np.ceil(center_y + bound_h / 2)], dtype=np.int32)
    if np.any(xy_start < 0) or xy_start[0] > img.shape[1] or xy_start[1] > img.shape[0] or \
            np.any(xy_end < 0) or xy_end[0] > img.shape[1] or xy_end[1] > img.shape[0]:
        print("Could not extract patch at location (" + str((center_x, center_y)) + ")")
        return None
    cropped_image_patch = img[xy_start[0]:xy_end[0], xy_start[1]:xy_end[1], :]
    cropped_height = cropped_image_patch.shape[0]
    cropped_width = cropped_image_patch.shape[1]

    # xy_rotation_centerpt = np.array([width / 2, height / 2])
    xy_translation = np.array([0.5 * (cropped_width - (cos_t * cropped_width + sin_t * cropped_height)),
                               0.5 * (cropped_height - (-sin_t * cropped_width + cos_t * cropped_height))])
    image_patch_T = np.float32([[cos_t, sin_t, xy_translation[0]], [-sin_t, cos_t, xy_translation[1]]])
    transformed_image_patch = cv2.warpAffine(cropped_image_patch, image_patch_T, (cropped_width, cropped_height),
                                             flags=cv2.INTER_CUBIC)

    xy_center_newimg = np.int32(np.array(transformed_image_patch.shape[:2]) / 2.0)
    xy_start = np.array([xy_center_newimg[0] - width / 2, xy_center_newimg[1] - height / 2], dtype=np.int32);
    xy_end = np.array([xy_center_newimg[0] + width / 2, xy_center_newimg[1] + height / 2], dtype=np.int32);
    image_patch_aug = transformed_image_patch[xy_start[1]:xy_end[1], xy_start[0]:xy_end[0]]
    return image_patch_aug

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


def get_image_mask_patches(img_dir, mask_dir, img_size=128, step=20, th_area=2):
    large_image_stack = cv2.imread(img_dir)
    large_mask_stack = cv2.imread(mask_dir)[:, :, 0:1]
    print(large_image_stack.shape)
    print(large_mask_stack.shape)

    # Step=128 for 128 patches means no overlap
    patches_img = patchify(large_image_stack, (img_size, img_size, 3), step=step)
    patches_mask = patchify(large_mask_stack, (img_size, img_size, 1), step=step)

    all_img_patches = []
    all_mask_patches = []

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):

            single_patch_mask = patches_mask[i, j, 0, :, :, :]

            single_patch_mask = (single_patch_mask.astype('float32'))
            # single_patch_mask = get_labels_from_mask(single_patch_mask)
            # area_thresh = get_area_covered(single_patch_mask, th_area)
            # if len(np.unique(single_patch_mask))>1 and area_thresh:
            WINDOWSIZE=41;
            if np.count_nonzero(single_patch_mask > 0.0) == WINDOWSIZE*WINDOWSIZE:
                # check =check_if_obj_border(single_patch_mask[:,:,0:1])
                # if check!=True:
                all_mask_patches.append(single_patch_mask[:, :, 0:1])

                single_patch_img = patches_img[i, j, 0, :, :, :]
                single_patch_img = (single_patch_img.astype('float32')) / 255.0
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)
    masks = np.array(all_mask_patches)
    return images, masks


def get_sample_display_multiple_img(original, ground_truth, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=2, figsize=(12, n * 2))
    c = 0
    j = 1

    for i in range(n):
        image_number = random.randint(0, ground_truth.shape[0])
        ax.ravel()[c].imshow(original[image_number], cmap='gray')
        ax.ravel()[c].set_title("Original Image: " + str(image_number))
        ax.ravel()[c + 1].imshow(ground_truth[image_number], cmap='gray')
        ax.ravel()[c + 1].set_title("Ground Truth: " + str(image_number))
        c = c + 2
        j = j + 1
    plt.tight_layout()
    plt.show()

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


if __name__ == "__main__":

    BATCH_SIZE = 60
    NUM_EPOCHS = 300
    IMAGE_SIZE = 128

    # Image augmentation settings
    NUM_AUGMENTATIONS_PER_IMAGE = 100
    SHOW_AUGMENTATION = False

    # split the data within each image test/train
    # test 20%
    # merge all the train and split the training into train/validate
    # train 65%
    # validate 15%
    pct_test = 0.2
    pct_val = 0.15

    home_folder = '/home/fjannat/Documents/EarthVision/data_resource/'
    home_folder = '/home/arwillis/PyCharm/'
    results_folder = 'results/'
    trial_folder = 'unet/trial'
    model_filename = '/as_unet.hdf5'
    training_filename = '/data_training.pkl'
    testing_filename = '/data_testing.pkl'
    validation_filename = '/data_validation.pkl'

    gis_data_path = ['data/KOM/raw/', 'data/MLS/raw/', 'data/UCB/raw/']
    gis_input_filenames = ['kom_dsm_lidar.png',
                           'MLS_DEM.png',
                           'UCB_elev_adjusted.png']
    gis_input_gt_filenames = ['kom_dsm_lidar_gt.png',
                              'MLS_DEM_gt.png',
                              'UCB_elev_adjusted_gt.png']

    img_filename1 = home_folder + gis_data_path[0] + gis_input_filenames[0]
    mask_filename1 = home_folder + gis_data_path[0] + gis_input_gt_filenames[0]

    img_filename2 = home_folder + gis_data_path[1] + gis_input_filenames[1]
    mask_filename2 = home_folder + gis_data_path[1] + gis_input_gt_filenames[1]

    img_filename3 = home_folder + gis_data_path[2] + gis_input_filenames[2]
    mask_filename3 = home_folder + gis_data_path[2] + gis_input_gt_filenames[2]


    image_1 = cv2.imread(img_filename1)
    mask_1 = cv2.imread(mask_filename1)[:, :, 0:1]
    image_2 = cv2.imread(img_filename2)
    mask_2 = cv2.imread(mask_filename2)[:, :, 0:1]
    image_3 = cv2.imread(img_filename3)
    mask_3 = cv2.imread(mask_filename3)[:, :, 0:1]

    dataset_images =[image_1, image_2, image_3]
    dataset_masks = [mask_1, mask_2, mask_3]

    # Apply the Component analysis function
    augmentation_data = []

    #
    # TODO: ARW
    # Should select region indices for training, testing and validation then perform augmentation on the indices
    # separately so the data can be added to the correct dataset.
    #
    if NUM_AUGMENTATIONS_PER_IMAGE > 0:
        if SHOW_AUGMENTATION:
            figure, handle = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))

        for datasetIdx in range(len(dataset_images)):
            analysis = cv2.connectedComponentsWithStats(dataset_masks[datasetIdx], cv2.CV_32S)
            (totalLabels, label_img, regionStats, regionCentroids) = analysis
            region_centroids = []
            for regionIdx in range(regionStats.shape[0]):
                if regionStats[regionIdx][2] == 41 and regionStats[regionIdx][3] == 41:
                    region_centroids.append(regionCentroids[regionIdx])
            dataset_augmentations = []
            augmentation_data.append(dataset_augmentations);
            for regionIdx in range(len(region_centroids)):
                for augmentationIdx in range(NUM_AUGMENTATIONS_PER_IMAGE):
                    sample = {'data': [], 'labels': []}
                    angle = np.random.uniform(low=0, high=359.9)
                    center_y, center_x = region_centroids[regionIdx]
                    dx = np.random.uniform(low=-30, high=30)
                    dy = np.random.uniform(low=-30, high=30)

                    aug_image_patch = getRigidImagePatch(dataset_images[datasetIdx],
                                                         IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle)
                    aug_mask_patch = getRigidImagePatch(dataset_masks[datasetIdx],
                                                        IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle)

                    aug_image_patch = np.array(aug_image_patch, dtype=np.float32) / 255.0
                    sample['data'].append(aug_image_patch)
                    sample['labels'].append(aug_mask_patch)
                    dataset_augmentations.append(sample)
                    if aug_image_patch is not None and SHOW_AUGMENTATION:
                        handle[0].imshow(aug_image_patch, cmap='gray')
                        handle[1].imshow(aug_mask_patch, cmap='gray')
                        plt.pause(0.5)

    img1, mask1 = get_image_mask_patches(img_filename1, mask_filename1, img_size=IMAGE_SIZE, step=40)
    print(len(img1))

    img2, mask2 = get_image_mask_patches(img_filename2, mask_filename2, img_size=IMAGE_SIZE, step=40)
    print(len(img2))

    img3, mask3 = get_image_mask_patches(img_filename3, mask_filename3, img_size=IMAGE_SIZE, step=20)
    print(len(img3))

    data = []
    numSamples = [img1.shape[0], img2.shape[0], img3.shape[0]]
    imagesets = [img1, img2, img3]
    labelsets = [mask1, mask2, mask3]

    for datasetIdx in range(len(imagesets)):
        samples = []
        for sampleIdx in range(numSamples[datasetIdx]):
            sample = {}
            sample['data'] = imagesets[datasetIdx][sampleIdx][:, :, 0]
            sample['labels'] = labelsets[datasetIdx][sampleIdx][:, :, 0]
            samples.append(sample)
        data.append(samples)

    data.append(augmentation_data[0])

    from sklearn.model_selection import train_test_split

    training_data = []
    test_data = []
    validation_data = []
    for datasetIdx in range(len(imagesets)):
        train, test = train_test_split(data[datasetIdx], test_size=pct_test, random_state=1)
        train, val = train_test_split(train, test_size=pct_val, random_state=1)
        training_data.append(train)
        test_data.append(test)
        validation_data.append(val)

    training_images = []
    training_labels = []
    for datasetIdx in range(3):
        for sampleIdx in range(len(training_data[datasetIdx])):
            training_images.append(training_data[datasetIdx][sampleIdx]['data'])
            training_labels.append(training_data[datasetIdx][sampleIdx]['labels'])

    val_images = []
    val_labels = []
    for datasetIdx in range(3):
        for sampleIdx in range(len(validation_data[datasetIdx])):
            val_images.append(validation_data[datasetIdx][sampleIdx]['data'])
            val_labels.append(validation_data[datasetIdx][sampleIdx]['labels'])

    test_images = []
    test_labels = []
    for datasetIdx in range(3):
        for sampleIdx in range(len(test_data[datasetIdx])):
            test_images.append(test_data[datasetIdx][sampleIdx]['data'])
            test_labels.append(test_data[datasetIdx][sampleIdx]['labels'])

    train_img = np.array(training_images)
    train_mask = np.array(training_labels)

    val_img = np.array(val_images)
    val_mask = np.array(val_labels)

    test_img = np.array(test_images)
    test_mask = np.array(test_labels)

    print("Train " + str(train_img.shape[0]))
    print("Validation " + str(val_img.shape[0]))
    print("Test " + str(test_img.shape[0]))

    get_sample_display_multiple_img(train_img, train_mask, n=5)

    ###############################################
    # Encode labels... but multi dim array so need to flatten, encode and reshape
    from sklearn.preprocessing import LabelEncoder

    labelencoder = LabelEncoder()
    n, h, w = train_mask[:, :, :].shape
    train_masks_reshaped = train_mask[:, :, :].reshape(-1, 1)
    print(train_masks_reshaped.shape)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    print(np.unique(train_masks_encoded_original_shape))

    n, h, w = val_mask[:, :, :].shape
    val_masks_reshaped = val_mask[:, :, :].reshape(-1, 1)
    val_masks_reshaped_encoded = labelencoder.fit_transform(val_masks_reshaped)
    val_masks_encoded_original_shape = val_masks_reshaped_encoded.reshape(n, h, w)
    print(np.unique(val_masks_encoded_original_shape))

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
    val_masks_input = np.expand_dims(val_masks_encoded_original_shape, axis=3)
    print(train_masks_input.shape)

    X_train = train_img
    X_val = val_img
    X_test = test_img

    y_train = train_masks_input
    y_val = val_masks_input
    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled

    from tensorflow.keras.utils import to_categorical

    n_classes = 2

    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

    val_masks_cat = to_categorical(y_val, num_classes=n_classes)
    y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))

    X_train = np.expand_dims(X_train[:, :, :], axis=3)
    X_val = np.expand_dims(X_val[:, :, :], axis=3)

    #######################################
    # Parameters for model

    # IMG_HEIGHT = X_train.shape[1]
    # IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS)
    input_shape

    sm.set_framework('tf.keras')

    sm.framework()

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([.5, .5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [total_loss, sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    save_results = home_folder + results_folder + trial_folder

    if not os.path.exists(save_results):
        os.makedirs(save_results)

    to_pickle({'data': train_img, 'labels': train_mask}, home_folder + results_folder + trial_folder + training_filename)
    to_pickle({'data': test_img, 'labels': test_mask}, home_folder + results_folder + trial_folder + testing_filename)
    to_pickle({'data': val_img, 'labels': val_mask}, home_folder + results_folder + trial_folder + validation_filename)

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
                                  epochs=NUM_EPOCHS,
                                  callbacks=callbacks_list)

    stop1 = datetime.now()
    # Execution time of the model
    execution_time_Unet = stop1 - start1
    print("UNet execution time is: ", execution_time_Unet)

    unet_model.save(save_results + model_filename)

    # loss, acc = unet_model.evaluate(X_test)
    # print("Accuracy", acc)

    display_learning_curves(unet_history)

    unet_model.load_weights(save_results)
    y_pred1 = unet_model.predict(X_val)
    y_pred1_argmax = np.argmax(y_pred1, axis=3)

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    val_mask = val_mask / 255.0
    IOU_keras.update_state(val_mask, y_pred1_argmax)
    print("Mean IoU on validation data =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)

    # test_img = X_val
    ground_truth = test_mask / 255.0
    test_pred1 = unet_model.predict(X_test)
    test_prediction1 = np.argmax(test_pred1, axis=3)
    IOU_keras.update_state(ground_truth, test_prediction1)
    print("Mean IoU on test data =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)

    display_multiple_img(X_test, ground_truth, test_prediction1, 'unet_plt_1', save_results, n=5)
