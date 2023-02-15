#!/usr/bin/env python
# coding: utf-8

# import tensorflow as tf
# import segmentation_models as sm
# import glob
import cv2
# import os
import numpy as np
import pickle
import random
import matplotlib
import h5py
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
# import keras
import segmentation_models as sm

# from keras.utils import normalize
# from tensorflow.keras.utils import normalize
from keras.metrics import MeanIoU
from keras.models import load_model, save_model

import scipy.io as sio


def to_pickle(thing, path):  # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def display_multiple_img(original, ground_truth, y_test_argmax, plt_name, save_results, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=3, figsize=(8, n * 2))
    c = 0
    for i in range(n):
        image_number = random.randint(0, len(ground_truth) - 1)
        ax.ravel()[c].imshow(original[image_number], cmap='gray')
        ax.ravel()[c].set_title("Hillshade Image: " + str(image_number))
        ax.ravel()[c + 1].imshow(ground_truth[image_number], cmap='gray')
        ax.ravel()[c + 1].set_title("Ground Truth: " + str(image_number))
        ax.ravel()[c + 2].imshow(y_test_argmax[image_number], cmap='gray')
        ax.ravel()[c + 2].set_title("Predicted Image: " + str(image_number))
        c = c + 3

    plt.tight_layout()
    plt.savefig(save_results + "/" + str(plt_name) + ".png")
    plt.show()


def do_inference_on_pickled_data(filename_str, unet_model):
    # training_data = from_pickle(home_folder + results_folder + trial_folder + training_filename)
    testing_data = from_pickle(home_folder + results_folder + trial_folder + testing_filename)
    # validation_data = from_pickle(home_folder + results_folder + trial_folder + validation_filename)

    # Encode labels... but multi dim array so need to flatten, encode and reshape
    # from sklearn.preprocessing import LabelEncoder
    # train_labels = training_data['labels']
    # validate_labels = validation_data['labels']
    test_labels = testing_data['labels']

    # labelencoder = LabelEncoder()
    # n, h, w = train_mask[:, :, :].shape
    # train_masks_reshaped = train_mask[:, :, :].reshape(-1, 1)
    # print(train_masks_reshaped.shape)
    # train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    # train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    # print(np.unique(train_masks_encoded_original_shape))

    # n, h, w = val_mask[:, :, :].shape
    # val_masks_reshaped = val_mask[:, :, :].reshape(-1, 1)
    # val_masks_reshaped_encoded = labelencoder.fit_transform(val_masks_reshaped)
    # val_masks_encoded_original_shape = val_masks_reshaped_encoded.reshape(n, h, w)
    # print(np.unique(val_masks_encoded_original_shape))

    # train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
    # val_masks_input = np.expand_dims(val_masks_encoded_original_shape, axis=3)
    # print(train_masks_input.shape)

    # X_train = training_data['data']
    # X_validate = validation_data['data']
    X_test = testing_data['data']
    # X_train = np.expand_dims(X_train[:, :, :], axis=3)
    # X_validate = np.expand_dims(X_validate[:, :, :], axis=3)
    X_test = np.expand_dims(X_test[:, :, :], axis=3)

    # Y_validate_predicted = unet_model.predict(X_validate)
    # Y_validate_predicted_argmax = np.argmax(Y_validate_predicted, axis=3)

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    # val_mask = validate_labels / 255.0
    # IOU_keras.update_state(validate_labels, Y_validate_predicted_argmax)
    # print("Mean IoU on validation data =", IOU_keras.result().numpy())
    # values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    # print(values)

    # display_multiple_img(X_validate, validate_labels, Y_validate_predicted_argmax, 'validation data', save_results, n=5)

    # test_img = X_val
    # ground_truth = test_labels / 255.0
    Y_test_predicted = unet_model.predict(X_test)
    Y_test_predicted_argmax = np.argmax(Y_test_predicted, axis=3)
    IOU_keras.update_state(test_labels, Y_test_predicted_argmax)
    print("Mean IoU on test data =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)

    display_multiple_img(X_test, test_labels, Y_test_predicted_argmax, 'test data', save_results, n=5)


class HybridDiceAndFocalLoss:
    def __init__(self):
        self.dice_loss = sm.losses.DiceLoss(class_weights=np.array([.5, .5]))
        self.focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = self.dice_loss + (1 * self.focal_loss)
        self._name = 'hybrid_dice_and_focal_loss'

    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    def __call__(self, gt, pr):
        # squared_difference = tf.square(y_true - y_pred)
        # return tf.reduce_mean(squared_difference, axis=-1)
        total_loss = self.dice_loss(gt, pr) + (1 * self.focal_loss(gt, pr))
        return total_loss


if __name__ == "__main__":
    home_folder = '/home/fjannat/Documents/EarthVision/data_resource/'
    home_folder = '/home/arwillis/PyCharm/'
    results_folder = 'results/'
    trial_folder = 'unet/trial/'

    model_filename = 'unet_model.hdf5'
    training_filename = 'data_training.pkl'
    testing_filename = 'data_testing.pkl'
    validation_filename = 'data_validation.pkl'

    metrics = [HybridDiceAndFocalLoss(),
               sm.metrics.IOUScore(threshold=0.5),
               sm.metrics.FScore(threshold=0.5)]
    save_results = home_folder + results_folder + trial_folder
    unet_model = load_model(filepath=save_results + model_filename,
                            custom_objects={"hybrid_dice_and_focal_loss": metrics[0],
                                            "iou_score": metrics[1],
                                            "f1-score": metrics[2]})
    INPUT_LAYER_DIMS = unet_model.input.shape[1:4]

    # data_filename = home_folder + results_folder + trial_folder + testing_filename
    data_filename = home_folder + results_folder + trial_folder + training_filename
    # do_inference_on_pickled_data(data_filename, unet_model)

    gis_data_path = ['data/KOM/', 'data/MLS/', 'data/UCB/', 'data/Sayil/']

    gis_input_filenames_hs = ['kom_dsm_lidar_hs.png',
                              'MLS_DEM_hs.png',
                              'UCB_elev_adjusted_hs.png',
                              'Sayil_regional_DEM_hs.png']

    gis_output_filenames = ['KOM_image_classified.png',
                            'MLS_image_classified.png',
                            'UCB_image_classified.png',
                            'Sayil_image_classified.png']

    gis_input_filenames_mat = ['KOM_image_data.mat',
                               'MLS_image_data.mat',
                               'UCB_image_data.mat',
                               'Sayil_image_data.mat']

    # image_data_hs = []
    # for datasetIdx in range(len(gis_input_filenames_mat)):
    #     img_filename_mat = home_folder + gis_data_path[datasetIdx] + gis_input_filenames_hs[datasetIdx]
    #     image_data_hs_ex = cv2.imread(gis_input_filenames_hs[0])
    #     image_data_hs.append(image_data_hs_ex)

    DATASET_INDEX = 3
    # SHOW_CLASSIFICATIONS = True
    SHOW_CLASSIFICATIONS = False


    img_filename_mat = home_folder + 'data/' + gis_input_filenames_mat[DATASET_INDEX]
    mat_data = []
    if DATASET_INDEX == 3:
        with h5py.File(img_filename_mat, 'r') as f:
            # print(f.keys())
            image_data = np.array(f['geotiff_data'])
    else:
        mat_data = sio.loadmat(img_filename_mat, squeeze_me=True)
        image_data = mat_data['geotiff_data']

    output_filename = home_folder + results_folder + gis_output_filenames[DATASET_INDEX]
    img_filename_hs = home_folder + gis_data_path[DATASET_INDEX] + gis_input_filenames_hs[DATASET_INDEX]
    image_data_hs = cv2.imread(img_filename_hs)

    [rows, cols] = image_data.shape[0:2]
    xy_pixel_skip = (32, 32)
    xy_pixel_margin = np.array([np.round((INPUT_LAYER_DIMS[1] + 1) / 2), np.round((INPUT_LAYER_DIMS[0] + 1) / 2)],
                               dtype=np.int)

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)

    x_vals = range(xy_pixel_margin[0], cols - xy_pixel_margin[0], xy_pixel_skip[0])
    y_vals = range(xy_pixel_margin[1], rows - xy_pixel_margin[1], xy_pixel_skip[1])

    n = 1

    # plt.tight_layout()
    if SHOW_CLASSIFICATIONS:
        figure, ax = plt.subplots(nrows=n, ncols=2, figsize=(8, n * 2))

    output_shape = unet_model.output.get_shape().as_list()
    num_classes = output_shape[3]
    label_image_predicted = np.zeros((image_data.shape[0],image_data.shape[1],int(num_classes)), dtype=np.float32)
    label_image = np.zeros(image_data.shape, dtype=np.float32)
    classification_count_image = np.zeros(image_data.shape, dtype=np.float32)

    for y in y_vals:
        for x in x_vals:
            print("(x,y) = " + "(" + str(x) + ", " + str(y) + ")")
            test_image = image_data[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
                         (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])]
            test_image_hs = image_data_hs[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
                         (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])]
            image_range = (np.max(test_image) - np.min(test_image))
            if image_range == 0:
                image_range = 1
            test_image = (test_image - np.min(test_image)) / image_range
            input_test_image = np.expand_dims(test_image, axis=0)
            input_test_image = np.expand_dims(input_test_image, axis=3)
            test_image_predicted = unet_model.predict(input_test_image)

            label_image_predicted[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
            (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])] += test_image_predicted[0]

            # label_image[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
            # (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])] += test_image_predicted_argmax[0]

            classification_count_image[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
            (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0])] += 1

            if SHOW_CLASSIFICATIONS:
                test_image_predicted_argmax = np.argmax(test_image_predicted, axis=3) / (num_classes - 1)
                # ax.ravel()[0].imshow(test_image, cmap='gray')
                ax.ravel()[0].imshow(test_image_hs, cmap='gray')
                ax.ravel()[0].set_title("Hillshade Image: " + "(" + str(x) + ", " + str(y) + ")")
                ax.ravel()[1].set_title("Predicted Image: " + "(" + str(x) + ", " + str(y) + ")")
                ax.ravel()[1].imshow(test_image_predicted_argmax[0], cmap='gray')
                plt.show(block=False)
                plt.pause(0.1)


    for y in range(0, rows):
        for x in range(0, cols):
            if classification_count_image[y,x] > 0:
                label_image[y,x] = label_image[y,x] / classification_count_image[y,x]
    label_image = np.argmax(label_image_predicted, axis=2) / (num_classes - 1)
    cv2.imwrite(output_filename, np.array(label_image*255, dtype=np.uint8))
    figure, bx = plt.subplots(nrows=n, ncols=2, figsize=(8, n * 2))
    # bx.ravel()[0].imshow(image_data, cmap='gray')
    bx.ravel()[0].imshow(image_data_hs, cmap='gray')
    bx.ravel()[0].set_title("Hillshade Image")
    bx.ravel()[1].set_title("Predicted Image")
    bx.ravel()[1].imshow(label_image, cmap='gray')
    plt.show()
