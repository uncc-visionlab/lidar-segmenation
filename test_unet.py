#!/usr/bin/env python
# coding: utf-8

# import tensorflow as tf
# import segmentation_models as sm
# import glob
# import cv2
# import os
import numpy as np
import pickle
import random
from matplotlib import pyplot as plt
# import keras

# from keras.utils import normalize
# from tensorflow.keras.utils import normalize
from keras.metrics import MeanIoU
from keras.models import load_model

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def display_multiple_img(original, ground_truth, y_test_argmax, plt_name, save_results, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=3, figsize=(8, n * 2))
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

if __name__ == "__main__":
    home_folder = '/home/fjannat/Documents/EarthVision/data_resource/'
    home_folder = '/home/arwillis/PyCharm/'
    results_folder = 'results/'
    trial_folder = 'unet/trial'
    model_filename = '/as_unet.hdf5'
    training_filename = '/data_training.pkl'
    testing_filename = '/data_testing.pkl'
    validation_filename = '/data_validation.pkl'

    training_data = from_pickle(home_folder + results_folder + trial_folder + training_filename)
    testing_data = from_pickle(home_folder + results_folder + trial_folder + testing_filename)
    validation_data = from_pickle(home_folder + results_folder + trial_folder + validation_filename)

    save_results = home_folder + results_folder + trial_folder

    # Encode labels... but multi dim array so need to flatten, encode and reshape
    # from sklearn.preprocessing import LabelEncoder
    train_mask = training_data['labels']
    val_mask = validation_data['labels']
    test_mask = testing_data['labels']

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

    X_train = training_data['data']
    X_val = validation_data['data']
    X_test = testing_data['data']
    X_train = np.expand_dims(X_train[:, :, :], axis=3)
    X_val = np.expand_dims(X_val[:, :, :], axis=3)

    # Set compile=False as we are not loading it for training, only for prediction.
    unet_model = load_model(home_folder+results_folder+trial_folder+model_filename, compile=False)

    # unet_model.load_weights(save_results)
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
    ground_truth = test_mask[:, :, :] / 255.0
    test_pred1 = unet_model.predict(X_test)
    test_prediction1 = np.argmax(test_pred1, axis=3)
    IOU_keras.update_state(ground_truth, test_prediction1)
    print("Mean IoU on test data =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)

    display_multiple_img(X_test, ground_truth, test_prediction1, 'unet_plt_1', save_results, n=5)
