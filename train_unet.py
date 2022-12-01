#!/usr/bin/env python
# coding: utf-8
import warnings
import keras

warnings.filterwarnings('ignore')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import pickle
import random
import tensorflow as tf
from keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from att_models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef

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
    theta = (angle / 180) * np.pi
    xy_center = (width / 2, height / 2)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    bound_w = int(height * np.abs(sin_t) + width * np.abs(cos_t))
    bound_h = int(height * np.abs(cos_t) + width * np.abs(sin_t))
    xy_start = np.array([np.floor(center_x - bound_w / 2), np.floor(center_y - bound_h / 2)], dtype=np.int32)
    xy_end = np.array([np.ceil(center_x + bound_w / 2), np.ceil(center_y + bound_h / 2)], dtype=np.int32)
    if np.any(xy_start < 0) or xy_start[0] > img.shape[1] or xy_start[1] > img.shape[0] or \
            np.any(xy_end < 0) or xy_end[0] > img.shape[1] or xy_end[1] > img.shape[0]:
        # print("Could not extract patch at location (" + str((center_x, center_y)) + ")")
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
    xy_start = np.array([xy_center_newimg[0] - width / 2, xy_center_newimg[1] - height / 2], dtype=np.int32)
    xy_end = np.array([xy_center_newimg[0] + width / 2, xy_center_newimg[1] + height / 2], dtype=np.int32)
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
            WINDOWSIZE = 40
            if np.count_nonzero(single_patch_mask > 0.0) == WINDOWSIZE * WINDOWSIZE:
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


def to_pickle(thing, path):  # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


# add sample images visualization to keras callbacks
class SaveImage(keras.callbacks.Callback):
    def __init__(self, train_images, val_images, generator, logdir):
        super(SaveImage, self).__init__()
        self.tr_images = train_images
        self.val_images = val_images
        self.g = generator
        self.writer = tf.summary.create_file_writer((logdir + '/vis'))  # Creates writer element to write the data for Tensorboard

    # Images will be generated after each epoch
    def on_epoch_end(self, epoch, logs=None):
        # Function to generate batch of images. Returns a list (l) with the 3-image display (input, generated, ground truth)
        def generate_imgs(g, imgs):
            l = []
            for i in range(len(imgs)):
                x = imgs[0][i]  # data
                y = imgs[1][i]  # label
                # Select only the first image of the batch -> None keeps the batch dimension so that the generator doesn't raise an exception
                x = x[None, ...]
                y = y[None, ...]
                out = g(x)
                out = np.argmax(out, axis=3)
                out = out[..., None]

                # Concatenate vertically input (x), output (out) and ground truth (y) to display the 3 images
                cat_image = np.concatenate((x.squeeze(axis=0), out.squeeze(axis=0), y.squeeze(axis=0)), axis=1)  # np.squeeze deletes the batch dimension
                #cat_image_uint8 = (cat_image * 255).astype('uint8')
                l.append(cat_image)
            return l

        # Generate images for training and validation images
        train_sample = generate_imgs(self.g, self.tr_images)
        val_sample = generate_imgs(self.g, self.val_images)

        # Write (store) the images within the writer element
        with self.writer.as_default():
            with tf.name_scope("train") as scope:  # tf.name_scope adds the prefix train/ to all the tf.summary.image names
                tf.summary.image(step=epoch, data=train_sample, name=scope, max_outputs=len(train_sample))
            with tf.name_scope("val") as scope:
                tf.summary.image(step=epoch, data=val_sample, name=scope, max_outputs=len(val_sample))


if __name__ == "__main__":

    # Run the following command to track training in a browser
    #
    #  tensorboard --logdir logs --reload_multifile=true
    #
    BATCH_SIZE = 100
    NUM_EPOCHS = 500
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
    # home_folder = './'
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
    gis_input_filenames_mat = ['KOM_image_data.mat',
                               'MLS_image_data.mat',
                               'UCB_image_data.mat']
    gis_input_gt_filenames_mat = ['KOM_ground_truth_labels.mat',
                                  'MLS_ground_truth_labels.mat',
                                  'UCB_ground_truth_labels.mat']
    gis_input_gt_filenames = ['kom_dsm_lidar_gt.png',
                              'MLS_DEM_gt.png',
                              'UCB_elev_adjusted_gt.png']

    img_filename1 = home_folder + gis_data_path[0] + gis_input_filenames[0]
    mask_filename1 = home_folder + gis_data_path[0] + gis_input_gt_filenames[0]

    img_filename2 = home_folder + gis_data_path[1] + gis_input_filenames[1]
    mask_filename2 = home_folder + gis_data_path[1] + gis_input_gt_filenames[1]

    img_filename3 = home_folder + gis_data_path[2] + gis_input_filenames[2]
    mask_filename3 = home_folder + gis_data_path[2] + gis_input_gt_filenames[2]

    if (False):
        image_1 = cv2.imread(img_filename1)
        mask_1 = cv2.imread(mask_filename1)[:, :, 0:1]
        image_2 = cv2.imread(img_filename2)
        mask_2 = cv2.imread(mask_filename2)[:, :, 0:1]
        image_3 = cv2.imread(img_filename3)
        mask_3 = cv2.imread(mask_filename3)[:, :, 0:1]

        datasets = {'data': [], 'labels': [], 'region_centroids': [], 'num_regions': [], 'analysis': []}
        datasets['data'] = [image_1, image_2, image_3]
        datasets['labels'] = [mask_1, mask_2, mask_3]
        num_datasets = len(datasets['data'])

        # for each dataset compute the connected components to recover the labeled regions of image data
        # store the result of the connected component analysis
        for datasetIdx in range(num_datasets):
            analysis = cv2.connectedComponentsWithStats(datasets['labels'][datasetIdx], cv2.CV_32S)
            (totalLabels, label_img, regionStats, regionCentroids) = analysis
            datasets['num_regions'].append(regionStats.shape[0])
            datasets['region_centroids'].append(regionCentroids)
            datasets['analysis'].append(analysis)
    else:
        import scipy.io as sio

        image_data = []
        for filenameIdx in range(len(gis_input_filenames_mat)):
            mat_data = {}
            img_filename_mat = home_folder + 'data/' + gis_input_filenames_mat[filenameIdx]
            mat_data = sio.loadmat(img_filename_mat, squeeze_me=True)
            image_data.append(mat_data['geotiff_data'])

        image_labels = []
        for filenameIdx in range(len(gis_input_gt_filenames_mat)):
            mat_data = {}
            img_gt_filename_mat = home_folder + 'data/' + gis_input_gt_filenames_mat[filenameIdx]
            mat_data = sio.loadmat(img_gt_filename_mat, squeeze_me=True)
            image_labels.append(mat_data['all_labels'])

        datasets = {'data': [], 'labels': [], 'region_centroids': [], 'num_regions': [], 'analysis': []}
        for datasetIdx in range(len(image_data)):
            datasets['data'].append(image_data[datasetIdx])

        for datasetIdx in range(len(image_data)):
            labelArr = datasets['labels']
            for labelIdx in range(len(image_labels[0]['labels'])):
                regions = []
                for regionIdx in range(len(image_labels[0]['labels'][labelIdx])):
                    region_data = {'ID': 0, 'centroid': [], 'vertices': []}
                    region_data['vertices'] = image_labels[0]['labels'][labelIdx][regionIdx]['vertices'].all()
                    region_data['centroid'] = np.mean(region_data['vertices'], axis=0)
                    region_data['ID'] = image_labels[0]['labels'][labelIdx][regionIdx]['ID'].all()
                    regions.append(region_data)
                if len(regions) > 0:
                    labelArr.append(regions)

        num_datasets = len(datasets['data'])

    # this will store all of our data for all datasets and their components which consist of the data split into
    # training, validation and testing sets
    augmentation_data = []

    # for each dataset setup a dictionary data structure to create test, train and validate components of the dataset
    # each component will have a list of indices indicating the index of regions/segmented parts of the original image
    # that will be used in each of the dataset components. Subsequent code will populate the 'region centroids',
    # image data ('data'), and label data ('labels') within this datastructure for training.
    for datasetIdx in range(num_datasets):
        dataset_components = {'train': {}, 'test': {}, 'validate': {}}
        augmentation_data.append(dataset_components)
        for dataset_component in dataset_components:
            dataset_attrs = {'indices': [], 'num_region_idxs': 0,
                             'region_centroids': [], 'num_regions': 0,
                             'data': [], 'labels': []}
            dataset_components[dataset_component] = dataset_attrs

    # for each dataset select the regions/data vectors to put into the training, validation and testing sets
    # by storing the indices of these regions and the number of indices/regions within each of these sets
    from sklearn.model_selection import train_test_split
    for datasetIdx in range(num_datasets):
        dataset_components = augmentation_data[datasetIdx]
        datavectorIndices = list(range(0, datasets['num_regions'][datasetIdx]))
        training_indices, testing_indices = train_test_split(datavectorIndices, test_size=pct_test, random_state=1)
        training_indices, validation_indices = train_test_split(training_indices, test_size=pct_val, random_state=1)
        dataset_components['train']['indices'] = training_indices
        dataset_components['test']['indices'] = testing_indices
        dataset_components['validate']['indices'] = validation_indices
        dataset_components['train']['num_region_idxs'] = len(training_indices)
        dataset_components['test']['num_region_idxs'] = len(testing_indices)
        dataset_components['validate']['num_region_idxs'] = len(validation_indices)

    # for each dataset visit the training, validation and testing sets and set the region centroids for all the regions
    # that are associated with each of these sets
    for datasetIdx in range(num_datasets):
        dataset_components = augmentation_data[datasetIdx]
        for component in dataset_components:
            dataset = dataset_components[component]
            regionCentroidArray = dataset['region_centroids']
            for localRegionIndex in range(dataset['num_region_idxs']):
                globalRegionIndex = dataset['indices'][localRegionIndex]
                (totalLabels, label_img, regionStats, regionCentroids) = datasets['analysis'][datasetIdx]
                # ensure that the connected component corresponds to a specific size/area region
                if regionStats[globalRegionIndex][2] == 40 and regionStats[globalRegionIndex][3] == 40:
                    # if regionStats[globalRegionIndex][2] * regionStats[globalRegionIndex][3] > 900:
                    regionCentroidArray.append(datasets['region_centroids'][datasetIdx][globalRegionIndex])
            dataset['num_regions'] = len(regionCentroidArray)

    if SHOW_AUGMENTATION:
        figure, handle = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))

    for datasetIdx in range(num_datasets):
        dataset_components = augmentation_data[datasetIdx]
        dataset_testRegionCentroidArray = dataset_components['test']['region_centroids']
        dataset_numTestRegions = len(dataset_testRegionCentroidArray)
        for dataset in dataset_components:
            dataArray = dataset_components[dataset]['data']
            labelArray = dataset_components[dataset]['labels']
            for regionIndex in range(dataset_components[dataset]['num_regions']):
                for augmentationIdx in range(NUM_AUGMENTATIONS_PER_IMAGE):
                    # randomly perturb the orientation
                    angle = np.random.uniform(low=0, high=359.9)
                    center_y, center_x = dataset_components[dataset]['region_centroids'][regionIndex]
                    # randomly perturb the offset of the region within the tile
                    dx = np.random.uniform(low=-30, high=30)
                    dy = np.random.uniform(low=-30, high=30)
                    aug_image_center = np.array([center_x + dx, center_y + dy], dtype=np.float32)
                    skip_this_image = False

                    # do not test for proximity when augmenting the testing dataset
                    if dataset != 'test':
                        for testRegionIdx in range(dataset_numTestRegions):
                            aug_image_center_to_testRegion_vector = dataset_testRegionCentroidArray[testRegionIdx] - aug_image_center
                            train_to_test_Distance = np.linalg.norm(aug_image_center_to_testRegion_vector)
                            if train_to_test_Distance < 1.5 * IMAGE_SIZE * np.sqrt(2) + 21:
                                print("Skip augmentation at image center (" + str(aug_image_center_to_testRegion_vector) +")"
                                      " distance to test set = " + str(train_to_test_Distance))
                                skip_this_image = True
                                continue

                    # if the augmentation may include data from the test set skip this augmentation
                    # this may occur when labels of the test set are in the vicinity of labels from the training or
                    # validation set
                    if skip_this_image:
                        continue

                    aug_image_patch = getRigidImagePatch(datasets['data'][datasetIdx],
                                                         IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle)
                    aug_mask_patch = getRigidImagePatch(datasets['labels'][datasetIdx],
                                                        IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle)

                    # if the augmentation was successful add it to the image augmentation dataset
                    if aug_image_patch is not None:
                        aug_image_patch = np.array(aug_image_patch[:, :, 1], dtype=np.float32) / 255.0
                        dataArray.append(aug_image_patch)
                        labelArray.append(aug_mask_patch)
                        if SHOW_AUGMENTATION:
                            handle[0].imshow(aug_image_patch, cmap='gray')
                            handle[1].imshow(aug_mask_patch, cmap='gray')
                            plt.pause(0.5)

    # form the training, testing and validation datasets from available labeled image data
    train_images = np.concatenate((augmentation_data[0]['train']['data'],
                                   augmentation_data[1]['train']['data'],
                                   augmentation_data[2]['train']['data']))
    train_labels = np.concatenate((augmentation_data[0]['train']['labels'],
                                   augmentation_data[1]['train']['labels'],
                                   augmentation_data[2]['train']['labels']))

    validate_images = np.concatenate((augmentation_data[0]['validate']['data'],
                                      augmentation_data[1]['validate']['data'],
                                      augmentation_data[2]['validate']['data']))
    validate_labels = np.concatenate((augmentation_data[0]['validate']['labels'],
                                      augmentation_data[1]['validate']['labels'],
                                      augmentation_data[2]['validate']['labels']))

    test_images = np.concatenate((augmentation_data[0]['test']['data'],
                                  augmentation_data[1]['test']['data'],
                                  augmentation_data[2]['test']['data']))
    test_labels = np.concatenate((augmentation_data[0]['test']['labels'],
                                  augmentation_data[1]['test']['labels'],
                                  augmentation_data[2]['test']['labels']))

    # print out the number of training vectors, validation vectors, and test vectors
    print("Train " + str(train_images.shape[0]))
    print("Validation " + str(validate_images.shape[0]))
    print("Test " + str(test_images.shape[0]))

    # show a collection of images from the training set
    get_sample_display_multiple_img(train_images, train_labels, n=5)

    # setup input and output images for the neural net
    X_train = train_images
    X_validate = validate_images
    X_test = test_images

    Y_train = train_labels
    Y_validate = validate_labels
    Y_test = test_labels
    print("Class values in the dataset are ... ", np.unique(Y_train))  # 0 is the background/few unlabeled

    # add component/channel dimension to data and label images
    X_train = np.expand_dims(X_train, axis=3)
    X_validate = np.expand_dims(X_validate, axis=3)
    Y_train = np.expand_dims(Y_train, axis=3)
    Y_validate = np.expand_dims(Y_validate, axis=3)

    # convert label images to categorical format for training
    from tensorflow.keras.utils import to_categorical
    n_classes = len(np.unique(Y_train))
    Y_train_cat = to_categorical(Y_train, num_classes=n_classes)
    Y_validate_cat = to_categorical(Y_validate, num_classes=n_classes)

    # Set input layer dimensions
    IMG_CHANNELS = X_train.shape[3]
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS)
    sm.set_framework('tf.keras')
    sm.framework()

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([.5, .5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [total_loss, sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    save_results = home_folder + results_folder + trial_folder

    if not os.path.exists(save_results):
        os.makedirs(save_results)

    to_pickle({'data': train_images, 'labels': train_labels}, home_folder + results_folder + trial_folder + training_filename)
    to_pickle({'data': test_images, 'labels': test_labels}, home_folder + results_folder + trial_folder + testing_filename)
    to_pickle({'data': validate_images, 'labels': validate_labels}, home_folder + results_folder + trial_folder + validation_filename)

    checkpoint = ModelCheckpoint(save_results, monitor="val_iou_score", verbose=1, save_best_only=True, mode="max")
    early_stopping = EarlyStopping(monitor="val_iou_score", patience=150, verbose=1, mode="max")

    # create list of callbacks
    callbacks_list = [checkpoint, early_stopping]  # early_stopping

    """
    UNet
    """
    unet_model = UNet(input_shape, NUM_CLASSES=2)
    unet_model.compile(optimizer=Adam(lr=1e-2),
                       loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.1),
                       metrics=metrics)

    # print(unet_model.summary())
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False,
                                                          write_images=False,
                                                          write_steps_per_second=True)

    # Initialize the SaveImage class by passing the arguments to the __init__() function
    save_image_call = SaveImage(
        # SaveImage will only evaluate 4 images from training and validation sets
        (X_train.take([0, 10], axis=0), Y_train.take([0, 10], axis=0)),
        (X_validate.take([0, 10], axis=0), Y_validate.take([0, 10], axis=0)),
        unet_model,  # generator
        log_dir)

    start1 = datetime.now()
    VALIDATION_BATCH_SIZE = 32
    validation_steps = X_validate.shape[0] / VALIDATION_BATCH_SIZE
    unet_history = unet_model.fit(X_train,
                                  Y_train_cat,
                                  verbose=1,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  validation_data=(X_validate, Y_validate_cat),
                                  validation_steps=validation_steps,
                                  validation_batch_size=VALIDATION_BATCH_SIZE,
                                  epochs=NUM_EPOCHS,
                                  callbacks=[save_image_call, tensorboard_callback])

    stop1 = datetime.now()
    # Execution time of the model
    execution_time_Unet = stop1 - start1
    print("UNet execution time is: ", execution_time_Unet)

    unet_model.save(save_results + model_filename)

    # loss, acc = unet_model.evaluate(X_test)
    # print("Accuracy", acc)

    unet_model.load_weights(save_results)
    Y_validate_predicted = unet_model.predict(X_validate)
    Y_validate_predicted_argmax = np.argmax(Y_validate_predicted, axis=3)

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(validate_labels, Y_validate_predicted_argmax)
    print("Mean IoU on validation data =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)

    Y_test_predicted = unet_model.predict(X_test)
    Y_test_predicted_argmax = np.argmax(Y_test_predicted, axis=3)
    IOU_keras.update_state(test_labels, Y_test_predicted_argmax)
    print("Mean IoU on test data =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)

    display_multiple_img(X_test, test_labels, Y_test_predicted_argmax, 'unet_plt_1', save_results, n=5)
