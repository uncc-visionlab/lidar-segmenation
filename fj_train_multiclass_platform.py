import warnings
import keras
import numpy
import tensorflow
from torch.optim import SGD
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
warnings.filterwarnings('ignore')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import pickle
import random
import tensorflow as tf

from keras.metrics import MeanIoU

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model, save_model


from datetime import datetime
from att_models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef
from hrnet_model import hrnet_keras
import scipy.io as sio
from keras_unet_collection import models
import random
from tensorflow.keras.metrics import MeanIoU

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
    #plt.show()

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
    cropped_image_patch = img[xy_start[1]:xy_end[1], xy_start[0]:xy_end[0], :]
    cropped_height = cropped_image_patch.shape[0]
    cropped_width = cropped_image_patch.shape[1]
    #if cropped_height != height or cropped_width != width:
    #    return None
    # xy_rotation_centerpt = np.array([width / 2, height / 2])
    xy_translation = np.array([0.5 * (cropped_width - (cos_t * cropped_width + sin_t * cropped_height)),
                               0.5 * (cropped_height - (-sin_t * cropped_width + cos_t * cropped_height))])
    image_patch_T = np.float32([[cos_t, sin_t, xy_translation[0]], [-sin_t, cos_t, xy_translation[1]]])
    transformed_image_patch = cv2.warpAffine(cropped_image_patch, image_patch_T, (cropped_width, cropped_height),
                                             flags=cv2.INTER_NEAREST)

    xy_center_newimg = np.int32(np.array(transformed_image_patch.shape[:2]) / 2.0)
    xy_start = np.array([xy_center_newimg[0] - width / 2, xy_center_newimg[1] - height / 2], dtype=np.int32)
    xy_end = np.array([xy_center_newimg[0] + width / 2, xy_center_newimg[1] + height / 2], dtype=np.int32)
    image_patch_aug = transformed_image_patch[xy_start[1]:xy_end[1], xy_start[0]:xy_end[0]]

    return image_patch_aug

def to_pickle(thing, path):  # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

def display_multiple_img_sanity_test(image_hs, mask, label, n=5):
    figure, ax = plt.subplots(nrows=n, ncols=2, figsize=(12, n * 4))
    c = 0
    j = 1
    i=0
    while i<5:
        image_number = random.randint(0, len(mask) - 1)
        l = np.unique(mask[image_number])
        if label in l and len(l)<=2:
            ax.ravel()[c].imshow(image_hs[image_number], cmap='gray')
            ax.ravel()[c].set_title("Hillshade Image: " + str(image_number))
            ax.ravel()[c + 1].imshow(mask[image_number], cmap='gray')
            ax.ravel()[c + 1].set_title("Mask: " + str(image_number))
            c = c + 2
            j = j + 1
            i+=1

    plt.tight_layout()
    plt.show()
def map_array_to_range(arr):
    unique_vals = np.unique(arr)
    mapping_dict = {val: i for i, val in enumerate(unique_vals)}
    mapped_arr = np.vectorize(mapping_dict.get)(arr)
    return mapped_arr
def split_dict_by_ratio(input_dict, ratio1, ratio2):
    # initialize the three new dictionaries and lists
    dict1 = {}
    dict2 = {}
    dict3 = {}
    list1 = []
    list2 = []
    list3 = []

    # loop over each key in the input dictionary
    for key, values in input_dict.items():
        # calculate the lengths of each split
        length = len(values)
        split1_length = max(1, int(length * ratio1))
        split2_length = max(1, int(length * ratio2))
        split3_length = length - split1_length - split2_length

        # shuffle the values and assign them to one of the three dictionaries
        random.shuffle(values)
        dict1[key] = values[:split1_length]
        dict2[key] = values[split1_length:split1_length + split2_length]
        dict3[key] = values[split1_length + split2_length:]

        # add the values to the appropriate list
        list1.extend(dict1[key])
        list2.extend(dict2[key])
        list3.extend(dict3[key])

    # return the three new dictionaries and lists
    return dict1, dict2, dict3, list1, list2, list3
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
            for i in range(len(imgs[0])):
                x = imgs[0][i]  # data
                y = imgs[1][i]  # label
                # Select only the first image of the batch -> None keeps the batch dimension so that the generator doesn't raise an exception
                x = x[None, ...]
                x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))

                out = g(x)
                num_classes = out.shape[3]
                out = np.argmax(out, axis=3) / (num_classes - 1)
                out = out[..., None]

                y = y[None, ...] / (num_classes - 1)

                # Concatenate vertically input (x), output (out) and ground truth (y) to display the 3 images
                cat_image = np.concatenate((x_norm.squeeze(axis=0), out.squeeze(axis=0), y.squeeze(axis=0)), axis=1)  # np.squeeze deletes the batch dimension
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

class HybridDiceAndFocalLoss:
    def __init__(self):
        self.dice_loss = sm.losses.DiceLoss(class_weights=np.array([.5, .5]))
        self.focal_loss = sm.losses.CategoricalFocalLoss()
        self._name = 'hybrid_dice_and_focal_loss'
    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name
    def __call__(self, gt, pr):
        # squared_difference = tf.square(y_true - y_pred)
        # return tf.reduce_mean(squared_dif ference, axis=-1)
        total_loss = self.dice_loss(gt, pr) + (1 * self.focal_loss(gt, pr))
        return total_loss

def calculate_individual_iou(iou_array, class_num):
    intersection = iou_array[class_num, class_num]
    union = np.sum(iou_array[class_num, :]) + np.sum(iou_array[:, class_num]) - intersection
    iou = intersection / union
    return iou


from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x
def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model




from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda

def multi_unet_model(n_classes=16, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
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

    # Expansive path
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
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    return model
def get_model():
        return multi_unet_model(n_classes=16, IMG_HEIGHT=IMAGE_SIZE, IMG_WIDTH=IMAGE_SIZE, IMG_CHANNELS=IMG_CHANNELS)


def remove_zero_images(x, y):
    """
    Remove images from x and y if any image from x does not contain any value except 0.

    Args:
        x (np.ndarray): Array of shape (n, 128, 128) containing images.
        y (np.ndarray): Array of shape (n,) containing labels.

    Returns:
        Tuple of two np.ndarrays: The filtered x and y arrays.
    """
    # Find the indices of the images that don't contain any value except 0
    zero_images = np.where(np.all(x == 0, axis=(1, 2)))[0]

    # Remove the zero images from x and y
    x_filtered = np.delete(x, zero_images, axis=0)
    y_filtered = np.delete(y, zero_images, axis=0)

    return x_filtered, y_filtered


################################################################
if __name__ == "__main__":

    # Run the following command to track training in a browser
    #
    #  tensorboard --logdir logs --reload_multifile=true
    #
    # Run the following command to track training in a browser
    #
    #  tensorboard --logdir logs --reload_multifile=true
    #
    BATCH_SIZE = 100
    NUM_EPOCHS = 200
    IMAGE_SIZE = 128
    # Image augmentation settings
    NUM_AUGMENTATIONS_PER_LABELED_REGION = 50
    NUM_RANDOM_AUGMENTATIONS = 20000
    #SHOW_AUGMENTATION = True
    SHOW_AUGMENTATION = False
    n_classes = 4
    # split the data within each image test/train
    # test 20%
    # merge all the train and split the training into train/validate
    # train 65%
    # validate 15%
    pct_test = 0.2
    pct_val = 0.2

    home_folder = '/home/fjannat/Documents/Lidar_seg/'
    # home_folder = '/home/arwillis/PyCharm/'
    home_folder = '../'
    results_folder = 'results/'
    trial_folder = 'unet_multi_platform/test_4_F_1/'
    model_filename = 'unet_model.hdf5'
    training_filename = 'data_training.pkl'
    testing_filename = 'data_testing.pkl'
    validation_filename = 'data_validation.pkl'

    gis_data_path = ['data/MLS/', 'data/KOM/', 'data/UCB/']
    gis_input_filenames_hs = [  # 'kom_dsm_lidar_hs.png',
        'MLS_DEM_hs.png',
        # 'UCB_elev_adjusted_hs.png'
    ]
    gis_input_filenames_mat = [  # 'KOM_image_data.mat',
        'MLS_multi_platform/MLS_image_data.mat',
        # 'UCB_image_data.mat'
    ]
    gis_input_gt_filenames_mat = [  # 'KOM_ground_truth_labels.mat',
        'MLS_multi_platform/MLS_ground_truth_labels.mat',
        # 'UCB_ground_truth_labels.mat'
    ]

    image_data_hs = []
    for filenameIdx in range(len(gis_input_filenames_hs)):
        img_filename_hs = home_folder + gis_data_path[filenameIdx] + gis_input_filenames_hs[filenameIdx]
        image_data_hs_ex = cv2.imread(img_filename_hs)
        image_data_hs.append(image_data_hs_ex)

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
        mat_data = sio.loadmat(img_gt_filename_mat)
        image_labels.append(mat_data['all_labels'])

    datasets = {'data': [], 'data_hs': [], 'labels': [], 'region_centroids': [], 'num_regions': [], 'analysis': []}
    for datasetIdx in range(len(image_data)):
        # image = np.zeros((np.array(image_data[datasetIdx]).shape[0], np.array(image_data[datasetIdx]).shape[1], 3))

        image = image_data[datasetIdx][:, :, None]
        datasets['data'].append(image)
        image_hs = image_data_hs[datasetIdx]
        datasets['data_hs'].append(image_hs)

    for datasetIdx in range(len(image_data)):
        # labelArr = datasets['labels']
        # for labelIdx in range(len(image_labels[datasetIdx])):   # A
        # image_shape = image_data[datasetIdx].shape
        # mask = np.zeros(image_shape)
        for labelIdx in range(1):  # A   #1,2
            regions = []

            # print("label " + str(labelIdx))

            for regionIdx in range(len(image_labels[datasetIdx][labelIdx][0][0][0])):  # B
                region_data = {'label_value': image_labels[datasetIdx][labelIdx][0][0]['label_value'][0][regionIdx],
                               # 'centroid': image_labels[datasetIdx].item()[labelIdx][0][regionIdx]['center'],
                               'centroid': np.mean(image_labels[datasetIdx][labelIdx][0][0]['vertices'][0][regionIdx],
                                                   0),
                               'vertices': image_labels[datasetIdx][labelIdx][0][0]['vertices'][0][regionIdx],
                               'ID': image_labels[datasetIdx][labelIdx][0][0]['ID'][0][regionIdx]}
                regions.append(region_data)
                # print("region " + str(regionIdx))
            #    if len(regions) > 0:
            #        labelArr.append(regions)

            datasets['region_centroids'].append(np.asarray([region_data['centroid'] for region_data in regions]))
            datasets['num_regions'].append(len(datasets['region_centroids'][datasetIdx]))
            image_shape = image_data[datasetIdx].shape
            mask = np.zeros(image_shape)

            keep_count = {}
            count_label = 0
            count_total = 0
            keep_regions = {}
            for i in range(len(regions)):
                val = int(regions[i]['label_value'])
                if val in keep_count:
                    keep_count[val] = keep_count[val] + 1
                    keep_regions[val].append(i)
                else:
                    keep_count[val] = 1

                    keep_regions[val] = [i]
                    count_label = count_label + 1

                count_total = count_total + 1
            print(keep_count)
            print(count_total)
            for i in range(len(regions)):
                n = int(regions[i]['label_value'])
                cv2.fillPoly(mask, np.int32([regions[i]['vertices']]), (n, n, n))

            datasets['labels'].append(mask.astype(np.uint8)[:, :, None])
            analysis = cv2.connectedComponentsWithStats(datasets['labels'][datasetIdx], cv2.CV_32S)
            (totalLabels, label_img, regionStats, regionCentroids) = analysis
            datasets['analysis'].append(analysis)

    # Encode labels... but multi dim array so need to flatten, encode and reshape
    from sklearn.preprocessing import LabelEncoder

    labelencoder = LabelEncoder()
    masks = datasets['labels'][0]
    n, h, w = masks.shape
    masks_reshaped = masks.reshape(-1, 1)
    masks_reshaped_encoded = labelencoder.fit_transform(masks_reshaped)
    mask_new_encoded = masks_reshaped_encoded.reshape(n, h, w)

    print(np.unique(mask_new_encoded))

    mask_new = np.where(np.isin(mask_new_encoded, [1,2,4,5,6,7,8,9,10,11]), mask_new_encoded, 0)
    mask_new = np.where(np.isin(mask_new, [1,2, 0]), mask_new, 3)
    #mask_new = np.where((mask_new!=0), 1, 0)
    print(np.unique(mask_new))

    n, h, w = mask_new.shape
    mask_new_encoded = mask_new.reshape(-1, 1)
    mask_new_encoded = labelencoder.fit_transform(mask_new_encoded)
    mask_new_encoded = mask_new_encoded.reshape(n, h, w)

    print(np.unique(mask_new_encoded))

    datasets['labels'][0] = mask_new_encoded

    #datasets['labels'][0] = map_array_to_range(datasets['labels'][0])
    # from sklearn.utils.class_weight import compute_class_weight
    #
    # class_weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=np.unique(np.ravel(mask_new_encoded,order='C')),
    #     y=np.ravel(mask_new_encoded,order='C')
    # )
    #class_weights = dict(zip(np.unique(masks_encoded_original_shape), class_weights))
    #print("Class weights are...:", class_weights)

    dict1, dict2, dict3, training_indices, testing_indices, validation_indices = split_dict_by_ratio(keep_regions, 0.70,
                                                                                                     0.15)
    print("Dict for Training Indices")
    print(dict1)
    print(" ")
    print("Dict for Testing Indices")
    print(dict2)
    print(" ")
    print("Dict for Validation Indices")
    print(dict3)
    print(" ")
    print("List of Training Indices")
    print(training_indices)
    print(" ")
    print("List of Testing Indices")
    print(testing_indices)
    print(" ")
    print("List of Validaition Indices")
    print(validation_indices)

    print("Labels in Dict1: " + str(sorted(dict1.keys())))
    print("Labels in Dict2: " + str(sorted(dict2.keys())))
    print("Labels in Dict3: " + str(sorted(dict3.keys())))

    print("Number of Training Indices: " + str(len(training_indices)))
    print("Number of Validation Indices: " + str(len(validation_indices)))
    print("Number of Testing Indices: " + str(len(testing_indices)))

    num_datasets = len(datasets['data'])
    print(num_datasets)
    # n_class = 0
    # for i in range(len(datasets['labels'])):
    #     n = max(np.unique(datasets['labels'][i]))
    #     if n > n_class:
    #         n_class = n
    # print("Num of Class: " + str(n_class))

    #############################################
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
                             'data': [], 'labels': [], 'data_hs': []}
            dataset_components[dataset_component] = dataset_attrs
    # for each dataset select the regions/data vectors to put into the training, validation and testing sets
    # by storing the indices of these regions and the number of indices/regions within each of these sets
    from sklearn.model_selection import train_test_split

    for datasetIdx in range(num_datasets):
        dataset_components = augmentation_data[datasetIdx]
        #     datavectorIndices = list(range(0, datasets['num_regions'][datasetIdx]))
        #     training_indices, testing_indices = train_test_split(datavectorIndices, test_size=pct_test, random_state=1)
        #     training_indices, validation_indices = train_test_split(training_indices, test_size=pct_val, random_state=1)

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
                regionCentroidArray.append(datasets['region_centroids'][datasetIdx][globalRegionIndex])
            dataset['num_regions'] = len(regionCentroidArray)

    if SHOW_AUGMENTATION:
        figure, handle = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    ########################
    # iterate through the available datasets
    for datasetIdx in range(num_datasets):
        dataset_components = augmentation_data[datasetIdx]
        dataset_testRegionCentroidArray = dataset_components['test']['region_centroids']
        dataset_numTestRegions = len(dataset_testRegionCentroidArray)

        #  iterate through test, train and validate components of dataset
        for dataset in dataset_components:
            dataArray = dataset_components[dataset]['data']
            labelArray = dataset_components[dataset]['labels']
            dataHsArray = dataset_components[dataset]['data_hs']

            [rows, cols] = datasets['data'][datasetIdx].shape[0:2]

            for augmentationIdx in range(NUM_RANDOM_AUGMENTATIONS):
                angle = np.random.uniform(low=0, high=359.9)
                center_y, center_x = (np.random.uniform(low=0, high=rows), np.random.uniform(low=0, high=cols))
                aug_image_center = np.array([center_x, center_y], dtype=np.float32)

                skip_this_image = False
                # do not test for proximity when augmenting the testing dataset
                if dataset != 'test':
                    for testRegionIdx in range(dataset_numTestRegions):
                        aug_image_center_to_testRegion_vector = dataset_testRegionCentroidArray[
                                                                    testRegionIdx] - aug_image_center
                        train_to_test_Distance = np.linalg.norm(aug_image_center_to_testRegion_vector)
                        if train_to_test_Distance < 1.5 * IMAGE_SIZE * np.sqrt(2) + 21:
                            #                         print("Skip augmentation at image center (" + str(aug_image_center_to_testRegion_vector)
                            #                               + ")" + " distance to test set = " + str(train_to_test_Distance))
                            skip_this_image = True
                            continue
                if skip_this_image:
                    continue

                aug_image_patch = getRigidImagePatch(datasets['data'][datasetIdx],
                                                     IMAGE_SIZE, IMAGE_SIZE, center_y, center_x, angle)
                aug_image_patch_hs = getRigidImagePatch(datasets['data_hs'][datasetIdx],
                                                        IMAGE_SIZE, IMAGE_SIZE, center_y, center_x, angle)
                aug_mask_patch = getRigidImagePatch(datasets['labels'][datasetIdx],
                                                    IMAGE_SIZE, IMAGE_SIZE, center_y, center_x, angle)

                if aug_image_patch is not None:
                    aug_image_patch = (aug_image_patch - np.min(aug_image_patch)) / (
                                np.max(aug_image_patch) - np.min(aug_image_patch))
                    aug_image_patch = np.array(aug_image_patch, dtype=np.float32)
                    aug_image_patch_hs = (aug_image_patch_hs - np.min(aug_image_patch_hs)) / (
                                np.max(aug_image_patch_hs) - np.min(aug_image_patch_hs))
                    aug_image_patch_hs = np.array(aug_image_patch_hs, dtype=np.float32)
                    dataArray.append(aug_image_patch)
                    labelArray.append(aug_mask_patch)
                    dataHsArray.append(aug_image_patch_hs)
                    if SHOW_AUGMENTATION:
                        handle[0].imshow(aug_image_patch_hs, cmap='gray')
                        # handle[0].imshow(aug_image_patch, cmap='gray')
                        handle[1].imshow(aug_mask_patch, cmap='gray')
                        plt.pause(0.5)

            for regionIndex in range(dataset_components[dataset]['num_regions']):
                for augmentationIdx in range(NUM_AUGMENTATIONS_PER_LABELED_REGION):
                    # randomly perturb the orientation
                    angle = np.random.uniform(low=0, high=359.9)
                    center_x, center_y = dataset_components[dataset]['region_centroids'][regionIndex]
                    # randomly perturb the offset of the region within the tile
                    dx = np.random.uniform(low=-70, high=70)
                    dy = np.random.uniform(low=-70, high=70)
                    aug_image_center = np.array([center_x + dx, center_y + dy], dtype=np.float32)
                    skip_this_image = False

                    # do not test for proximity when augmenting the testing dataset
                    if dataset != 'test':
                        for testRegionIdx in range(dataset_numTestRegions):
                            aug_image_center_to_testRegion_vector = dataset_testRegionCentroidArray[
                                                                        testRegionIdx] - aug_image_center
                            train_to_test_Distance = np.linalg.norm(aug_image_center_to_testRegion_vector)
                            if train_to_test_Distance < 1.5 * IMAGE_SIZE * np.sqrt(2) + 21:
                                #                             print("Skip augmentation at image center (" + str(aug_image_center_to_testRegion_vector) +")"
                                #                                   " distance to test set = " + str(train_to_test_Distance))
                                skip_this_image = True
                                continue

                    # if the augmentation may include data from the test set skip this augmentation
                    # this may occur when labels of the test set are in the vicinity of labels from the training or
                    # validation set
                    if skip_this_image:
                        continue

                    aug_image_patch = getRigidImagePatch(datasets['data'][datasetIdx],
                                                         IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle)
                    aug_image_patch_hs = getRigidImagePatch(datasets['data_hs'][datasetIdx],
                                                            IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle)
                    aug_mask_patch = getRigidImagePatch(datasets['labels'][datasetIdx],
                                                        IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle)

                    # if the augmentation was successful add it to the image augmentation dataset
                    if aug_image_patch is not None:
                        aug_image_patch = (aug_image_patch - np.min(aug_image_patch)) / (
                                    np.max(aug_image_patch) - np.min(aug_image_patch))
                        aug_image_patch = np.array(aug_image_patch, dtype=np.float32)

                        aug_image_patch_hs = (aug_image_patch_hs - np.min(aug_image_patch_hs)) / (
                                    np.max(aug_image_patch_hs) - np.min(aug_image_patch_hs))
                        aug_image_patch_hs = np.array(aug_image_patch_hs, dtype=np.float32)

                        dataArray.append(aug_image_patch)
                        labelArray.append(aug_mask_patch)
                        dataHsArray.append(aug_image_patch_hs)
                        if SHOW_AUGMENTATION:
                            handle[0].imshow(aug_image_patch_hs, cmap='gray')
                            # handle[0].imshow(aug_image_patch, cmap='gray')
                            handle[1].imshow(aug_mask_patch, cmap='gray')
                            plt.pause(0.5)
    # form the training, testing and validation datasets from available labeled image data
    train_images = np.concatenate((augmentation_data[0]['train']['data'],
                                   # augmentation_data[1]['train']['data'],
                                   # augmentation_data[2]['train']['data']
                                   ))
    train_labels = np.concatenate((augmentation_data[0]['train']['labels'],
                                   # augmentation_data[1]['train']['labels'],
                                   # augmentation_data[2]['train']['labels']
                                   ))
    validate_images = np.concatenate((augmentation_data[0]['validate']['data'],
                                      # augmentation_data[1]['validate']['data'],
                                      # augmentation_data[2]['validate']['data']
                                      ))
    validate_labels = np.concatenate((augmentation_data[0]['validate']['labels'],
                                      # augmentation_data[1]['validate']['labels'],
                                      # augmentation_data[2]['validate']['labels']
                                      ))
    test_images = np.concatenate((augmentation_data[0]['test']['data'],
                                  # augmentation_data[1]['test']['data'],
                                  # augmentation_data[2]['test']['data']
                                  ))
    test_labels = np.concatenate((augmentation_data[0]['test']['labels'],
                                  # augmentation_data[1]['test']['labels'],
                                  # augmentation_data[2]['test']['labels']
                                  ))
    train_images_hs = np.concatenate((augmentation_data[0]['train']['data_hs'],))
    validate_images_hs = np.concatenate((augmentation_data[0]['validate']['data_hs'],))
    test_images_hs = np.concatenate((augmentation_data[0]['test']['data_hs'],))

    from random import shuffle

    ind_list = [i for i in range(train_images.shape[0])]
    shuffle(ind_list)
    train_images = train_images[ind_list, :, :]
    train_labels = train_labels[ind_list, :, :]
    train_images_hs = train_images_hs[ind_list, :, :, :]
    ind_list = [i for i in range(validate_images.shape[0])]
    shuffle(ind_list)
    validate_images = validate_images[ind_list, :, :]
    validate_labels = validate_labels[ind_list, :, :]
    validate_images_hs = validate_images_hs[ind_list, :, :, :]
    ind_list = [i for i in range(test_images.shape[0])]
    shuffle(ind_list)
    test_images = test_images[ind_list, :, :]
    test_labels = test_labels[ind_list, :, :]
    test_images_hs = test_images_hs[ind_list, :, :, :]
    # print out the number of training vectors, validation vectors, and test vectors
    print("Train " + str(train_images.shape[0]))
    print("Validation " + str(validate_images.shape[0]))
    print("Test " + str(test_images.shape[0]))

    #remove classes those have few samples:




    # show a collection of images from the training set
    #get_sample_display_multiple_img(train_images_hs, train_labels, n=5)

    # setup input and output images for the neural net
    X_train = train_images_hs[:,:,:,0]
    X_validate = validate_images_hs[:,:,:,0]
    X_test = test_images_hs[:1000,:,:,0]

    Y_train = train_labels
    Y_validate = validate_labels
    Y_test = test_labels[:1000,:,:]

    # add component/channel dimension to data and label images
    Y_train, X_train = remove_zero_images(Y_train, X_train)
    Y_validate, X_validate = remove_zero_images(Y_validate, X_validate)
    Y_test, X_test = remove_zero_images(Y_test, X_test)
    X_train = np.expand_dims(X_train, axis=3)
    X_validate = np.expand_dims(X_validate, axis=3)
    Y_train = np.expand_dims(Y_train, axis=3)
    Y_validate = np.expand_dims(Y_validate, axis=3)

    # convert label images to categorical format for training
    # from tensorflow.keras.utils import to_categorical
    n_classes_train = len(np.unique(Y_train))
    n_classes_validate = len(np.unique(Y_validate))
    n_classes_test = len(np.unique(Y_test))

    print("Classes in Training set: " + str(np.unique(Y_train)))
    print("Number of Classes in Training set: " + str(n_classes_train))
    print("Classes in Validation set: " + str(np.unique(Y_validate)))
    print("Number of Classes in Validation set: " + str(n_classes_validate))
    print("Classes in Test set: " + str(np.unique(Y_test)))
    print("Number of Classes in Test set: " + str(n_classes_test))

    print("Samples Training set: " + str(len(Y_train)))
    print("Samples in Validation set: " + str(len(Y_validate)))
    print("Samples in Test set: " + str(len(Y_test)))

    # Y_train_cat = tf.keras.utils.to_categorical(Y_train, num_classes=n_class)
    # Y_validate_cat = tf.keras.utils.to_categorical(Y_validate, num_classes=n_class)

    # Set input layer dimensions
    IMG_CHANNELS = X_train.shape[3]
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS)
    print("input_shape: ")
    print(input_shape)
    from tensorflow.keras.utils import to_categorical

    train_masks_cat = to_categorical(Y_train, num_classes=n_classes)
    Y_train_cat = train_masks_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))

    val_masks_cat = to_categorical(Y_validate, num_classes=n_classes)
    Y_validate_cat = val_masks_cat.reshape((Y_validate.shape[0], Y_validate.shape[1], Y_validate.shape[2], n_classes))

    #change it back to the previous one
    # metrics = [#HybridDiceAndFocalLoss(),
    #            MeanIoU(num_classes=16),
    # ]
    metrics = [#HybridDiceAndFocalLoss(),
               sm.metrics.IOUScore(threshold=0.5),
               sm.metrics.FScore(threshold=0.5)]


    save_results = home_folder + results_folder + trial_folder
    if not os.path.exists(save_results):
        os.makedirs(save_results)
    to_pickle({'data': train_images, 'labels': train_labels},
              home_folder + results_folder + trial_folder + training_filename)
    to_pickle({'data': test_images, 'labels': test_labels},
              home_folder + results_folder + trial_folder + testing_filename)
    to_pickle({'data': validate_images, 'labels': validate_labels},
              home_folder + results_folder + trial_folder + validation_filename)

    from segmentation_models import Unet
    from keras.layers import Input, Conv2D
    from keras.models import Model

    filedir = os.path.join(save_results, "save_results.txt")
    with open(filedir, 'w') as f:
        f.write('IOU scores per models:' + '\n')
        f.write('__________________________' + '\n')

    from keras_unet_collection import models
    models =['unet_resnet34']

    #models = ['Unet', 'unet_resnet34', 'unet_resnet50']
    for i in models:

        save_weights = home_folder + results_folder + trial_folder + i + "/"
        if not os.path.exists(save_weights):
            os.makedirs(save_weights)

        checkpoint = ModelCheckpoint(save_weights, monitor="val_iou_score", verbose=1, save_best_only=True, mode="max")
        early_stopping = EarlyStopping(monitor="val_iou_score", patience=150, verbose=1, mode="max")

        # create list of callbacks
        callbacks_list = [checkpoint, early_stopping]  # early_stopping

        if i == "Unet":

            unet_model = build_unet(input_shape, n_classes)


        if i == "unet_resnet34":
            BACKBONE = 'resnet34'
        if i == "unet_resnet50":
            BACKBONE = 'resnet50'
        if i == "unet_resnet34" or i == "unet_resnet34":
            preprocess_input1 = sm.get_preprocessing(BACKBONE)
            # preprocess input
            X_train = preprocess_input1(X_train)
            X_test = preprocess_input1(X_test)
            #from segmentation_models import Unet
            from keras.layers import Input, Conv2D
            from keras.models import Model

            base_model = sm.Unet(backbone_name=BACKBONE, encoder_weights='imagenet', classes=n_classes,
                              activation='softmax')  # encoder_freeze = True/False
            inp = Input(shape=(128, 128, 1))
            l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
            out = base_model(l1)
            unet_model = Model(inp, out, name=base_model.name)


        from segmentation_models.losses import bce_jaccard_loss
        #loss = bce_jaccard_loss,
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        unet_model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-1),
                           loss=loss,
                           metrics=metrics)

        print(unet_model.summary())
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False,
                                                              write_images=False,
                                                              write_steps_per_second=True)

        # Initialize the SaveImage class by passing the arguments to the __init__() function
        save_image_call = SaveImage(
            # SaveImage will only evaluate 4 images from training and validation sets
            (X_train.take(range(0, 30), axis=0), Y_train.take(range(0, 30), axis=0)),
            (X_validate.take(range(0, 30), axis=0), Y_validate.take(range(0, 30), axis=0)),
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
                                      #sample_weight=class_weights,
                                      callbacks=[save_image_call, tensorboard_callback, checkpoint])

        stop1 = datetime.now()
        # Execution time of the model
        execution_time_Unet = stop1 - start1
        print("UNet execution time is: ", execution_time_Unet)

        save_model(model=unet_model, filepath=save_weights + model_filename)

        unet_model = load_model(filepath=save_weights + model_filename,
                                custom_objects={#"hybrid_dice_and_focal_loss": metrics[0],
                                                "iou_score": metrics[0],
                                                "f1-score": metrics[1]})

        Y_validate_predicted = unet_model.predict(X_validate)
        Y_validate_predicted_argmax = np.argmax(Y_validate_predicted, axis=3)



        IOU_keras = MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(Y_validate, Y_validate_predicted_argmax)
        iou_val = IOU_keras.result().numpy()
        print("IoU on validation data =", iou_val)
        iou_array_val = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

        for i in range(n_classes):
            iou_class = calculate_individual_iou(iou_array_val, i)
            print("Validation IOU for class " +str(i)+": " +str(iou_class))


        Y_test_predicted = unet_model.predict(X_test)
        Y_test_predicted_argmax = np.argmax(Y_test_predicted, axis=3)
        IOU_keras.update_state(Y_test, Y_test_predicted_argmax)
        iou_test = IOU_keras.result().numpy()

        print("IoU on test data =", iou_test)
        iou_array_test = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        for i in range(n_classes):
            iou_class = calculate_individual_iou(iou_array_test, i)
            print("Test IOU for class " + str(i) + ": " + str(iou_class))

        plt_name = str(i) + "_plt"
        display_multiple_img(X_test, Y_test, Y_test_predicted_argmax, plt_name, save_weights, n=5)

        with open(filedir, 'a') as f:
            f.write('Model Name: ')
            f.write(str(i) + '\n')
            f.write("Best IOU on Validation data: ")
            f.write(str(iou_val) + '\n')
            f.write("Best IOU on Test data: ")
            f.write(str(iou_test) + '\n')
            f.write("Total Execution Time: ")
            f.write(str(execution_time_Unet) + '\n')
            f.write("_________________________" + '\n')

