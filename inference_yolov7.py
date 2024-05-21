import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
import h5py
import scipy.io as sio
from matplotlib import pyplot as plt

from bbox_processing import merge_bboxes

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def load_model(device):
    # Initialize
    weights = opt.weights
    img_sz = opt.img_size
    trace = opt.trace
    set_logging()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_sz = check_img_size(img_sz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier (default False)
    if opt.classify:
        model_c = load_classifier(name='resnet101', n=2)  # initialize
        model_c.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    else:
        model_c = None

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # class_colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [[255, 0, 0], [0, 0, 255]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_sz, img_sz).to(device).type_as(next(model.parameters())))  # run once

    return model, model_c, names, colors


def detect(img, img_hs, model, model_c, names, colors, device, tl_pos, bboxes):
    save_label, img_sz = opt.save_label, opt.img_size
    half = device.type != 'cpu'  # half precision only supported on CUDA

    old_img_w = old_img_h = img_sz
    old_img_b = 1

    t0 = time.time()

    # Padded resize
    img0 = img.copy()
    # Convert
    img0 = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x imgsz x imgsz
    img0 = np.ascontiguousarray(img0)

    img0 = torch.from_numpy(img0).to(device)
    img0 = img0.half() if half else img0.float()  # uint8 to fp16/32
    img0 /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img0.ndimension() == 3:
        img0 = img0.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img0.shape[0] or old_img_h != img0.shape[2] or old_img_w != img0.shape[3]):
        for i in range(3):
            model(img0, augment=opt.augment)[0]

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img0, augment=opt.augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t3 = time_synchronized()

    # Apply Classifier
    if opt.classify and model_c is not None:
        pred = apply_classifier(pred, model_c, img0, img)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s = 'No target found, '
        im0 = img_hs
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_label:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    xc_yc_w_h = [tl_pos[0] + xywh[0] * img_sz, tl_pos[1] + xywh[1] * img_sz, xywh[2] * img_sz, xywh[3] * img_sz]  # [x_c, y_c, w, h]
                    line = [cls.cpu().data.numpy(), xc_yc_w_h, conf.cpu().data.numpy()]  # label
                    # print(line)
                    bboxes.append(line)

                label = ''  # f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

        # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (time.time() - t0)):.1f}ms) Total, ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS.')

        # Stream results
        # if view_img:
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        # if save_img:
        #     cv2.imwrite(save_path, im0)
        #     print(f" The image with the result is saved in: {save_path}")

    return im0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-label', action='store_true', help='save results to *.csv')
    parser.add_argument('--merge-bbox', action='store_true', help='merge adjacent bounding boxes')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--classify', action='store_true', help='no second-stage classification')
    opt = parser.parse_args()
    print(opt)

    IMAGE_SIZE = opt.img_size
    num_classes = 2
    SHOW_CLASSIFICATIONS = False
    home_folder = './'
    # home_folder = '/home/arwillis/PyCharm/'
    results_folder = 'inference_results/'

    gis_data_path = ['data/KOM/', 'data/MLS/', 'data/UCB/', 'data/Sayil/']

    gis_input_filenames_hs = ['kom_dsm_lidar_hs.png',
                              'MLS_DEM_hs.png',
                              'UCB_elev_adjusted_hs.png',
                              'Sayil_regional_DEM_hs.png']

    gis_output_image_filenames = ['yolo_KOM_image_classified.png',
                                  'yolo_MLS_image_classified.png',
                                  'yolo_UCB_image_classified.png',
                                  'yolo_Sayil_image_classified.png']

    gis_output_label_filenames = ['yolo_KOM_image_classified_labels.csv',
                                  'yolo_MLS_image_classified_labels.csv',
                                  'yolo_UCB_image_classified_labels.csv',
                                  'yolo_Sayil_image_classified_labels.csv']

    gis_output_label_merged_filenames = ['yolo_KOM_image_classified_labels_merged.csv',
                                         'yolo_MLS_image_classified_labels_merged.csv',
                                         'yolo_UCB_image_classified_labels_merged.csv',
                                         'yolo_Sayil_image_classified_labels_merged.csv']

    gis_input_filenames_mat = ['KOM_image_data.mat',
                               'MLS_image_data.mat',
                               'UCB_image_data.mat',
                               'Sayil_image_data.mat']

    # prepare the model
    device = select_device(opt.device)
    net, net_c, class_names, class_colors = load_model(device)
    output_folder = home_folder + results_folder
    os.makedirs(output_folder, exist_ok=True)

    for DATASET_INDEX in [2]:  # range(len(gis_input_filenames_mat)):
        print("Inference on " + gis_input_filenames_mat[DATASET_INDEX].split('.')[0])
        img_filename_mat = home_folder + 'data/' + gis_input_filenames_mat[DATASET_INDEX]
        if DATASET_INDEX == 3:
            with h5py.File(img_filename_mat, 'r') as f:
                # print(f.keys())
                image_data = np.array(f['geotiff_data']).transpose()
        else:
            mat_data = sio.loadmat(img_filename_mat, squeeze_me=True)
            image_data = mat_data['geotiff_data']

        output_filename = output_folder + gis_output_image_filenames[DATASET_INDEX]
        label_save_dir = output_folder + gis_output_label_filenames[DATASET_INDEX]
        img_filename_hs = home_folder + gis_data_path[DATASET_INDEX] + gis_input_filenames_hs[DATASET_INDEX]
        image_data_hs = cv2.imread(img_filename_hs)

        [rows, cols] = image_data.shape[0:2]
        xy_pixel_skip = (80, 80)    # (IMAGE_SIZE, IMAGE_SIZE)  # (32, 32)
        xy_pixel_margin = np.array([np.round((IMAGE_SIZE + 1) / 2), np.round((IMAGE_SIZE + 1) / 2)],
                                   dtype=np.int32)

        x_vals = range(xy_pixel_margin[0], cols - xy_pixel_margin[0], xy_pixel_skip[0])
        y_vals = range(xy_pixel_margin[1], rows - xy_pixel_margin[1], xy_pixel_skip[1])

        # plt.tight_layout()
        if SHOW_CLASSIFICATIONS:
            figure, ax = plt.subplots(nrows=num_classes, ncols=2, figsize=(8, num_classes * 2))

        label_image_predicted = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.float32)
        label_image = np.zeros(image_data.shape, dtype=np.float32)
        classification_count_image = np.zeros(image_data.shape, dtype=np.float32)
        pred_bboxes = []

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
                test_image = np.float32(test_image)
                test_image = cv2.normalize(test_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                input_test_image = np.stack((test_image,) * 3, axis=-1)

                tl_pos_xy = [x - xy_pixel_margin[0], y - xy_pixel_margin[1]]  # top_left corner of the current tile in the big image
                with torch.no_grad():
                    test_image_predicted = detect(input_test_image, test_image_hs, net, net_c, class_names, class_colors, device, tl_pos_xy,
                                                  pred_bboxes)

                label_image_predicted[(y - xy_pixel_margin[1]):(y + xy_pixel_margin[1]),
                (x - xy_pixel_margin[0]):(x + xy_pixel_margin[0]), :] += test_image_predicted

                if SHOW_CLASSIFICATIONS:
                    ax.ravel()[0].imshow(test_image_hs, cmap='gray')
                    ax.ravel()[0].set_title("Hillshade Image: " + "(" + str(x) + ", " + str(y) + ")")
                    ax.ravel()[1].set_title("Predicted Image: " + "(" + str(x) + ", " + str(y) + ")")
                    ax.ravel()[1].imshow(test_image_predicted, cmap='gray')
                    plt.show(block=False)
                    plt.pause(0.1)

        cv2.imwrite(output_filename, np.array(label_image_predicted, dtype=np.uint8))
        print("Prediction completed. " + str(len(pred_bboxes)) + " total bounding boxes detected.")

        if opt.save_label:
            for k in range(len(pred_bboxes)):
                pred_bboxes[k] = [float(pred_bboxes[k][0]), [float(ele) for ele in pred_bboxes[k][1]], float(pred_bboxes[k][2])]
                # print(pred_bboxes[k])
            with open(label_save_dir, 'w') as file_1:
                w = csv.writer(file_1)
                field = ["class", "bbox (x_c, y_c, width, height)", "conf"]
                w.writerow(field)
                pred_bboxes = sorted(pred_bboxes, key=lambda bbox: bbox[1][0])  # Sort bboxes by x_c
                w.writerows(pred_bboxes)
            if opt.merge_bbox:
                # file_1 = open(label_save_dir, 'r')
                # data = list(csv.reader(file_1, delimiter=","))
                # file_1.close()
                data = pred_bboxes
                new_data = merge_bboxes(data)
                label_merged_save_dir = output_folder + gis_output_label_merged_filenames[DATASET_INDEX]
                with open(label_merged_save_dir, 'w') as file_2:
                    writer_2 = csv.writer(file_2)
                    writer_2.writerow(field)
                    writer_2.writerows(new_data)
                print("Merging completed. " + str(len(new_data)) + " total bounding boxes after merging.")

        if SHOW_CLASSIFICATIONS:
            figure, bx = plt.subplots(nrows=n, ncols=2, figsize=(8, n * 2))
            bx.ravel()[0].imshow(image_data_hs, cmap='gray')
            bx.ravel()[0].set_title("Hillshade Image")
            bx.ravel()[1].set_title("Predicted Image")
            bx.ravel()[1].imshow(label_image, cmap='gray')
            plt.show()

        print("Dataset", DATASET_INDEX, "inference completed. Results saved.")
