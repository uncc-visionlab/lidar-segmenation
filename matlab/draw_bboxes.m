clear;
clc;
close all;
DATASET_INDEX = 1;
IMAGE_SIZE = 320;
SHOW_TILES = true;

PATH_ROOT ="/home/arwillis/PyCharm/data";
%PATH_ROOT ="/home.local/local-arwillis/PyCharm/data";
%PATH_ROOT ="/home.md1/jzhang72/PycharmProjects/lidar-segmentation/yolov7/data/lidar_data";

image_datafile_in = "yolo_KOM_image_classified.png";
image_datafile_out = "yolo_KOM_image_classified_w_gt.png";
image_data = imread(image_datafile_in);
data_file = ["KOM_image_data.mat","MLS_image_data.mat","UCB_image_data.mat","Sayil_image_data.mat"];
input_filenames_hs = ["KOM/kom_dsm_lidar_hs.png","MLS/MLS_DEM_hs.png","UCB/UCB_elev_adjusted_hs.png"];
label_files = ["KOM_ground_truth_labels.mat","MLS_ground_truth_labels.mat","UCB_ground_truth_labels.mat"];
data_filename = strcat(PATH_ROOT,'/',data_file(DATASET_INDEX));
data_hs_filename = strcat(PATH_ROOT,'/',input_filenames_hs(DATASET_INDEX));
label_filename = strcat(PATH_ROOT,'/',label_files(DATASET_INDEX));

load(label_filename);
[mask_rows, mask_cols] = size(image_data);

region(1).Name = 'Annular structure';
region(1).WINDOWSIZE = 40;
region(1).Color = [1 .8 .8]; % light red
region(1).LabelValue = 1;
region(2).Name = 'Platform';
region(2).WINDOWSIZE = 80;
region(2).Color = [.67 .84 .9]; % light blue
region(2).LabelValue = 2;


for label_idx=1:length(all_labels)
    label_set = all_labels(label_idx).labels;
    num_labels = length(label_set);
    for dataIdx=1:num_labels
        dataValue = label_set(dataIdx);
        polygon_vertices = dataValue.vertices;
        center = mean(polygon_vertices, 1);
        bbox_tlc = min(polygon_vertices,[],1);
        bbox_dims = max(polygon_vertices,[],1) - bbox_tlc;
        bbox_vertices = [bbox_tlc;
            bbox_tlc(1), bbox_tlc(2) + bbox_dims(2);
            bbox_tlc(1) + bbox_dims(1), bbox_tlc(2) + bbox_dims(2);
            bbox_tlc(1) + bbox_dims(1), bbox_tlc(2);
            bbox_tlc;];
        polygon_vertices = bbox_vertices;
        tile_tlc = int32([(center(1) - (IMAGE_SIZE/2)), (center(2) - (IMAGE_SIZE/2))]);
        tile_tlc(tile_tlc <= 0) = 1;
        tile_brc = int32([(center(1) + (IMAGE_SIZE/2)), (center(2) + (IMAGE_SIZE/2))]);
        if (tile_brc(1) > mask_cols)
            tile_brc(1) = mask_cols;
        end
        if (tile_brc(2) > mask_rows)
            tile_brc(2) = mask_rows;
        end
        num_vertices = size(polygon_vertices,1);
        % Draw the objects on the frame.
        annotation_bbox = [bbox_tlc, bbox_dims];
        label_str = region(label_idx).Name;
        image_data = insertObjectAnnotation(image_data, 'rectangle', ...
            annotation_bbox, label_str, TextBoxOpacity=0.9, FontSize=18, Color="Cyan");
        image_tile_data = image_data(tile_tlc(2):tile_brc(2),tile_tlc(1):tile_brc(1),:);
        if (SHOW_TILES)
            imshow(image_tile_data);
            drawnow;
        end
    end
end
imwrite(image_data, image_datafile_out);