clear;
clc;
close all;

IMAGE_SIZE = 320;
SHOW_TILES = false;

%PATH_ROOT ="/home/arwillis/PyCharm/data";
%PATH_ROOT ="/home.local/local-arwillis/PyCharm/data";
PATH_ROOT ="/home.md1/jzhang72/PycharmProjects/lidar-segmentation";

input_images = ["yolo_KOM_image_classified.png","yolo_MLS_image_classified.png", "yolo_UCB_image_classified.png"];
output_images = ["yolo_KOM_image_classified_w_gt.png","yolo_MLS_image_classified_w_gt.png", "yolo_UCB_image_classified_w_gt.png"];
label_files = ["KOM_ground_truth_labels.mat","MLS_ground_truth_labels.mat","UCB_ground_truth_labels.mat"];

%data_file = ["KOM_image_data.mat","MLS_image_data.mat","UCB_image_data.mat","Sayil_image_data.mat"];
%data_filename = strcat(PATH_ROOT,'/data',data_file(DATASET_INDEX));
%data_hs_filename = strcat(PATH_ROOT,'/data',input_filenames_hs(DATASET_INDEX));

region(1).Name = 'Annular structure';
region(1).WINDOWSIZE = 40;
region(1).Color = [0 255 0]; % green
region(1).LabelValue = 1;
region(2).Name = 'Platform';
region(2).WINDOWSIZE = 80;
region(2).Color = [255 255 0]; % yellow
region(2).LabelValue = 2;

for DATASET_INDEX=1:length(input_images)
    label_filename = strcat(PATH_ROOT, '/data/', label_files(DATASET_INDEX));
    image_datafile_in = strcat(PATH_ROOT, '/inference_results/', input_images(DATASET_INDEX));
    image_datafile_out = strcat(PATH_ROOT, '/inference_results/', output_images(DATASET_INDEX));
    image_data = imread(image_datafile_in);
    load(label_filename);
    [mask_rows, mask_cols, channel] = size(image_data);

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
            label_str = ''; %region(label_idx).Name;
            label_color = region(label_idx).Color;
            image_data = insertObjectAnnotation(image_data, 'rectangle', ...
                annotation_bbox, label_str, 'TextBoxOpacity', 0.9, 'FontSize', 18, ...
                'LineWidth', 3, 'Color', label_color);
            image_tile_data = image_data(tile_tlc(2):tile_brc(2),tile_tlc(1):tile_brc(1),:);
            if (SHOW_TILES)
                imshow(image_tile_data);
                drawnow;
            end
        end
    end
    imwrite(image_data, image_datafile_out);
    fprintf("Dataset %d processing completed. Image saved.\n", DATASET_INDEX);
end