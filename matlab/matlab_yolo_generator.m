clear;
clc;
close all;
%DATASET_INDEX = 1;
IMAGE_SIZE = 320;

NUM_AUGMENTATIONS_PER_LABELED_REGION = 10;
NUM_RANDOM_AUGMENTATIONS = 1;
SHOW_AUGMENTATION = true;
%SHOW_AUGMENTATION = false;

% split the data within each image test/train
% test 20%
% validate 15%
% train 65%
pct_test = 0.2;
pct_val = 0.15;

PATH_ROOT ="/home/arwillis/PyCharm/data";
%PATH_ROOT ="/home.local/local-arwillis/PyCharm/data";
%PATH_ROOT ="/home.md1/jzhang72/PycharmProjects/lidar-segmentation/yolov7/data/lidar_data";

data_file = ["KOM_image_data.mat","MLS_image_data.mat","UCB_image_data.mat","Sayil_image_data.mat"];
input_filenames_hs = ["KOM/kom_dsm_lidar_hs.png","MLS/MLS_DEM_hs.png","UCB/UCB_elev_adjusted_hs.png"];
label_files = ["KOM_ground_truth_labels.mat","MLS_ground_truth_labels.mat","UCB_ground_truth_labels.mat"];

% yolov7_output_data_paths = ["yolov7/train/", "yolov7/val/", "yolov7/test/"];
yolov7_output_data_paths = ["yolov7/images/train/", "yolov7/images/val/", "yolov7/images/test/"];

region(1).Name = 'Annular structure';
region(1).WINDOWSIZE = 40;
region(1).Color = [1 .8 .8]; % light red
region(1).LabelValue = 1;
region(2).Name = 'Platform';
region(2).WINDOWSIZE = 80;
region(2).Color = [.67 .84 .9]; % light blue
region(2).LabelValue = 2;

jsonYoloData{1}=struct('imageList', [], 'categoryList', [], 'annotationList', []);
jsonYoloData{2}=jsonYoloData{1};
jsonYoloData{3}=jsonYoloData{1};
jsonYoloImage = struct("filename", "0.png", ...
    "dims", struct("width", 640, ...
    "height", 480), ...
    "id", 0);
jsonYoloCategory = struct("name", "platform", ...
    "id", 1);
jsonYoloAnnotation = struct("category_id", 0, ...
    "image_id", 0, ...
    "bb", struct("x", 0, ...
    "y", 0, ...
    "width", 0, ...
    "height", 0), ...
    "area", 0, ...
    "segmentation", [0,0,0,0]);

for dataset_idx=1:3
    for label_idx=1:length(region)
        jsonYoloCategoryNew = jsonYoloCategory;
        jsonYoloCategoryNew.name = region(label_idx).Name;
        jsonYoloCategoryNew.id = region(label_idx).LabelValue;
        jsonYoloData{dataset_idx}.categoryList = [jsonYoloData{dataset_idx}.categoryList, jsonYoloCategoryNew];
    end
end

data_vector_ID = 1;
% loop over ground_truth datasets
for label_file_idx=1:length(label_files)
    clear all_labels;
    clear geotiff_data;
    data_filename = strcat(PATH_ROOT,'/',data_file(label_file_idx));
    data_hs_filename = strcat(PATH_ROOT,'/',input_filenames_hs(label_file_idx));
    label_filename = strcat(PATH_ROOT,'/',label_files(label_file_idx));
    load(data_filename);
    load(label_filename);
    geotiff_data_hs = imread(data_hs_filename);
    
    [mask_rows, mask_cols] = size(geotiff_data);
    geotiff_data_mask = zeros(mask_rows, mask_cols);
        
    % generate a segmentation mask for the dataset image
    annotation_index = 1;
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
            geotiff_annotations(annotation_index).category_id = region(label_idx).LabelValue;
            geotiff_annotations(annotation_index).vertices = polygon_vertices;
            annotation_index = annotation_index + 1;
            num_vertices = size(polygon_vertices,1);
            img_tile = geotiff_data_hs(tile_tlc(2):tile_brc(2),tile_tlc(1):tile_brc(1));
            subplot(1,2,1), hold off, imshow(img_tile,[0 255]);
            bw = poly2mask(polygon_vertices(:,1), polygon_vertices(:,2), mask_rows, mask_cols);
            geotiff_data_mask(bw==1) = region(label_idx).LabelValue;
            polygon_vertices = polygon_vertices - ones(num_vertices,1)*double(tile_tlc);
            subplot(1,2,1), hold on, drawpolygon('Position', polygon_vertices, ...
                'LineWidth',1,'FaceAlpha', 0.3, 'Color', region(label_idx).Color, ...
                'SelectedColor', region(label_idx).Color);
            subplot(1,2,1), hold off;
            mask_tile = geotiff_data_mask(tile_tlc(2):tile_brc(2),tile_tlc(1):tile_brc(1));
            subplot(1,2,2), imshow(mask_tile,[0 2]);
            drawnow;
            %pause(0.5);
        end
    end
    
    % loop over labels within a given dataset
    for label_idx=1:length(all_labels)
        label_set = all_labels(label_idx).labels;
        num_labels = length(label_set);
        random_indices=randperm(num_labels);
        pct_train = 1 - pct_test - pct_val;
        split_idx_train_val = floor(num_labels*pct_train);
        split_idx_val_test = floor(num_labels*(pct_train+pct_val));
        training_indices = random_indices(1:split_idx_train_val);
        validation_indices = random_indices((split_idx_train_val+1):split_idx_val_test);
        test_indices = random_indices((split_idx_val_test+1):end);
        training_set=all_labels(label_idx).labels(training_indices);
        validation_set=all_labels(label_idx).labels(validation_indices);
        test_set=all_labels(label_idx).labels(test_indices);
        datasets = {training_set, validation_set, test_set};
        
        % store are all the region centers for the test data
        test_region_centers = zeros(length(test_set),2);
        for test_region_idx=1:length(test_set)
            test_region_centers(test_region_idx,:) = mean(test_set(test_region_idx).vertices);
        end
        
        % loop over train, validation and test datasets for generation
        for output_dataset_idx=1:length(datasets)
            dataset = datasets{output_dataset_idx};
            yolov7_output_path = yolov7_output_data_paths(output_dataset_idx);
            if ~exist(yolov7_output_path, 'dir')
                mkdir(yolov7_output_path)
            end
            for dataIdx=1:length(dataset)
                dataValue = dataset(dataIdx);
                for augmentationIdx=1:NUM_AUGMENTATIONS_PER_LABELED_REGION
                    % randomly perturb the orientation
                    angle = rand(1,1)*360;
                    center = mean(dataValue.vertices); %datavalue.center;
                    center_x = center(1);
                    center_y = center(2);
                    % randomly perturb the offset of the region within the tile
                    dx = 140*(rand(1,1)-0.5);
                    dy = 140*(rand(1,1)-0.5);
                    aug_image_center = [center_x + dx, center_y + dy];
                    skip_this_image = false;
                    % do not test for proximity when augmenting the testing dataset
                    if (output_dataset_idx ~= 3)
                        for testRegionIdx=1:length(datasets(3))
                            aug_image_center_to_testRegion_vector = test_region_centers(testRegionIdx,:) - aug_image_center;
                            train_to_test_Distance = norm(aug_image_center_to_testRegion_vector);
                            if (train_to_test_Distance < 1.5 * IMAGE_SIZE * sqrt(2) + 21)
                                fprintf(1,"Skip augmentation at image center (x,y)=(%d,%d) distance to test set = %d.\n", ...
                                    aug_image_center(1),aug_image_center(2),train_to_test_Distance)
                                skip_this_image = true;
                                continue
                            end
                        end
                    end
                    % if the augmentation may include data from the test set skip this augmentation
                    % this may occur when labels of the test set are in the vicinity of labels from the training or
                    % validation set
                    if (skip_this_image)
                        %fprintf(1,"Skipping this image.\n");
                        continue;
                    end
                    aug_image_patch = getRigidImagePatch(geotiff_data, ...
                        IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle);
                    aug_image_patch_hs = getRigidImagePatch(geotiff_data_hs, ...
                        IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle);
                    aug_mask_patch = getRigidImagePatch(geotiff_data_mask, ...
                        IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle);
                    % if the augmentation was successful, 
                    % compute the annotations for the image and add it to the image augmentation dataset
                    if ~isempty(aug_image_patch)
                        aug_annotations = getRigidImagePatchAnnotations(geotiff_annotations, ...
                            IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle, data_vector_ID);
                        aug_image_patch = (aug_image_patch - min(aug_image_patch,[],'all')) / (max(aug_image_patch,[],'all') - min(aug_image_patch,[],'all'));
                        aug_image_patch = single(aug_image_patch);
                        img_filename = sprintf("img_%05d.png",data_vector_ID);
                        jsonYoloImageNew = jsonYoloImage;
                        jsonYoloImageNew.filename = img_filename;
                        jsonYoloImageNew.dims.width = IMAGE_SIZE;
                        jsonYoloImageNew.dims.height = IMAGE_SIZE;
                        jsonYoloImageNew.id = data_vector_ID;
                        jsonYoloData{output_dataset_idx}.imageList = [jsonYoloData{output_dataset_idx}.imageList, ...
                            jsonYoloImageNew];
                        % jsonYoloData.annotationList = [jsonYoloData.annotationList, jsonYoloAnnotation];
                        jsonYoloData{output_dataset_idx}.annotationList = [jsonYoloData{output_dataset_idx}.annotationList, ...
                            aug_annotations'];
                        data_vector_ID = data_vector_ID + 1;
                        
                        output_image_file = strcat(yolov7_output_path,'/',img_filename);
                        imwrite(aug_image_patch, output_image_file);
                        if (SHOW_AUGMENTATION)
                            subplot(1,3,1), hold off, imshow(aug_image_patch_hs, [0 255])
                            
                            for annotation_idx=1:length(aug_annotations)
                                bbox_tlc = [aug_annotations(annotation_idx).bb.x, ...
                                    aug_annotations(annotation_idx).bb.y];
                                bbox_dims = [aug_annotations(annotation_idx).bb.width, ...
                                    aug_annotations(annotation_idx).bb.height];
                                bbox_vertices = [bbox_tlc;
                                    bbox_tlc(1), bbox_tlc(2) + bbox_dims(2);
                                    bbox_tlc(1) + bbox_dims(1), bbox_tlc(2) + bbox_dims(2);
                                    bbox_tlc(1) + bbox_dims(1), bbox_tlc(2);
                                    bbox_tlc;];
                                label_idx = aug_annotations(annotation_idx).category_id;
                                subplot(1,3,1), hold on, drawpolygon('Position', bbox_vertices, ...
                                    'LineWidth',1,'FaceAlpha', 0.3, 'Color', region(label_idx).Color, ...
                                    'SelectedColor', region(label_idx).Color);
                            end
                            
                            subplot(1,3,2), imshow(aug_image_patch, [])
                            subplot(1,3,3), imshow(aug_mask_patch, [])
                            %figure(3), imshow(aug_mask_patch, [])
                            drawnow;
                            %pause(0.5)
                        end
                    end
                end
            end
        end
    end
end

% Write the JSON annotations to a file with pretty printing
yolov7_output_annotation_paths = "yolov7/annotations";
if ~exist(yolov7_output_annotation_paths, 'dir')
    mkdir(yolov7_output_annotation_paths)
end

output_json_files = ["train.json","val.json", "test.json"];
for dataset_idx=1:3
    jsonStr = jsonencode(jsonYoloData{dataset_idx});
    annotation_output_path = strcat(yolov7_output_annotation_paths,"/",output_json_files(dataset_idx));
    fid = fopen(annotation_output_path, 'w');
    fprintf(fid, '%s\n', jsonStr);
    fclose(fid);
end

function image_annotations = getRigidImagePatchAnnotations(annotations, height, width, center_y, center_x, angle, image_id)
jsonYoloAnnotation = struct("category_id", 0, ...
    "image_id", 0, ...
    "bb", struct("x", 0, ...
    "y", 0, ...
    "width", 0, ...
    "height", 0), ...
    "area", 0, ...
    "segmentation", [0,0,0,0]);
theta = (angle / 180) * pi;
cos_t = cos(theta);
sin_t = sin(theta);
bound_w = ceil(height * abs(sin_t) + width * abs(cos_t));
bound_h = ceil(height * abs(cos_t) + width * abs(sin_t));
xy_start = [floor(center_x - (bound_w / 2) + 1), floor(center_y - (bound_h / 2) + 1)];
xy_end = [ceil(center_x + (bound_w / 2) + 1), ceil(center_y + (bound_h / 2) + 1)];
cropped_width = xy_end(1) - xy_start(1) + 1;
cropped_height = xy_end(2) - xy_start(2) + 1;
xy_translation = [0.5 * (cropped_width - (cos_t * cropped_width + sin_t * cropped_height)), ...
    0.5 * (cropped_height - (-sin_t * cropped_width + cos_t * cropped_height))];
image_patch_T = [cos_t, sin_t, xy_translation(1); -sin_t, cos_t, xy_translation(2); 0, 0, 1];
image_annotations = [];
xy_start_img = [floor(cropped_width/2) - (width / 2) + 1, floor(cropped_height/2) - (height / 2) + 1];
imageBBox = [xy_start_img, width, height];
for annotation_idx=1:length(annotations)
    cur_annotation = annotations(annotation_idx);
    num_vertices = size(cur_annotation.vertices,1);
    vertices_img = cur_annotation.vertices - ones(num_vertices,1)*xy_start;
    vertices_transformed = image_patch_T*[vertices_img'; 
        ones(1,num_vertices)];
    vertices_transformed = vertices_transformed(1:2,:)';
    vertices_transformed = vertices_transformed - ones(num_vertices,1)*xy_start_img;    
    bbox_tlc_img = min(vertices_transformed,[],1);
    bbox_dims = max(vertices_transformed,[],1) - bbox_tlc_img;
    bbox_annotation_transformed = [bbox_tlc_img, bbox_dims];
    
    % test intersection
    overlapRatio = bboxOverlapRatio(imageBBox, bbox_annotation_transformed);
        
    if (overlapRatio > 0)
        % put bbox in image coordinate system
        % clip the bbox to the image boundaries
        bbox_tlc_img(bbox_tlc_img < 0) = 0;
        if (bbox_tlc_img(1) + bbox_dims(1) > width)
            bbox_dims(1) = width - bbox_tlc_img(1);
        end
        if (bbox_tlc_img(2) + bbox_dims(2) > height)
            bbox_dims(2) = height - bbox_tlc_img(2);
        end
        
        %fprintf(1,"Found annotation in image.\n");
        jsonYoloAnnotationNew = jsonYoloAnnotation;
        jsonYoloAnnotationNew.category_id = cur_annotation.category_id;
        jsonYoloAnnotationNew.image_id = image_id;
        jsonYoloAnnotationNew.bb.x = int32(bbox_tlc_img(1));
        jsonYoloAnnotationNew.bb.y = int32(bbox_tlc_img(2));
        jsonYoloAnnotationNew.bb.width = int32(bbox_dims(1));
        jsonYoloAnnotationNew.bb.height = int32(bbox_dims(2));
        jsonYoloAnnotationNew.area =  bbox_dims(1) * bbox_dims(2);
        jsonYoloAnnotationNew.segmentation = [];
        if (bbox_dims(1) > 3 && bbox_dims(2) > 3) 
            image_annotations = [image_annotations; jsonYoloAnnotationNew];
        end
    end
end
end

function image_patch_aug = getRigidImagePatch(img, height, width, center_y, center_x, angle)
theta = (angle / 180) * pi;
cos_t = cos(theta);
sin_t = sin(theta);
bound_w = ceil(height * abs(sin_t) + width * abs(cos_t));
bound_h = ceil(height * abs(cos_t) + width * abs(sin_t));
xy_start = [floor(center_x - (bound_w / 2) + 1), floor(center_y - (bound_h / 2) + 1)];
xy_end = [ceil(center_x + (bound_w / 2) + 1), ceil(center_y + (bound_h / 2) + 1)];
[rows, cols, ~] = size(img);
image_patch_aug = [];
if (any(xy_start < 1) || xy_start(1) > cols || xy_start(2) > rows || ...
        any(xy_end < 1) || xy_end(1) > cols || xy_end(2) > rows)
    %print("Could not extract patch at location (" + str((center_x, center_y)) + ")")
    return
end
cropped_image_patch = img(xy_start(2):xy_end(2), xy_start(1):xy_end(1), :);
[cropped_height, cropped_width] = size(cropped_image_patch);
%if (cropped_height ~= height || cropped_width ~= width)
%    return
%end
% xy_rotation_centerpt = np.array([width / 2, height / 2])
xy_translation = [0.5 * (cropped_width - (cos_t * cropped_width + sin_t * cropped_height)), ...
    0.5 * (cropped_height - (-sin_t * cropped_width + cos_t * cropped_height))];
image_patch_T = [cos_t, sin_t, xy_translation(1); -sin_t, cos_t, xy_translation(2); 0, 0, 1];
T = affine2d(image_patch_T');
transformed_image_patch = imwarp(cropped_image_patch, T, 'OutputView', imref2d(size(cropped_image_patch)));
xy_center_newimg = floor(size(transformed_image_patch) / 2.0);
xy_start = [xy_center_newimg(1) - (width / 2) + 1, xy_center_newimg(2) - (height / 2) + 1];
xy_end = [xy_center_newimg(1) + (width / 2), xy_center_newimg(2) + (height / 2)];
image_patch_aug = transformed_image_patch(xy_start(2):xy_end(2), xy_start(1):xy_end(1));
end


% {
%   "imageList": [
%     {
%       "filename": "0.png",
%       "dims": {
%         "width": 640,
%         "height": 480
%       },
%       "id": 0
%     }
%   ],
%   "categoryList": [
%     {
%       "name": "annular structure",
%       "id": 0
%     },
%     {
%       "name": "paltform",
%       "id": 1
%     },
%   ],
%   "annotationList": [
%     {
%       "category_id": 0,
%       "image_id": 0,
%       "bb": {
%         "x": 134.46558,
%         "y": 390.5586,
%         "width": 343.91245,
%         "height": 89.44141
%       },
%       "area": 0.0,
%       “segmentation”: [[34, 55, 10, 71, 76, 23, 98, 43, 11, 8]]  		% not necessary
%     },
% }