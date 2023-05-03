clear;
clc;
hold off;
close all;

output_folder_name = 'YOLO_data/annular_structure/';
platform = false;
IMAGE_SIZE = 128;
% Image augmentation settings
NUM_AUGMENTATIONS_PER_LABELED_REGION = 70;
NUM_RANDOM_AUGMENTATIONS = 500;
% SHOW_AUGMENTATION = true;
SHOW_AUGMENTATION = false;

% split the data within each image test/train
% test 20%
% merge all the train and split the training into train/validate
% train 65%
% validate 15%
pct_test = 0.2;
pct_val = 0.15;

home_folder = '/home/fjannat/Documents/Lidar_seg/';
% results_folder = 'results/';
% trial_folder = 'unet/trial/';
% model_filename = 'unet_model.hdf5';
% training_filename = 'data_training.pkl';
% testing_filename = 'data_testing.pkl';
% validation_filename = 'data_validation.pkl';

gis_data_path = {'data/KOM/', 'data/MLS/', 'data/UCB/'};
gis_input_filenames_hs = {'kom_dsm_lidar_hs.png', 'MLS_DEM_hs.png', 'UCB_elev_adjusted_hs.png'};
gis_input_filenames_mat = {'KOM_image_data.mat', 'MLS_image_data.mat', 'UCB_image_data.mat'};
gis_input_gt_filenames_mat = {'KOM_ground_truth_labels.mat', 'MLS_ground_truth_labels.mat', 'UCB_ground_truth_labels.mat'};

image_data_hs = {};

for filenameIdx = 1:length(gis_input_filenames_hs)
    img_filename_hs = strcat(home_folder, gis_data_path{filenameIdx}, gis_input_filenames_hs{filenameIdx});
    image_data_hs_ex = imread(img_filename_hs);
    image_data_hs{filenameIdx} = image_data_hs_ex;
end

image_data = {};
for filenameIdx = 1:length(gis_input_filenames_mat)
    mat_data = struct;
    img_filename_mat = strcat(home_folder, 'data/', gis_input_filenames_mat{filenameIdx});
    mat_data = load(img_filename_mat, 'geotiff_data');
    image_data{end+1} = mat_data.geotiff_data;
end

image_labels = {};
for filenameIdx = 1:length(gis_input_gt_filenames_mat)
    mat_data = struct;
    img_gt_filename_mat = strcat(home_folder, 'data/', gis_input_gt_filenames_mat{filenameIdx});
    mat_data = load(img_gt_filename_mat, 'all_labels');
    image_labels{end+1} = mat_data.all_labels;
end
datasets = struct('data', {}, 'data_hs', {}, 'labels', {}, 'region_centroids', {}, 'num_regions', {}, 'analysis', {});
for datasetIdx = 1:length(image_data)
    % image = zeros([size(image_data{datasetIdx}, 1), size(image_data{datasetIdx}, 2), 3]);
    image = image_data{datasetIdx}(:, :, 1);
    datasets(end+1).data = image;
    image_hs = image_data_hs{datasetIdx};
    datasets(end).data_hs = image_hs;
    
    if platform
        start = 2;
        finish = 3;
    else
        start = 1;
        finish = 2;
    end
    for labelIdx = start:finish-1 % A
        regions = [];
        fprintf('label %d\n', labelIdx);
        for regionIdx = 1:length(image_labels{datasetIdx}(labelIdx).labels) % B
            region_data = struct('label_value', image_labels{datasetIdx}(labelIdx).labels(regionIdx).label_value,...
                                 'centroid', mean(image_labels{datasetIdx}(labelIdx).labels(regionIdx).vertices),...
                                 'vertices', image_labels{datasetIdx}(labelIdx).labels(regionIdx).vertices,...
                                 'ID', image_labels{datasetIdx}(labelIdx).labels(regionIdx).ID);
            regions = [regions, region_data];
            fprintf('region %d\n', regionIdx);
        end
        datasets(end).region_centroids = [datasets(end).region_centroids; cell2mat({regions.centroid}')];
        datasets(end).num_regions = length(datasets(end).region_centroids);
        image_shape = size(image_data{datasetIdx});
        mask = zeros(size(image_data{datasetIdx}, 1), size(image_data{datasetIdx}, 2));
        for i = 1:length(regions)
            mask = mask + poly2mask(regions(i).vertices(:,1), regions(i).vertices(:,2), size(mask,1), size(mask,2));
        end
        mask = mask > 0;
        datasets(end).labels = cat(3, datasets(end).labels, uint8(mask));
        analysis = bwconncomp(datasets(end).labels(:, :, end), 4);
        datasets(end).analysis = [datasets(end).analysis, analysis];
    end
end
num_datasets = length(datasets);
disp("num of datasets");
disp(num_datasets);


% This will store all of our data for all datasets and their components which consist of the data split into
% training, validation and testing sets
augmentation_data = {};

% for each dataset setup a dictionary data structure to create test, train and validate components of the dataset
% each component will have a list of indices indicating the index of regions/segmented parts of the original image
% that will be used in each of the dataset components. Subsequent code will populate the 'region centroids',
% image data ('data'), and label data ('labels') within this datastructure for training.
for datasetIdx = 1:num_datasets
    dataset_components.train = struct('indices', [], 'num_region_idxs', 0, 'region_centroids', [], 'num_regions', 0, 'data', [], 'labels', []);
    dataset_components.test = struct('indices', [], 'num_region_idxs', 0, 'region_centroids', [], 'num_regions', 0, 'data', [], 'labels', []);
    dataset_components.validate = struct('indices', [], 'num_region_idxs', 0, 'region_centroids', [], 'num_regions', 0, 'data', [], 'labels', []);
    augmentation_data{end+1} = dataset_components;
end


% for each dataset select the regions/data vectors to put into the training, validation and testing sets
% by storing the indices of these regions and the number of indices/regions within each of these sets
for datasetIdx = 1:num_datasets
    dataset_components = augmentation_data{datasetIdx};
    datavectorIndices = 1:datasets(datasetIdx).num_regions;
    [training_indices, testing_indices] = split_sample(datavectorIndices, pct_test);
    [training_indices, validation_indices] = split_sample(training_indices, pct_val);
    dataset_components.train.indices = training_indices;
    dataset_components.test.indices = testing_indices;
    dataset_components.validate.indices = validation_indices;
    dataset_components.train.num_region_idxs = length(training_indices);
    dataset_components.test.num_region_idxs = length(testing_indices);
    dataset_components.validate.num_region_idxs = length(validation_indices);
    augmentation_data{datasetIdx} = dataset_components;
end

disp("here1");

% for each dataset visit the training, validation and testing sets and set the region centroids for all the regions
% that are associated with each of these sets
for datasetIdx = 1:num_datasets
    dataset_components = augmentation_data{datasetIdx};
    fn = fieldnames(dataset_components);
    for i = 1:numel(fn)
        component = fn{i};
        dataset = dataset_components.(component);
        regionCentroidArray = dataset.region_centroids;
        for localRegionIndex = 1:dataset.num_region_idxs
            globalRegionIndex = dataset.indices(localRegionIndex);
%             [totalLabels, label_img, regionStats, regionCentroids] = datasets(datasetIdx).analysis;
            regionCentroidArray = [regionCentroidArray; datasets(datasetIdx).region_centroids(globalRegionIndex, :)];
        end
        dataset_components.(component).num_regions = size(regionCentroidArray, 1);
        dataset_components.(component).region_centroids = regionCentroidArray;
    end
    augmentation_data{datasetIdx} = dataset_components;
end
disp("here2");


% iterate through the available datasets
for datasetIdx = 1:num_datasets
    dataset_components = augmentation_data{datasetIdx};
    dataset_testRegionCentroidArray = dataset_components.test.region_centroids;
    dataset_numTestRegions = length(dataset_testRegionCentroidArray);

    % iterate through test, train and validate components of dataset
    fields = fieldnames(dataset_components);
    for j = 1:numel(fields)
        dataset = fields{j};
        dataArray = dataset_components.(dataset).data;
        labelArray = dataset_components.(dataset).labels;
        [rows, cols] = size(datasets(datasetIdx).data);
        for augmentationIdx = 1:NUM_RANDOM_AUGMENTATIONS
            angle = rand()*359.9;
            center_y = rand()*rows;
            center_x = rand()*cols;
            aug_image_center = [center_x, center_y];

            skip_this_image = false;
            % do not test for proximity when augmenting the testing dataset
            if ~strcmp(dataset, 'test')
                for testRegionIdx = 1:dataset_numTestRegions
                    aug_image_center_to_testRegion_vector = dataset_testRegionCentroidArray(testRegionIdx,:) - aug_image_center;
                    train_to_test_Distance = norm(aug_image_center_to_testRegion_vector);
                    if train_to_test_Distance < 1.5 * IMAGE_SIZE * sqrt(2) + 21
                        disp(['Skip augmentation at image center (' num2str(aug_image_center_to_testRegion_vector) ')' ...
                            ' distance to test set = ' num2str(train_to_test_Distance)]);
                        skip_this_image = true;
                        continue;
                    end
                end
            end
            if skip_this_image
                continue;
            end

            aug_image_patch = getRigidImagePatch(mat2gray(datasets(datasetIdx).data), ...
                IMAGE_SIZE, IMAGE_SIZE, center_y, center_x, angle);
            aug_image_patch_hs = getRigidImagePatch(datasets(datasetIdx).data_hs, ...
                IMAGE_SIZE, IMAGE_SIZE, center_y, center_x, angle);
            aug_mask_patch = getRigidImagePatch(datasets(datasetIdx).labels, ...
                IMAGE_SIZE, IMAGE_SIZE, center_y, center_x, angle);
            if ~isempty(aug_image_patch)
                aug_image_patch = (aug_image_patch - min(aug_image_patch(:))) / (max(aug_image_patch(:)) - min(aug_image_patch(:)));
%                 aug_image_patch = im2uint8(aug_image_patch);
                dataArray{end+1} = aug_image_patch;
                labelArray{end+1} = aug_mask_patch;
                augmentation_data{datasetIdx}.(dataset).data{end+1} = aug_image_patch;
                augmentation_data{datasetIdx}.(dataset).labels{end+1} = aug_mask_patch;
            end
        end
        disp('here2*');
        
        for regionIndex = 1:dataset_components.(dataset).num_regions
            for augmentationIdx = 1:NUM_AUGMENTATIONS_PER_LABELED_REGION
                % randomly perturb the orientation
                angle = rand() * 359.9;
                center_x = dataset_components.(dataset).region_centroids(regionIndex, 1);
                center_y = dataset_components.(dataset).region_centroids(regionIndex, 2);
                % randomly perturb the offset of the region within the tile
                dx = rand() * (70 + 70) - 70;
                dy = rand() * (70 + 70) - 70;
                aug_image_center = [center_x + dx, center_y + dy];
                skip_this_image = false;

                % do not test for proximity when augmenting the testing dataset
                if ~strcmp(dataset, 'test')
                    for testRegionIdx = 1:dataset_numTestRegions
                        aug_image_center_to_testRegion_vector = dataset_testRegionCentroidArray(testRegionIdx, :) - aug_image_center;
                        train_to_test_Distance = norm(aug_image_center_to_testRegion_vector);
                        if train_to_test_Distance < 1.5 * IMAGE_SIZE * sqrt(2) + 21
                            fprintf("Skip augmentation at image center (%s) distance to test set = %s\n", num2str(aug_image_center_to_testRegion_vector), num2str(train_to_test_Distance));
                            skip_this_image = true;
                            continue;
                        end
                    end
                end

                % if the augmentation may include data from the test set skip this augmentation
                % this may occur when labels of the test set are in the vicinity of labels from the training or
                % validation set
                if skip_this_image
                    continue;
                end

                aug_image_patch = getRigidImagePatch(mat2gray(datasets(datasetIdx).data), IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle);
                aug_image_patch_hs = getRigidImagePatch(datasets(datasetIdx).data_hs, IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle);
                aug_mask_patch = getRigidImagePatch(datasets(datasetIdx).labels, IMAGE_SIZE, IMAGE_SIZE, center_y + dy, center_x + dx, angle);

                % if the augmentation was successful add it to the image augmentation dataset
                if ~isempty(aug_image_patch)
                    aug_image_patch = (aug_image_patch - min(aug_image_patch(:))) / (max(aug_image_patch(:)) - min(aug_image_patch(:)));
%                     aug_image_patch = im2uint8(aug_image_patch);
                    dataArray{end+1} = aug_image_patch;
                    labelArray{end+1} = aug_mask_patch;
                    augmentation_data{datasetIdx}.(dataset).data{end+1} = aug_image_patch;
                    augmentation_data{datasetIdx}.(dataset).labels{end+1} = aug_mask_patch;
                end
            end
        end
        
    end

end

disp("here3");
% Form the training, testing and validation datasets from available labeled image data
train_images = [augmentation_data{1}.train.data, ...
                augmentation_data{2}.train.data, ...
                augmentation_data{3}.train.data];
train_labels = [augmentation_data{1}.train.labels, ...
                augmentation_data{2}.train.labels, ...
                augmentation_data{3}.train.labels];
validate_images = [augmentation_data{1}.validate.data, ...
                   augmentation_data{2}.validate.data, ...
                   augmentation_data{3}.validate.data];
validate_labels = [augmentation_data{1}.validate.labels, ...
                           augmentation_data{2}.validate.labels, ...
                           augmentation_data{3}.validate.labels];
test_images = [augmentation_data{1}.test.data, ...
                       augmentation_data{2}.test.data, ...
                       augmentation_data{3}.test.data];
test_labels = [augmentation_data{1}.test.labels, ...
                       augmentation_data{2}.test.labels, ...
                       augmentation_data{3}.test.labels];

disp("here4");

% print out the number of training vectors, validation vectors, and test vectors
fprintf('Train %d\n', size(train_images, 2));
fprintf('Validation %d\n', size(validate_images, 2));
fprintf('Test %d\n', size(test_images, 2));

count_images_with_object(train_images, train_labels)

count_images_with_object(test_images, test_labels)
count_images_with_object(validate_images, validate_labels)



save_images_and_masks(train_images, test_images, validate_images, train_labels, test_labels, validate_labels, output_folder_name);
disp("here4");

% Create the output directory of json file if they don't exist
mkdir(fullfile(output_folder_name, 'annotations', 'train'));
mkdir(fullfile(output_folder_name, 'annotations', 'test'));
mkdir(fullfile(output_folder_name, 'annotations', 'val'));
%Create a JSON annotation file for all the tiles using the COCO annotation format

imageDir1 = fullfile(output_folder_name, 'images', 'train');
imageDir2 = fullfile(output_folder_name, 'images', 'test');
imageDir3 = fullfile(output_folder_name, 'images', 'val');
maskDir1 = fullfile(output_folder_name, 'masks', 'train');
maskDir2 = fullfile(output_folder_name, 'masks', 'test');
maskDir3 = fullfile(output_folder_name, 'masks', 'val');


for dirIdx = 1:3
    annotation = struct('images', [], 'annotation', []);
    imageId = 1;
    annotationId = 1;
    if dirIdx == 1
        imageDir = imageDir1;
        maskDir = maskDir1;
        annotationFileName = fullfile(output_folder_name, 'annotations', 'train', 'annotations.json');
    elseif dirIdx == 2
        imageDir = imageDir2;
        maskDir = maskDir2;
        annotationFileName = fullfile(output_folder_name, 'annotations', 'test', 'annotations.json');
    else
        imageDir = imageDir3;
        maskDir = maskDir3;
        annotationFileName = fullfile(output_folder_name, 'annotations', 'val', 'annotations.json');
    end
    
      
    % Get a list of image files in the directory
    imageFiles = dir(fullfile(imageDir, '*.png'));

    % Loop over the image files
    for i = 1:numel(imageFiles)

        % Add image information to annotation
        %filename = imageFiles(i).name;
        filename =  sprintf('image_%d.png', i);

        % Add annotation information to annotation
        annotation.images(imageId).id = imageId;
        annotation.images(imageId).file_name = filename;
        annotation.images(imageId).width = IMAGE_SIZE;
        annotation.images(imageId).height = IMAGE_SIZE;
        annotation.images(imageId).license = 0;
        annotation.images(imageId).flickr_url = '';
        annotation.images(imageId).coco_url = '';
        annotation.images(imageId).date_captured = datestr(now, 'yyyy-mm-dd HH:MM:SS');

        % Add annotation information to annotation
        tileMask = imread(fullfile(maskDir, sprintf('image_%d.png', i)));
        mask = logical(tileMask);

        
        % Get connected components in the mask
        cc = bwconncomp(mask);
        props = regionprops(cc);

        for j = 1:length(props)
            % Create a bounding box for each object
            bbox = props(j).BoundingBox;

            % Convert from [x,y,width,height] format to [x1,y1,x2,y2] format
            x1 = bbox(1);
            y1 = bbox(2);
            x2 = x1 + bbox(3) - 1;
            y2 = y1 + bbox(4) - 1;

            % Convert from MATLAB indexing to COCO indexing
            x1 = x1 - 1;
            y1 = y1 - 1;

            % Add annotation information to annotation
            annotation.annotations(annotationId).id = annotationId;
            annotation.annotations(annotationId).image_id = imageId ;
            annotation.annotations(annotationId).category_id = 1; % Assume only one category for now
            annotation.annotations(annotationId).area = props(j).Area;
            annotation.annotations(annotationId).bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1];
            annotation.annotations(annotationId).iscrowd = 0;
            annotation.annotations(annotationId).segmentation = [];

            annotationId = annotationId + 1;
        end
        imageId = imageId + 1;

    end
    % Write the JSON annotations to a file with pretty printing
    jsonStr = jsonencode(annotation);
    jsonStr = strrep(jsonStr, ',', sprintf(',\n'));
    jsonStr = strrep(jsonStr, '{', sprintf('{\n'));
    jsonStr = strrep(jsonStr, '}', sprintf('\n}'));
    fid = fopen(annotationFileName, 'wt');
    fprintf(fid, '%s\n', jsonStr);
    fclose(fid);

end

function save_images_and_masks(train_images, test_images, validate_images, train_labels, test_labels, validate_labels, output_directory)

% Create the output directories if they don't exist
mkdir(fullfile(output_directory, 'images', 'train'));
mkdir(fullfile(output_directory, 'images', 'val'));
mkdir(fullfile(output_directory, 'images', 'test'));
mkdir(fullfile(output_directory, 'masks', 'train'));
mkdir(fullfile(output_directory, 'masks', 'val'));
mkdir(fullfile(output_directory, 'masks', 'test'));
disp("directory created");
% Save the train images and labels
for i = 1:numel(train_images)    
    imwrite(train_images{i}, fullfile(output_directory, 'images', 'train', sprintf('image_%d.png', i)));
    imwrite(logical(train_labels{i}), fullfile(output_directory, 'masks', 'train', sprintf('image_%d.png', i)));
end

% Save the validation images and labels
for i = 1:numel(validate_images)
    imwrite(validate_images{i}, fullfile(output_directory, 'images', 'val', sprintf('image_%d.png', i)));
    imwrite(logical(validate_labels{i}), fullfile(output_directory, 'masks', 'val', sprintf('image_%d.png', i)));
end

% Save the test images and labels
for i = 1:numel(test_images)
    imwrite(test_images{i}, fullfile(output_directory, 'images', 'test', sprintf('image_%d.png', i)));
    imwrite(logical(test_labels{i}), fullfile(output_directory, 'masks', 'test', sprintf('image_%d.png', i)));
end
disp("images saving done!")

end
function count_images_with_object(images, masks)
    total_images = length(images);
    images_with_object = 0;
    
    for i = 1:total_images
        if sum(masks{i}(:)) > 0
            images_with_object = images_with_object + 1;
        end
    end
    
    fprintf("Total images: %d\n", total_images);
    fprintf("Images with object: %d\n", images_with_object);
end
function [image_patch_aug] = getRigidImagePatch(img, height, width, center_y, center_x, angle)
    height = height-1;
    width = width -1;
    theta = (angle / 180) * pi;
    xy_center = [width / 2, height / 2];
    cos_t = cos(theta);
    sin_t = sin(theta);
    bound_w = int32(height * abs(sin_t) + width * abs(cos_t));
    bound_h = int32(height * abs(cos_t) + width * abs(sin_t));
    xy_start = int32([floor(center_x - bound_w / 2), floor(center_y - bound_h / 2)]);
    xy_end = int32([ceil(center_x + bound_w / 2), ceil(center_y + bound_h / 2)]);

    if any(xy_start < 1) || any(xy_end > [size(img, 2), size(img, 1)])
        % Indices are not valid, return empty image
        image_patch_aug = [];
        return
    end

    cropped_image_patch = img(xy_start(2):xy_end(2), xy_start(1):xy_end(1), :);
    cropped_height = size(cropped_image_patch, 1);
    cropped_width = size(cropped_image_patch, 2);

    xy_translation = [0.5 * (cropped_width - (cos_t * cropped_width + sin_t * cropped_height));                      0.5 * (cropped_height - (-sin_t * cropped_width + cos_t * cropped_height))];
    image_patch_T = [cos_t, sin_t, xy_translation(1); -sin_t, cos_t, xy_translation(2); 0, 0, 1];
    T = affine2d(image_patch_T');
    transformed_image_patch = imwarp(cropped_image_patch, T, 'OutputView', imref2d(size(cropped_image_patch)));

    xy_center_newimg = int32(size(transformed_image_patch) / 2.0);
    xy_start = int32([xy_center_newimg(1) - width / 2, xy_center_newimg(2) - height / 2]);
    xy_end = int32([xy_center_newimg(1) + width / 2, xy_center_newimg(2) + height / 2]);
    image_patch_aug = transformed_image_patch(xy_start(2):xy_end(2), xy_start(1):xy_end(1), :);
end
function [train_idx, test_idx] = split_sample(data_idx, test_ratio)
    % randomly split data indices into train and test indices based on a given test ratio
    % data_idx: a vector of indices to be split
    % test_ratio: the ratio of test samples to all samples
    
    num_data = length(data_idx);
    test_size = round(num_data * test_ratio);
    shuffled_idx = randperm(num_data);
    
    train_idx = data_idx(shuffled_idx(test_size+1:end));
    test_idx = data_idx(shuffled_idx(1:test_size));
end
