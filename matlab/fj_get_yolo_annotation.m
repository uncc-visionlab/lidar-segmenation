clear;
clc;

hold off;
close all;

DATASETINDEX = 1;

switch DATASETINDEX
    case 1
        json_file = 'YOLO_data/annular_structure/annotations/train/annotations.json'; 
        output_folder_name = 'YOLO_data/annular_structure/labels/train/';
        
    case 2
        json_file = 'YOLO_data/annular_structure/annotations/test/annotations.json'; 
        output_folder_name = 'YOLO_data/annular_structure/labels/test/';
    case 3
        json_file = 'YOLO_data/annular_structure/annotations/val/annotations.json'; 
        output_folder_name = 'YOLO_data/annular_structure/labels/val/';
     
    
    otherwise
        printf(1,"Error\n");
        return;
end


create_directory_if_not_exist(output_folder_name);
convert_coco_to_yolo(json_file, output_folder_name);
disp("Done Saving!");

function convert_coco_to_yolo(json_file, annotation_dir)
% Load JSON file
data = jsondecode(fileread(json_file));

% Loop through images and create annotation text file for each
for i = 1:length(data.images)
    img_id = data.images(i).id;
    file_name = data.images(i).file_name;
    img_width = data.images(i).width;
    img_height = data.images(i).height;
    
    % Get the list of bboxes for this image
    bbox_list = [];
    for j = 1:length(data.annotations)
        if data.annotations(j).image_id == img_id
            bbox = data.annotations(j).bbox;
            if ~isempty(bbox) && bbox(3) > 0 && bbox(4) > 0
                bbox = bbox';
                bbox_list = [bbox_list; bbox];
            end
        end
    end
    
    % Create annotation text file for this image
    if ~isempty(bbox_list)
%         bbox_list = [bbox_list; bbox_list];
        txt_file_name = fullfile(annotation_dir, strrep(file_name, '.png', '.txt'));
        fid = fopen(txt_file_name, 'wt');
        for j = 1:size(bbox_list, 1)
            bbox = bbox_list(j, :);
            x_center = bbox(1) + bbox(3)/2;
            y_center = bbox(2) + bbox(4)/2;
            x_center = x_center / img_width;
            y_center = y_center / img_height;
            width = bbox(3) / img_width;
            height = bbox(4) / img_height;
            fprintf(fid, '0 %.6f %.6f %.6f %.6f\n', x_center, y_center, width, height);
        end
        fclose(fid);
    else
        txt_file_name = fullfile(annotation_dir, strrep(file_name, '.png', '.txt'));
        fid = fopen(txt_file_name, 'wt');
        fclose(fid);
    end
end
end

function create_directory_if_not_exist(directory_name)
% CREATE_DIRECTORY_IF_NOT_EXIST Creates a directory if it does not already exist
%   CREATE_DIRECTORY_IF_NOT_EXIST(DIRECTORY_NAME) creates a directory with the
%   specified name if it does not already exist.

if ~exist(directory_name, 'dir')
    mkdir(directory_name);
end
end
