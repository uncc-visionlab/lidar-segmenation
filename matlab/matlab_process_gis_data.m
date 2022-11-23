clear;
clc;

% For DATASETINDEX=1 it seems like region index 35 may be incorrect - ask Dr. Ringle
%
% For DATASETINDEX=2 it seems like regions with indices {11} may be off-center
% and regions with indices {41,49} may be incorrect - ask Dr. Ringle
% MLS region 11 for annular structure is not useable (on boundary)

DATASETINDEX = 1;
PLOT_REGION_MESH = false;
%PLOT_REGION_MESH = true;
IGNORE_POLYGONS = true;
MAX_INTENSITY = 255;

%
% Interactions
%
INTERACTIVE_ANNULAR_REGION_SPECIFICATION = false;
INTERACTIVE_PLATFORM_REGION_SPECIFICATION = true;
INTERACTION_LABEL_INDICES = 10:1000;

uiDoneFlag = false;

region(1).Name = 'Annular structure';
region(1).WINDOWSIZE = 40;
region(1).Color = [1 .8 .8]; % light red
region(1).LabelValue = 1;
region(2).Name = 'Platform';
region(2).WINDOWSIZE = 80;
region(2).Color = [.67 .84 .9]; % light blue
region(2).LabelValue = 2;

NUMDATASETS = 1;

for datasetIdx=NUMDATASETS:NUMDATASETS
    DATASETINDEX=datasetIdx;
    hold off;
    close all;
    switch DATASETINDEX
        case 1
            gis_geotiff_filename = 'KOM/raw/kom_dsm_lidar.tif';
            gis_esri_shapefilenames = {'KOM/raw/Kom_Annular_strs.shp','KOM/raw/Kom_platforms.shp'};
            %gis_esri_shapefilenames = {'KOM/raw/Kom_Annular_strs.shp','KOM/raw/Kom_AI_platforms.shp'};
            %gis_esri_shapefilenames = {'KOM/raw/Kom_Annular_strs.shp'};
            %gis_esri_shapefilenames = {'KOM/raw/Kom_Annular_strs.shp'};
            gis_output_filename = 'KOM/raw/kom_dsm_lidar.png';
            gis_output_hillshade_filename = 'KOM/raw/kom_dsm_lidar_hs.png';
            gis_output_gt_filename = 'KOM/raw/kom_dsm_lidar_gt.png';
            gt_labels_annular_filename = 'ground_truth_annular_structure_labels_kom.mat';
            gt_labels_platform_filename = 'ground_truth_platform_labels_kom.mat';            
        case 2
            gis_geotiff_filename = 'MLS/raw/MLS_DEM.tif';
            gis_esri_shapefilenames = {'MLS/raw/MLS_Annular_strs.shp'};
            gis_output_filename = 'MLS/raw/MLS_DEM.png';
            gis_output_hillshade_filename = 'MLS/raw/MLS_DEM_hs.png';
            gis_output_gt_filename = 'MLS/raw/MLS_DEM_gt.png';
            gt_labels_annular_filename = 'ground_truth_annular_structure_labels_mls.mat';
            gt_labels_platform_filename = 'ground_truth_platform_labels_mls.mat';            
            
        case 3
            gis_geotiff_filename = 'UCB/raw/UCB_elev_adjusted.tif';
            gis_esri_shapefilenames = {'UCB/raw/UBM_anulares.shp'};
            gis_output_filename = 'UCB/raw/UCB_elev_adjusted.png';
            gis_output_hillshade_filename = 'UCB/raw/UCB_elev_adjusted_hs.png';
            gis_output_gt_filename = 'UCB/raw/UCB_elev_adjusted_gt.png';
            gt_labels_annular_filename = 'ground_truth_annular_structure_labels_ucb.mat';
            gt_labels_platform_filename = 'ground_truth_platform_labels_ucb.mat';            
            
        otherwise
            printf(1,"Error\n");
            return;
    end
    
    geotiff_info = geotiffinfo(gis_geotiff_filename);
    geotiff_data = readgeoraster(gis_geotiff_filename);
    
    if (strcmp(gis_geotiff_filename,'MLS/raw/MLS_DEM.tif') == 1 || ...
            strcmp(gis_geotiff_filename,'KOM/raw/kom_dsm_lidar.tif') == 1)
        bad_pixel_values = max(geotiff_data(:));
        artificial_min_value = min(geotiff_data(:))-0.1;
        geotiff_data(geotiff_data==bad_pixel_values)=artificial_min_value;
    end
    
    % normalize the elevation data to the 0-MAX_INTENSITY intensity range
    minValue = min(geotiff_data(:));
    maxValue = max(geotiff_data(:));
    range = maxValue - minValue;
    image_geo_output = uint8(MAX_INTENSITY*(geotiff_data-minValue)/range);
    imwrite(image_geo_output, gis_output_filename);

    % generate a hillshade image with a normalized 0-MAX_INTENSITY intensity range
    x_hs=(1:size(geotiff_data, 1))';
    y_hs=1:size(geotiff_data, 2);  
    hillshade_image=hillshade_esri(geotiff_data, x_hs, y_hs);
    minValue = min(hillshade_image(:));
    maxValue = max(hillshade_image(:));
    range = maxValue - minValue;
    image_geo_hillshade_output = uint8(MAX_INTENSITY*(hillshade_image-minValue)/range);
    figure(4), imshow(image_geo_hillshade_output);
    imwrite(image_geo_hillshade_output, gis_output_hillshade_filename);

    image_geo_ground_truth = zeros(size(image_geo_output));
    
    image_size=size(geotiff_data);
    x0 = geotiff_info.SpatialRef.XWorldLimits(1);
    y0 = geotiff_info.SpatialRef.YWorldLimits(2);
    range_x = geotiff_info.SpatialRef.XWorldLimits(2) - geotiff_info.SpatialRef.XWorldLimits(1);
    range_y = geotiff_info.SpatialRef.YWorldLimits(2) - geotiff_info.SpatialRef.YWorldLimits(1);
    figure(1), imshow(geotiff_data,[])

   
    for shapefileIndex=1:length(gis_esri_shapefilenames)
        gis_esri_shapefilename = gis_esri_shapefilenames{shapefileIndex};
        shapefile_structure = shapeinfo(gis_esri_shapefilename);
        shapefile_data = shaperead(gis_esri_shapefilename);
        shp_range_x = shapefile_structure(1).BoundingBox(2,1) - shapefile_structure(1).BoundingBox(1,1);
        shp_range_y = shapefile_structure(1).BoundingBox(2,2) - shapefile_structure(1).BoundingBox(1,2);
        
        WINDOWSIZE = region(shapefileIndex).WINDOWSIZE;
        
        if (shapefileIndex == 1 && INTERACTIVE_ANNULAR_REGION_SPECIFICATION)
            strVal = sprintf('You will need to provide ellipse-shaped object regions for each annular structure in this dataset.');
            message = {strVal ...
                'Draw an ellipse that encompasses the annular structure region then' ...
                'press ESC after each region is specified to move to the next region'};
            waitfor(msgbox(message, 'Select Training Data','help'))
            if isfile(gt_labels_annular_filename)
                load(gt_labels_annular_filename);
            end
            if (exist('labelInfo','var') == 0) 
                labelInfo = cell(length(shapefile_data),1);
            end
        elseif (shapefileIndex == 1 && INTERACTIVE_PLATFORM_REGION_SPECIFICATION)
            if isfile(gt_labels_platform_filename)
                load(gt_labels_platform_filename);
            end
            if (exist('labelInfo','var') == 0) 
                labelInfo = cell(length(shapefile_data),1);
            end
        end
        
        for regionIdx=1:length(shapefile_data)
            coords_x = image_size(2)*(shapefile_data(regionIdx).X - x0)./range_x;
            coords_x(isnan(coords_x))=coords_x(1);
            coords_y = image_size(1)*(y0 - shapefile_data(regionIdx).Y)./range_y;
            coords_y(isnan(coords_y))=coords_y(1);
            boundingbox_vertices = [coords_x', coords_y'];
            xy_region_min = min(boundingbox_vertices,[],1);
            xy_region_max = max(boundingbox_vertices,[],1);
            xy_region_range = xy_region_max - xy_region_min;
            xy_region_center = mean(boundingbox_vertices,1);
            
            %
            % TODO: ARW
            % Change region shape to a circle consisting of at least 8
            % vertices (better 16)
            %
            if (prod(xy_region_range) == 0 || IGNORE_POLYGONS)
                xy_region_min = xy_region_center - WINDOWSIZE/2;
                xy_region_max = xy_region_center + WINDOWSIZE/2;
                xy_region_min(xy_region_min < 0) = 0;
                if (xy_region_max(1) > image_size(2))
                    xy_region_max(1) = image_size(2);
                end
                if (xy_region_max(2) > image_size(1))
                    xy_region_max(2) = image_size(1);
                end
                xy_region_range = [WINDOWSIZE + 1, WINDOWSIZE + 1];
                boundingbox_vertices=[xy_region_min(1), xy_region_min(2);
                    xy_region_max(1), xy_region_min(2);
                    xy_region_max(1), xy_region_max(2);
                    xy_region_min(1), xy_region_max(2)];
            end
            
            figure(1), hold on, drawpolygon('Position', boundingbox_vertices, ...
                'MarkerSize', 5, ...
                'LineWidth',1,'FaceAlpha', 0.3, 'Color', region(shapefileIndex).Color, ...
                'SelectedColor', region(shapefileIndex).Color);
            figure(4), hold on, drawpolygon('Position', boundingbox_vertices, ...
                'MarkerSize', 5, ...
                'LineWidth',1,'FaceAlpha', 0.3, 'Color', region(shapefileIndex).Color, ...
                'SelectedColor', region(shapefileIndex).Color);
            
            if (PLOT_REGION_MESH)
                PLOT_MARGIN = 10;
                x_coord_list = (xy_region_min(1)-PLOT_MARGIN/2):(xy_region_max(1)+PLOT_MARGIN/2);
                x_coord_list(x_coord_list < 0) = [];
                x_coord_list(x_coord_list > image_size(2)) = [];
                y_coord_list = (xy_region_min(2)-PLOT_MARGIN/2):(xy_region_max(2)+PLOT_MARGIN/2);
                y_coord_list(y_coord_list < 0) = [];
                y_coord_list(y_coord_list > image_size(1)) = [];
                xx_vals = int32(x_coord_list);
                yy_vals = int32(y_coord_list);
                zz_vals = geotiff_data(yy_vals,xx_vals);
                [x_meshgrid, y_meshgrid] = meshgrid(xx_vals,yy_vals);
                titlestr = sprintf('%s region index %d', region(shapefileIndex).Name, regionIdx);
                figure(2), hold off, mesh(x_meshgrid, y_meshgrid, zz_vals), title(titlestr);
                %     figure(2), view(0,90)
                %     [new_centerpt_x, new_centerpt_y] = ginput(1);
                %     coords_x = coords_x - centerpt(1) + new_centerpt_x;
                %     coords_y = coords_y - centerpt(2) + new_centerpt_y;
                % save the new coordinates
                %pause(0.5);
            end
            
            if (INTERACTIVE_ANNULAR_REGION_SPECIFICATION && shapefileIndex == 1 && ...
                ~isempty(find(INTERACTION_LABEL_INDICES == regionIdx, 1)) && ~uiDoneFlag)
                PLOT_MARGIN = 40;
                x_coord_list = (xy_region_min(1)-PLOT_MARGIN/2):(xy_region_max(1)+PLOT_MARGIN/2);
                x_coord_list(x_coord_list <= 0) = [];
                x_coord_list(x_coord_list > image_size(2)) = [];
                y_coord_list = (xy_region_min(2)-PLOT_MARGIN/2):(xy_region_max(2)+PLOT_MARGIN/2);
                y_coord_list(y_coord_list <= 0) = [];
                y_coord_list(y_coord_list > image_size(1)) = [];
                xx_vals = int32(x_coord_list);
                yy_vals = int32(y_coord_list);
                zz_vals = geotiff_data(yy_vals,xx_vals);
                [x_meshgrid, y_meshgrid] = meshgrid(xx_vals,yy_vals);
                titlestr = sprintf('%s region index %d', region(shapefileIndex).Name, regionIdx);
                [K, H, Pmax, Pmin] = surfature(double(x_meshgrid), double(y_meshgrid), zz_vals);

                fig5_h = figure(5);
                figure(fig5_h);
                set(fig5_h, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
                fig5_sp122_h = subplot(1,2,2), mesh(x_meshgrid, y_meshgrid, zz_vals), title(titlestr);
                %view(0,90)
                %fig5_sp121_h = subplot(1,2,1), imshow(H, [], 'InitialMagnification', 'fit');
                fig5_sp121_h = subplot(1,2,1), imshow(hillshade_image(yy_vals,xx_vals), [], 'InitialMagnification', 'fit');
                
                if (exist('labelInfo','var') && regionIdx <= size(labelInfo,1) && isfield(labelInfo{regionIdx},'vertices'))
                    num_vertices = size(labelInfo{regionIdx}.vertices,1);

                    poly_origin = [x_coord_list(1), y_coord_list(1)];

                    win_center = labelInfo{regionIdx}.center - poly_origin;
                    win_vertices = labelInfo{regionIdx}.vertices - ones(num_vertices,1)*poly_origin;
                    regionOfInterest = drawpolygon('Position',win_vertices);
                    wait(regionOfInterest);
                    if ~ishandle(5)
                        uiDoneFlag = true;
                    else
                        num_vertices = size(regionOfInterest.Position,1);
                        labelInfo{regionIdx}.index = region(shapefileIndex).LabelValue;
                        labelInfo{regionIdx}.name = sprintf('%d',regionIdx);
                        labelInfo{regionIdx}.vertices = ones(num_vertices,1)*poly_origin + regionOfInterest.Position;
                    end
                else
                    regionOfInterest = drawellipse('Center',[(WINDOWSIZE + PLOT_MARGIN)/2, (WINDOWSIZE + PLOT_MARGIN)/2], ...
                        'SemiAxes',[WINDOWSIZE/1.5, WINDOWSIZE/1.5]);
                    wait(regionOfInterest);
                    if ~ishandle(5)
                        uiDoneFlag = true;
                    else
                        
                        poly_origin = [x_coord_list(1), y_coord_list(1)];
                        num_vertices = size(regionOfInterest.Vertices,1);
                        labelInfo{regionIdx}.index = region(shapefileIndex).LabelValue;
                        labelInfo{regionIdx}.name = sprintf('%d',regionIdx);
                        labelInfo{regionIdx}.center = poly_origin + regionOfInterest.Center;
                        labelInfo{regionIdx}.rotation_angle = regionOfInterest.RotationAngle;
                        labelInfo{regionIdx}.semi_axes = regionOfInterest.SemiAxes;
                        labelInfo{regionIdx}.vertices = ones(num_vertices,1)*poly_origin + regionOfInterest.Vertices;
                    end
                end
            end
            
            if (INTERACTIVE_PLATFORM_REGION_SPECIFICATION && shapefileIndex == 2 && ...
                    ~isempty(find(INTERACTION_LABEL_INDICES == regionIdx, 1)) && ~uiDoneFlag)
                PLOT_MARGIN = 40;
                x_coord_list = (xy_region_min(1)-PLOT_MARGIN/2):(xy_region_max(1)+PLOT_MARGIN/2);
                x_coord_list(x_coord_list <= 0) = [];
                x_coord_list(x_coord_list > image_size(2)) = [];
                y_coord_list = (xy_region_min(2)-PLOT_MARGIN/2):(xy_region_max(2)+PLOT_MARGIN/2);
                y_coord_list(y_coord_list <= 0) = [];
                y_coord_list(y_coord_list > image_size(1)) = [];
                xx_vals = int32(x_coord_list);
                yy_vals = int32(y_coord_list);
                zz_vals = geotiff_data(yy_vals,xx_vals);
                [x_meshgrid, y_meshgrid] = meshgrid(xx_vals,yy_vals);
                titlestr = sprintf('%s region index %d', region(shapefileIndex).Name, regionIdx);
                [K, H, Pmax, Pmin] = surfature(double(x_meshgrid), double(y_meshgrid), zz_vals);

                fig5_h = figure(5);
                figure(fig5_h);
                set(fig5_h, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
                fig5_sp122_h = subplot(1,2,2), mesh(x_meshgrid, y_meshgrid, zz_vals), title(titlestr);
                %view(0,90)
                %fig5_sp121_h = subplot(1,2,1), imshow(H, [], 'InitialMagnification', 'fit');
                %fig5_sp121_h = subplot(1,2,1), imagesc(H);
                fig5_sp121_h = subplot(1,2,1), imshow(hillshade_image(yy_vals,xx_vals), [], 'InitialMagnification', 'fit');
                % Create a push button
                %'Position',[420, 218, 100, 22],...
%                 uipanel_handle = uipanel(fig5_h)
%                 %uipanel_handle.uiDoneFlag = false;
%                 btn_next = uibutton(uipanel_handle,'push',...
%                     'ButtonPushedFcn', @(btn, event) nextButtonPushed(btn));
%                 btn_done = uibutton(uipanel_handle,'push',...
%                     'ButtonPushedFcn', @(btn, uiDoneFlag) doneButtonPushed(btn));
                poly_origin = [x_coord_list(1), y_coord_list(1)];
                if (exist('labelInfo','var') && regionIdx <= size(labelInfo,1) && isfield(labelInfo{regionIdx},'vertices'))
                    num_vertices = size(labelInfo{regionIdx}.vertices,1);

                    %win_center = labelInfo{regionIdx}.center - poly_origin;
                    win_vertices = labelInfo{regionIdx}.vertices - ones(num_vertices,1)*poly_origin;
                    regionOfInterest = drawpolygon('Position',win_vertices);
                else
                    num_vertices = size(boundingbox_vertices,1);
                    win_vertices = 0.75*(boundingbox_vertices - ones(num_vertices,1)*xy_region_center) + ones(4,1)*xy_region_center;
                    win_vertices = win_vertices - ones(num_vertices,1)*poly_origin;
                    regionOfInterest = drawpolygon('Position',win_vertices);
                end
                wait(regionOfInterest);
                if ~ishandle(5)
                    uiDoneFlag = true;
                else
                    num_vertices = size(regionOfInterest.Position,1);
                    labelInfo{regionIdx}.index = region(shapefileIndex).LabelValue;
                    labelInfo{regionIdx}.name = sprintf('%d',regionIdx);
                    labelInfo{regionIdx}.vertices = ones(num_vertices,1)*poly_origin + regionOfInterest.Position;
                end
            end    
            
            % create the ground truth labels for this region
            bw = poly2mask(boundingbox_vertices(:,1), boundingbox_vertices(:,2), image_size(1), image_size(2));
            % shortcoming - annular structure labels may be overwritten by
            % platform label values
            image_geo_ground_truth(bw==1) = region(shapefileIndex).LabelValue;
            
        end
    end
    image_geo_ground_truth = uint8(image_geo_ground_truth);
    %colorized_label_image = label2rgb(image_geo_ground_truth);
    %figure(3), imshow(colorized_label_image,[]);
    segmentation_image = labeloverlay(image_geo_output, image_geo_ground_truth);
    figure(3), imshow(segmentation_image);
    imwrite(image_geo_ground_truth, gis_output_gt_filename);   
    if (INTERACTIVE_ANNULAR_REGION_SPECIFICATION)
        save(gt_labels_annular_filename, 'labelInfo');
    end
    if (INTERACTIVE_PLATFORM_REGION_SPECIFICATION)
        save(gt_labels_platform_filename, 'labelInfo');
    end
    %pause;
end

% Create the function for the ButtonPushedFcn callback
function out = doneButtonPushed(btn)
    out = true;
end     

% Create the function for the ButtonPushedFcn callback
function out = nextButtonPushed(btn)
    out = true;
end                
