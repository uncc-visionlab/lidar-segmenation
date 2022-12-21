clear;
clc;

hold off;
close all;

DATASETINDEX = 2;
PLOT_REGION_MESH = false;
MAX_INTENSITY = 255;

region(1).Name = 'Annular structure';
region(1).WINDOWSIZE = 40;
region(1).Color = [1 .8 .8]; % light red
region(1).LabelValue = 1;
region(2).Name = 'Platform';
region(2).WINDOWSIZE = 80;
region(2).Color = [.67 .84 .9]; % light blue
region(2).LabelValue = 2;

switch DATASETINDEX
    case 1
        gis_geotiff_filename = 'KOM/raw/kom_dsm_lidar.tif';
        importData(1).filename = 'KOM/raw/Kom_Annular_strs.shp';
        importData(1).labelValue = 1;
        importData(2).filename = 'KOM/raw/Kom_Platforms_strs.shp';
        importData(2).labelValue = 1;
        gis_output_filename = 'KOM/raw/kom_dsm_lidar.png';
        gis_output_hillshade_filename = 'KOM/raw/kom_dsm_lidar_hs.png';
        gis_output_gt_filename = 'KOM/raw/kom_dsm_lidar_gt.png';
        matlab_data_filename = 'KOM_image_data.mat';
        matlab_gt_labels_all_filename = 'KOM_ground_truth_labels.mat';
        
    case 2
        gis_geotiff_filename = 'MLS/raw/MLS_DEM.tif';
        importData(1).filename = 'MLS/raw/MLS_Annular_strs.shp';
        importData(1).labelValue = 1;
        importData(2).filename = './MLS_platforms_all_outline.shp';
        importData(2).labelValue = 2;
        gis_output_filename = 'MLS/raw/MLS_DEM.png';
        gis_output_hillshade_filename = 'MLS/raw/MLS_DEM_hs.png';
        gis_output_gt_filename = 'MLS/raw/MLS_DEM_gt.png';
        matlab_data_filename = 'MLS_image_data.mat';
        matlab_gt_labels_all_filename = 'MLS_ground_truth_labels.mat'; % also change the exsiting gt mat file to be annulars
        
    case 3
        gis_geotiff_filename = 'UCB/raw/UCB_elev_adjusted.tif';
        importData(1).filename = 'UCB/raw/UBM_anulares.shp';
        importData(1).labelValue = 1;
        importData(2).filename = 'UCB/raw/UBM_platforms_outline.shp';
        importData(2).labelValue = 2;
        gis_output_filename = 'UCB/raw/UCB_elev_adjusted.png';
        gis_output_hillshade_filename = 'UCB/raw/UCB_elev_adjusted_hs.png';
        gis_output_gt_filename = 'UCB/raw/UCB_elev_adjusted_gt.png';
        matlab_data_filename = 'UCB_image_data.mat';
        matlab_gt_labels_all_filename = 'UCB_ground_truth_labels.mat';
        
    otherwise
        printf(1,"Error\n");
        return;
end

geotiff_info = geotiffinfo(gis_geotiff_filename);
geotiff_data = readgeoraster(gis_geotiff_filename);
image_size = size(geotiff_data);

if (strcmp(gis_geotiff_filename,'MLS/raw/MLS_DEM.tif') == 1 || ...
        strcmp(gis_geotiff_filename,'KOM/raw/kom_dsm_lidar.tif') == 1)
    bad_pixel_values = max(geotiff_data(:));
    artificial_min_value = min(geotiff_data(:))-0.1;
    geotiff_data(geotiff_data==bad_pixel_values)=artificial_min_value;
end
%save(matlab_data_filename, 'geotiff_data','-v7','-nocompression');

if isfile(matlab_gt_labels_all_filename)
    load(matlab_gt_labels_all_filename);
end


% normalize the elevation data to the 0-MAX_INTENSITY intensity range
minValue = min(geotiff_data(:));
maxValue = max(geotiff_data(:));
range = maxValue - minValue;
image_geo_output = uint8(MAX_INTENSITY*(geotiff_data-minValue)/range);
%figure(1), imshow(geotiff_data,[]);
%imwrite(image_geo_output, gis_output_filename);

% generate a hillshade image with a normalized 0-MAX_INTENSITY intensity range
x_hs=(1:size(geotiff_data, 1))';
y_hs=1:size(geotiff_data, 2);  
hillshade_image=hillshade_esri(geotiff_data, x_hs, y_hs);
minValue = min(hillshade_image(:));
maxValue = max(hillshade_image(:));
range = maxValue - minValue;
image_geo_hillshade_output = uint8(MAX_INTENSITY*(hillshade_image-minValue)/range);
figure(4), imshow(image_geo_hillshade_output);
%imwrite(image_geo_hillshade_output, gis_output_hillshade_filename);

% write a for loop to iterate two classes. If only one classes is
% considered, set the iteration range to accomendate.
for shapefileIndex=2:length(importData)     % 1 is annular structure; 2 is platform.
    gis_esri_shapefilename = importData(shapefileIndex).filename;
    shapefile_structure = shapeinfo(gis_esri_shapefilename);
    shapefile_data = shaperead(gis_esri_shapefilename);
    shp_range_x = shapefile_structure(1).BoundingBox(2,1) - shapefile_structure(1).BoundingBox(1,1);
    shp_range_y = shapefile_structure(1).BoundingBox(2,2) - shapefile_structure(1).BoundingBox(1,2);

    WINDOWSIZE = region(shapefileIndex).WINDOWSIZE;

    image_size=size(geotiff_data);
    x0 = geotiff_info.SpatialRef.XWorldLimits(1);
    y0 = geotiff_info.SpatialRef.YWorldLimits(2);
    range_x = geotiff_info.SpatialRef.XWorldLimits(2) - geotiff_info.SpatialRef.XWorldLimits(1);
    range_y = geotiff_info.SpatialRef.YWorldLimits(2) - geotiff_info.SpatialRef.YWorldLimits(1);



    if (exist('all_labels','var') == 0)
        labelInfo = struct('ID', {}, 'label_value', {}, 'label_name', {}, 'vertices', {}, 'center', {});
        newRegionIdx = 1;   % start to write data from the beginning if labelInfo is emtpy
    else
        labelInfo = all_labels(shapefileIndex).labels;
        newRegionIdx = length(shapefile_data)+1;  % append the data to the end
    end

    for regionIdx=1:length(shapefile_data) %110
        coords_x = image_size(2)*(shapefile_data(regionIdx).X - x0)./range_x;
        coords_x(isnan(coords_x))=coords_x(1);
        coords_y = image_size(1)*(y0 - shapefile_data(regionIdx).Y)./range_y;
        coords_y(isnan(coords_y))=coords_y(1);
        polygon_vertices = [coords_x', coords_y'];
        %polygon_vertices(any(isnan(polygon_vertices), 2), :) = [coords_x(1), coords_y(1)];
        xy_region_min = min(polygon_vertices,[],1);
        xy_region_max = max(polygon_vertices,[],1);
        xy_region_range = xy_region_max - xy_region_min;
        xy_region_center = mean(polygon_vertices,1);

        %xy_region_center_pixel_id = round(xy_region_center(1)/20)*image_size(1) + round(xy_region_center(2)/20);
        xy_region_center_pixel_id = shapefile_data(regionIdx).str_name;

    %    poly_origin = [x_coord_list(1), y_coord_list(1)];
    %     labelInfo{regionIdx}.ID = regionIdx;
    %     labelInfo{regionIdx}.index = region(shapefileIndex).LabelValue;
    %     labelInfo{regionIdx}.name = sprintf('%d',regionIdx);
    %     labelInfo{regionIdx}.vertices = ones(num_vertices,1)*poly_origin + regionOfInterest.Position;
    %     labelInfo{regionIdx}.center = poly_origin + mean(labelInfo{regionIdx}.vertices,1);


        % if region size is 0 --> this is a point feature  
        if (prod(xy_region_range) == 0)
            fprintf(1,'Cannot import shapefile with point features.\n');
            return;
        end

        matchFound = false;
        for matchedRegionIdx=1:length(labelInfo)
            %if (labelInfo(matchedRegionIdx).ID == xy_region_center_pixel_id)
            if (strcmp(labelInfo(matchedRegionIdx).ID, xy_region_center_pixel_id))
                fprintf(1,'Replaced region data.\n');
                labelInfo(regionIdx).ID = xy_region_center_pixel_id;
                labelInfo(regionIdx).label_value = region(shapefileIndex).LabelValue;
                labelInfo(regionIdx).label_name = region(shapefileIndex).Name;
                labelInfo(regionIdx).vertices = polygon_vertices;
                labelInfo(regionIdx).center = xy_region_center;
                matchFound = true;
                break;
            end
        end
        if (~matchFound)
            fprintf(1,'Added new region.\n');
            labelInfo(newRegionIdx).ID = xy_region_center_pixel_id;
            labelInfo(newRegionIdx).label_value = region(shapefileIndex).LabelValue;
            labelInfo(newRegionIdx).label_name = region(shapefileIndex).Name;
            labelInfo(newRegionIdx).vertices = polygon_vertices;
            labelInfo(newRegionIdx).center = xy_region_center;              
            newRegionIdx = newRegionIdx + 1;
        end


        %   'MarkerSize', 5, ...
    %     figure(1), hold on, drawpolygon('Position', polygon_vertices, ...
    %         'LineWidth',1,'FaceAlpha', 0.3, 'Color', region(shapefileIndex).Color, ...
    %         'SelectedColor', region(shapefileIndex).Color);
        %   'MarkerSize', 5, ...
        figure(4), hold on, drawpolygon('Position', polygon_vertices, ...
            'LineWidth',1,'FaceAlpha', 0.3, 'Color', region(shapefileIndex).Color, ...
            'SelectedColor', region(shapefileIndex).Color);

    %     if (PLOT_REGION_MESH)
    %         PLOT_MARGIN = 10;
    %         x_coord_list = (xy_region_min(1)-PLOT_MARGIN/2):(xy_region_max(1)+PLOT_MARGIN/2);
    %         x_coord_list(x_coord_list < 0) = [];
    %         x_coord_list(x_coord_list > image_size(2)) = [];
    %         y_coord_list = (xy_region_min(2)-PLOT_MARGIN/2):(xy_region_max(2)+PLOT_MARGIN/2);
    %         y_coord_list(y_coord_list < 0) = [];
    %         y_coord_list(y_coord_list > image_size(1)) = [];
    %         xx_vals = int32(x_coord_list);
    %         yy_vals = int32(y_coord_list);
    %         zz_vals = geotiff_data(yy_vals,xx_vals);
    %         [x_meshgrid, y_meshgrid] = meshgrid(xx_vals,yy_vals);
    %         titlestr = sprintf('%s region index %d', region(shapefileIndex).Name, regionIdx);
    %         figure(2), hold off, mesh(x_meshgrid, y_meshgrid, zz_vals), title(titlestr);
    %         %     figure(2), view(0,90)
    %         %     [new_centerpt_x, new_centerpt_y] = ginput(1);
    %         %     coords_x = coords_x - centerpt(1) + new_centerpt_x;
    %         %     coords_y = coords_y - centerpt(2) + new_centerpt_y;
    %         % save the new coordinates
    %         %pause(0.5);
    %     end    

        num_vertices = size(polygon_vertices,1);

        %labelInfo = all_labels(shapefileIndex).labels;

    end
end

all_labels(shapefileIndex).labels = labelInfo;
save(matlab_gt_labels_all_filename, 'all_labels','-v7','-nocompression');

