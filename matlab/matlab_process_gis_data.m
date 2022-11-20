clear;
clc;
hold off;
close all;

% For DATASETINDEX=1 it seems like region index 35 may be incorrect - ask Dr. Ringle
%
% For DATASETINDEX=2 it seems like regions with indices {11} may be off-center
% and regions with indices {41,49} may be incorrect - ask Dr. Ringle

DATASETINDEX = 1;
% PLOT_REGION_MESH = false;
PLOT_REGION_MESH = true;
IGNORE_POLYGONS = true;

region(1).Name = 'Annular structure';
region(1).WINDOWSIZE = 40;
region(2).Name = 'Platform';
region(2).WINDOWSIZE = 80;

%for datasetIdx=1:NUMDATASETS
%    DATASETINDEX=datasetIdx;

switch DATASETINDEX
    case 1
        gis_geotiff_filename = 'KOM/raw/kom_dsm_lidar.tif';
        gis_esri_shapefilenames = {'KOM/raw/Kom_Annular_strs.shp','KOM/raw/Kom_platforms.shp'};
        gis_output_filename = 'KOM/raw/kom_dsm_lidar.png';
        gis_output_gt_filename = 'KOM/raw/kom_dsm_lidar_gt.png';
        
    case 2
        gis_geotiff_filename = 'MLS/raw/MLS_DEM.tif';
        gis_esri_shapefilenames = {'MLS/raw/MLS_Annular_strs.shp'};
        gis_output_filename = 'MLS/raw/MLS_DEM.png';
        gis_output_gt_filename = 'MLS/raw/MLS_DEM_gt.png';

    case 3
        gis_geotiff_filename = 'UCB/raw/UCB_elev_adjusted.tif';
        gis_esri_shapefilenames = {'UCB/raw/UBM_anulares.shp'};
        gis_output_filename = 'UCB/raw/UCB_elev_adjusted.png';
        gis_output_gt_filename = 'UCB/raw/UCB_elev_adjusted_gt.png';
        
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
MAX_INTENSITY = 255;
minValue = min(geotiff_data(:));
maxValue = max(geotiff_data(:));
range = maxValue - minValue;
image_geo_output = uint8(MAX_INTENSITY*(geotiff_data-minValue)/(maxValue-minValue));
imwrite(image_geo_output,gis_output_filename);

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
    
    for regionIdx=1:length(shapefile_data)
        coords_x = image_size(2)*(shapefile_data(regionIdx).X - x0)./range_x;
        coords_x(isnan(coords_x))=coords_x(1);
        coords_y = image_size(1)*(y0 - shapefile_data(regionIdx).Y)./range_y;
        coords_y(isnan(coords_y))=coords_y(1);
        my_vertices = [coords_x', coords_y'];
        xy_region_min = min(my_vertices,[],1);
        xy_region_max = max(my_vertices,[],1);
        xy_region_range = xy_region_max - xy_region_min;
        xy_region_center = mean(my_vertices,1);
        
        if (prod(xy_region_range) == 0 || IGNORE_POLYGONS)
            xy_region_min = xy_region_center - WINDOWSIZE/2;
            xy_region_max = xy_region_center + WINDOWSIZE/2;
            xy_region_range = [WINDOWSIZE + 1, WINDOWSIZE + 1];
            my_vertices=[xy_region_min(1), xy_region_min(2);
                xy_region_max(1), xy_region_min(2);
                xy_region_max(1), xy_region_max(2);
                xy_region_min(1), xy_region_max(2)];
        end
        
        figure(1), hold on, drawpolygon('Position',my_vertices,'MarkerSize',1, ...
            'LineWidth',1,'FaceAlpha', 1, 'Color',[1 0 0],'SelectedColor','red');
        
        if (PLOT_REGION_MESH)
            PLOT_MARGIN = 0;
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
        
        % create the ground truth labels for this region
        bw = poly2mask(my_vertices(:,1), my_vertices(:,2), image_size(1), image_size(2));
        image_geo_ground_truth(bw==1) = shapefileIndex;
        
    end
end

%colorized_label_image = label2rgb(image_geo_ground_truth);
%figure(3), imshow(colorized_label_image,[]);
segmentation_image = labeloverlay(image_geo_output, image_geo_ground_truth);
figure(3), imshow(segmentation_image);
imwrite(image_geo_ground_truth, gis_output_gt_filename);

