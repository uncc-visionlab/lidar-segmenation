clear;

DATASETINDEX=4;
hold off;
close all;
switch DATASETINDEX
    case 1
        gis_geotiff_filename = 'KOM/raw/kom_dsm_lidar.tif';
        gis_output_hillshade_filename = 'KOM/raw/kom_dsm_lidar_hs.png';
        gis_output_gt_filename = 'KOM/raw/kom_dsm_lidar_gt.png';
        matlab_data_filename = 'KOM_image_data.mat';
        matlab_gt_labels_all_filename = 'KOM_ground_truth_labels.mat';
        classified_labels_filename = '../results/KOM_image_classified.png';
        fused_output_filename = '../results/KOM_image_fused.tif';
        
    case 2
        gis_geotiff_filename = 'MLS/raw/MLS_DEM.tif';
        gis_output_hillshade_filename = 'MLS/raw/MLS_DEM_hs.png';
        gis_output_gt_filename = 'MLS/raw/MLS_DEM_gt.png';
        matlab_data_filename = 'MLS_image_data.mat';
        matlab_gt_labels_all_filename = 'MLS_ground_truth_labels.mat';
        classified_labels_filename = '../results/MLS_image_classified.png';
        fused_output_filename = '../results/MLS_image_fused.tif';
        
    case 3
        gis_geotiff_filename = 'UCB/raw/UCB_elev_adjusted.tif';
        gis_output_hillshade_filename = 'UCB/raw/UCB_elev_adjusted_hs.png';
        gis_output_gt_filename = 'UCB/raw/UCB_elev_adjusted_gt.png';
        matlab_data_filename = 'UCB_image_data.mat';
        matlab_gt_labels_all_filename = 'UCB_ground_truth_labels.mat';
        classified_labels_filename = '../results/UCB_image_classified.png';
        fused_output_filename = '../results/UCB_image_fused.tif';
        
    case 4
        gis_output_hillshade_filename = 'Sayil/Sayil_regional_DEM_hs.png';
        classified_labels_filename = '../results/Sayil_image_classified.png';
        fused_output_filename = '../results/Sayil_image_fused.tif';

    otherwise
        printf(1,"Error\n");
        return;
end

I_hs=imread(gis_output_hillshade_filename);
[rows, cols] = size(I_hs);
I_labels=imread(classified_labels_filename);

if DATASETINDEX==4
    I_labels=I_labels';
    %I_gt_labels = I_hs;
    I_gt_labels = zeros(rows, cols,'uint8');
else
    gt_regions=load(matlab_gt_labels_all_filename);
    I_gt_labels = zeros(rows, cols,'uint8');
    for regionIdx=1:length(gt_regions.all_labels(1).labels)
        vertices = gt_regions.all_labels(1).labels(regionIdx).vertices;
        I_region_mask =  poly2mask( vertices(:,1), vertices(:,2), rows, cols);        
        I_gt_labels(I_region_mask==1) = 255;
    end
end

color_image = zeros(rows, cols, 3);
color_image(:,:,1) = I_hs - I_gt_labels/5;
color_image(:,:,2) = I_hs - I_labels/5;
color_image(:,:,3) = I_hs;
% configure tiff
tiff_output = Tiff(fused_output_filename,'w8');
setTag(tiff_output,'ImageLength',rows);
setTag(tiff_output,'ImageWidth',cols);
setTag(tiff_output,'Photometric',Tiff.Photometric.RGB);
setTag(tiff_output,'PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
setTag(tiff_output,'BitsPerSample',8);
setTag(tiff_output,'SamplesPerPixel',3);
setTag(tiff_output,'Compression',Tiff.Compression.LZW);
% write data
write(tiff_output,uint8(color_image));
close(tiff_output);

%imwrite(color_image, fused_output_filename);
