clear; clc; close all

% Anna Roma
% Assignment 2 - Line Detection using GOLD Lane Detection Algorithm

% archive path : could change depending on where the archive is locally stored
archive_path = 'archive/044/camera/front_camera/*.jpg';
files = dir(archive_path);
num_files = length(files);
if isempty(files)
    error('No images found on: %s', archive_path);
else
    fprintf('Found %d images on: %s\n', num_files, archive_path);
end

%% OUTPUT FOLDER
scriptDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'results');

if exist(resultsDir, 'dir')
    fprintf('Output folder "%s" already exists and it will be emptied.\n', resultsDir);
    rmdir(resultsDir, 's');
end

mkdir(resultsDir);
fprintf('Created output folder: %s\n', resultsDir);

%% define camera parameters
camera = struct( ...
    'ImageSize',[1920 1080], ...
    'PrincipalPoint',[970 483], ... 
    'FocalLength',[1970 1970], ... 
    'Position',[1.8750 0 1.6600], ...
    'Rotation',[0 0 0] ...
);

focalLength    = camera.FocalLength; %[fx fy]
principalPoint = camera.PrincipalPoint; %[cx cy]
imageSize      = camera.ImageSize; %[width height]
height         = camera.Position(3);
pitch          = camera.Rotation(2);

%% BIRD'S EYE VIEW
%K=intrinsic (internal) camera calibration matrix 
%K=[fx 0 cx; 
%   0 fy cy;
%   0  0  1]
Intrinsics = cameraIntrinsics(focalLength, principalPoint, [imageSize(2) imageSize(1)]);

%object sensor rapresents the camera in the vehicle, with its intrinsic
%parmeters, height from the ground and pitch angle
sensor = monoCamera(Intrinsics, height, 'Pitch', pitch);

%define area to see with BEV [meters]
distAheadStart = 3;
distAheadEnd   = 30;
spaceToLeft    = 6;
spaceToRight   = 6;
area = [distAheadStart distAheadEnd -spaceToLeft spaceToRight];

%define BEV resolution
outImageSize = [NaN 800];

%object birds eye
birdsEye_object = birdsEyeView(sensor, area, outImageSize);

%% ROI - Region Of Interest
% ROI is the area of the image where it makes sense look for the lane. It is usefull to:
% - avoid sky, trees, buildings
% - reduce noise and false positives
% - make the papeline more stable and faster

% define a trapezio that has the longer base close to the camera and the shorter base far from the camera, to include the lane lines in the ROI
% the coordinates are in the image plane, so they are in pixel

%define the centre of the trapezio as the optical centre of the camera
cx = principalPoint(1);  %x coordinate of the optical centre 

top_width = 380;
bottom_width = 1250;
top_y = 600;
bottom_y = 1050;

roi_X = [cx - top_width/2, cx + top_width/2, cx + bottom_width/2, cx - bottom_width/2];
roi_Y = [top_y, top_y, bottom_y, bottom_y];
roi_poly = polyshape(roi_X, roi_Y);

% dopo aver costruito il trapezion nell'immagine originale, 
% lo porto nella BEV e costruisco una maschera binaria da usare nella pipeline
I0 = imread(fullfile(files(1).folder, files(1).name)); %read first image in the archive
BEV0 = transformImage(birdsEye_object, I0); %apply BEV to the first image
[bev_h, bev_w, ~] = size(BEV0); %get the size of the BEV image

roi_vehicle = imageToVehicle(sensor, [roi_X', roi_Y']); %convert the 4 vertices of the ROI trapezio (x,y pixels) in the vehicle coordinates
roi_bev = vehicleToImage(birdsEye_object, roi_vehicle); %convert the 4 vertices of the ROI trapezio (x,y in vehicle coordinates) in the BEV image coordinates (x,y pixels)
bev_mask = poly2mask(roi_bev(:,1), roi_bev(:,2), bev_h, bev_w); %create the binary mask, gives 1 if inside ROI, 0 if outside the trapezio

test_idx = [1 10 20 30 40];
for k = 1:length(test_idx)
    idx = test_idx(k);
    I = imread(fullfile(files(idx).folder, files(idx).name));
    %inverse perspective mapping
    BEV = transformImage(birdsEye_object, I);
    BEV_gray = rgb2gray(BEV); %convert to black and white
    % flag = 1 se pixel is in the ROI AND is NOT black (0), flag = 0 otherwise 
    flag = (BEV_gray > 0) & bev_mask; 

    figure;
    subplot(1,2,1);
    imshow(I);
    title(sprintf('Original image %d', idx));
    hold on;
    plot([roi_X roi_X(1)], [roi_Y roi_Y(1)], 'r-', 'LineWidth', 2);
    hold off;

    subplot(1,2,2);
    imshow(BEV_gray);
    title(sprintf('BEV image %d', idx));
    hold on;
    plot([roi_bev(:,1); roi_bev(1,1)], [roi_bev(:,2); roi_bev(1,2)], 'g-', 'LineWidth', 2);
    hold off;
end


%% YOLO DETECTOR
YOLODetector = yolov4ObjectDetector("tiny-yolov4-coco");
YOLOScoreThreshold = 0.35;


%% PIPELINE LANE DETECTION
for i=1:num_files
    I = imread(fullfile(files(i).folder, files(i).name));
    fprintf('Processing image %d/%d: %s\n', i, num_files, files(i).name);

    % 0. YOLO detection for cars and people on the original image
    [YOLO_bboxes, YOLO_scores, YOLO_labels] = detect(YOLODetector, I, ...
        'Threshold', YOLOScoreThreshold);

    % Keep only detections inside the road ROI
    [YOLO_bboxes, YOLO_scores, YOLO_labels] = ...
        filterBBoxesByROI(YOLO_bboxes, YOLO_scores, YOLO_labels, roi_poly);



    % 1. BEV on the current image
    BEV = transformImage(birdsEye_object, I);

    [bev_h, bev_w, ~] = size(BEV);

    obstacle_mask_bev = false(bev_h, bev_w);

    for d = 1:size(YOLO_bboxes, 1)
        box = YOLO_bboxes(d, :);

        x1 = box(1);
        y1 = box(2);
        x2 = box(1) + box(3);
        y2 = box(2) + box(4);

        bbox_pts_img = [
            x1 y1;
            x2 y1;
            x2 y2;
            x1 y2
        ];

        bbox_pts_vehicle = imageToVehicle(sensor, bbox_pts_img);
        valid = all(isfinite(bbox_pts_vehicle), 2);
        bbox_pts_vehicle = bbox_pts_vehicle(valid, :);

        if size(bbox_pts_vehicle,1) < 3
            continue;
        end

        bbox_pts_bev = vehicleToImage(birdsEye_object, bbox_pts_vehicle);
        valid = all(isfinite(bbox_pts_bev), 2);
        bbox_pts_bev = bbox_pts_bev(valid, :);

        if size(bbox_pts_bev,1) < 3
            continue;
        end

        obstacle_mask_bev = obstacle_mask_bev | ...
            poly2mask(bbox_pts_bev(:,1), bbox_pts_bev(:,2), bev_h, bev_w);
    end

    %2. grayscale 
    BEV_gray = rgb2gray(BEV);

    %3. road mask : flag = 1 if pixel in the ROI AND is NOT black (0), flag = 0 otherwise
    ROI_region_mask = (BEV_gray > 0) & bev_mask & ~obstacle_mask_bev;

    %4. noise reduction
    BEV_filtered = imgaussfilt(BEV_gray, 2);

    %5. lane enhancement : use a kernel to enhance the lane lines, which are vertical in the BEV
    BEV_enhanced = imtophat(BEV_filtered, strel('rectangle', [1 15]));

    % 6. Find the threshold 
    % choose a iterative thresholding method because 
    valid_pixels = double(BEV_enhanced(ROI_region_mask));

    if isempty(valid_pixels)
        Th = 50;
    else
        %set initial threshold
        Th = (max(valid_pixels) + min(valid_pixels)) / 2;
        Th_old = 0;

        %repeat util convergence, i.e. until the change in threshold is less than 0.5
        %my goal is to find a Th stabile and optimum to the image
        while abs(Th - Th_old) > 0.5
            Th_old = Th;

            %divide pixels in the vector in 2 groups:
        % - group A: pixels with intensity >= Th threshold  + brillanti
        % - group B: pixels with intensity < Th threshold   + scuri
            groupA = valid_pixels(valid_pixels >= Th);
            groupB = valid_pixels(valid_pixels < Th);

            %calculate mean for each group 
            if isempty(groupA)
                meanA = Th;
            else
                meanA = mean(groupA);
            end

            if isempty(groupB)
                meanB = Th;
            else
                meanB = mean(groupB);
            end

            % update threshold as the average od the 2 means
            % new Th is halfway between the mean of pixel + scuri and the mean of pixel + brillanti
            Th = (meanA + meanB) / 2;
        end
    end

    % 7. Binarization
    BEV_binary = (BEV_enhanced >= Th) & ROI_region_mask;
    BEV_binary = BEV_binary & ~obstacle_mask_bev;

    % 8. Remove small noisy components
    BEV_binary = bwareaopen(BEV_binary, 50);

    
    % 9. Column histogram for lane localization
    % BEV_binary is a binary image where 1=lane and 0=background
    % I sum the columns of the binary image and visualize that sum on a histogram that should show the lines
    h_bev = size(BEV_binary, 1);
    start_row = round(h_bev * 0.5);

    histogram_columns_sum = sum(BEV_binary(start_row:end, :), 1);

    midpoint = round(length(histogram_columns_sum)/2);
    margin = 50;

    % thresholds to decide whether a lane is present
    analyzed_height = h_bev - start_row + 1;
    dashedThreshold = max(20, round(0.07 * analyzed_height));
    solidThreshold  = max(40, round(0.20 * analyzed_height));

    % find locL and locR, the horizontal coordinates of left and right lanes
    leftHist  = histogram_columns_sum(margin:midpoint);
    rightHist = histogram_columns_sum(midpoint+1:end-margin);

    % find lane candidates as peaks in the histogram
    [leftPeaks, leftLocs] = findpeaks(leftHist, ...
        'MinPeakHeight', dashedThreshold, ...
        'MinPeakDistance', 60);

    [rightPeaks, rightLocs] = findpeaks(rightHist, ...
        'MinPeakHeight', dashedThreshold, ...
        'MinPeakDistance', 60);

    maxL = 0;
    maxR = 0;
    locL = NaN;
    locR = NaN;

    if ~isempty(leftPeaks)
        [maxL, idxL] = max(leftPeaks);
        locL = leftLocs(idxL) + margin - 1;
    end

    if ~isempty(rightPeaks)
        [maxR, idxR] = max(rightPeaks);
        locR = rightLocs(idxR) + midpoint;
    end

    y_bev = (1:size(BEV_binary,1))';

    % dashed lines _ _ _ _ _  or solid ______________ ??
    left_found = false;
    if maxL > dashedThreshold
        left_found = true;
        left_type = "dashed";
        color_left = 'g';
        if maxL > solidThreshold
            left_type = "solid";
            color_left = 'b';
        end
        bev_points_left = [repmat(locL, size(y_bev)), y_bev];
        vehicle_points_left = imageToVehicle(birdsEye_object, bev_points_left);
        image_points_left = vehicleToImage(sensor, vehicle_points_left);
    end

    right_found = false;
    if maxR > dashedThreshold
        right_found = true;
        right_type = "dashed";
        color_right = 'g';
        if maxR > solidThreshold
            right_type = "solid";
            color_right = 'b';
        end
        bev_points_right = [repmat(locR, size(y_bev)), y_bev];
        vehicle_points_right = imageToVehicle(birdsEye_object, bev_points_right);
        image_points_right = vehicleToImage(sensor, vehicle_points_right);
    end

    %% VISUALIZATION
    fig = figure(1); clf;
    set(fig, 'Position', [100 100 1400 700]);

    subplot(2,3,1);
    imshow(I);
    title('Original image');
    hold on;

    % Draw ROI on original image
    plot([roi_X roi_X(1)], [roi_Y roi_Y(1)], 'r-', 'LineWidth', 2);

    % Draw left and right lanes on original image
    if left_found
        plot(image_points_left(:,1), image_points_left(:,2), color_left, 'LineWidth', 2);
    end
    if right_found
        plot(image_points_right(:,1), image_points_right(:,2), color_right, 'LineWidth', 2);
    end

    % If no lanes found
    if ~left_found && ~right_found
        text(size(I,2)/2, size(I,1)-40, 'No lanes found', ...
            'Color', 'red', ...
            'FontSize', 11, ...
            'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', ...
            'BackgroundColor', 'black');
    end

        % Draw YOLO detections on original image
    for d = 1:size(YOLO_bboxes, 1)
        thisBox = YOLO_bboxes(d, :);

        % Approximate distance from bounding-box height
        real_height_m = 1.5;   % rough average obstacle height
        bbox_height_px = thisBox(4);

        if bbox_height_px > 0
            distance_m = (real_height_m * focalLength(2)) / bbox_height_px;
        else
            distance_m = NaN;
        end

        rectangle('Position', thisBox, 'EdgeColor', [1 1 0], 'LineWidth', 2);

        labelText = formatDetectionLabel(YOLO_labels(d), YOLO_scores(d), distance_m);
        text(thisBox(1), max(15, thisBox(2)-10), labelText, ...
            'Color', 'yellow', 'FontSize', 9, 'FontWeight', 'bold', ...
            'BackgroundColor', 'black', 'Margin', 1);
    end


    hold off;

    subplot(2,3,2);
    imshow(BEV_gray);
    title('BEV grayscale');
    hold on;

    if left_found
        xline(locL, color_left, 'LineWidth', 2);
        text(locL + 8, 35, char(left_type), ...
            'Color', color_left, 'FontSize', 10, 'FontWeight', 'bold');
    end

    if right_found
        xline(locR, color_right, 'LineWidth', 2);
        text(locR + 8, 70, char(right_type), ...
            'Color', color_right, 'FontSize', 10, 'FontWeight', 'bold');
    end

    if ~left_found && ~right_found
        text(30, 40, 'No lanes found', ...
            'Color', 'g', ...
            'FontSize', 12, ...
            'FontWeight', 'bold', ...
            'BackgroundColor', 'k');
    end
    hold off;

    subplot(2,3,3);
    imshow(ROI_region_mask);
    title('ROI region mask');

    subplot(2,3,4);
    imshow(BEV_enhanced, []);
    title('Enhanced BEV');

    subplot(2,3,5);
    imshow(BEV_binary);
    title('Binary BEV');

    subplot(2,3,6);
    plot(histogram_columns_sum, 'LineWidth', 1.5);
    title('Column histogram');
    grid on;
    hold on;
    if left_found,  xline(locL, 'g', 'LineWidth', 2); end
    if right_found, xline(locR, 'g', 'LineWidth', 2); end
    hold off;

    sgtitle(sprintf('Lane detection pipeline - %s', files(i).name), 'Interpreter', 'none');

    saveas(fig, fullfile(resultsDir, sprintf('result_%02d.png', i)));
    drawnow;
    
    
end

function [filtered_bboxes, filtered_scores, filtered_labels] = ...
    filterBBoxesByROI(bboxes, scores, labels, roi_poly)

    if isempty(bboxes)
        filtered_bboxes = bboxes;
        filtered_scores = scores;
        filtered_labels = labels;
        return;
    end

    keep = false(size(bboxes,1),1);

    for k = 1:size(bboxes,1)
        x_center = bboxes(k,1) + bboxes(k,3)/2;
        y_center = bboxes(k,2) + bboxes(k,4)/2;

        keep(k) = isinterior(roi_poly, x_center, y_center);
    end

    filtered_bboxes = bboxes(keep, :);
    filtered_scores = scores(keep, :);
    filtered_labels = labels(keep, :);
end

function labelText = formatDetectionLabel(label, score, distance_m)
    if isnan(distance_m)
        labelText = sprintf('%s | %.2f', string(label), score);
    else
        labelText = sprintf('%s | %.2f | %.1f m', string(label), score, distance_m);
    end
end