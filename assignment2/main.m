clear; clc; close all

% Anna Roma
% Assignment 2 - Line Detection using GOLD Lane Detection Algorithm

% Visualization tuning for narrow BEV subplots
roiLineWidth = 0.75;
laneLineWidth = 1.0;
histLineWidth = 0.75;
peakLineWidth = 0.75;

search_path = 'archive/front_camera_46/*.jpg';

scriptDir = fileparts(mfilename('fullpath'));
resolved_search_path = resolveSearchPath(search_path, scriptDir);
files = dir(resolved_search_path);
num_files = length(files);
if isempty(files)
    error('No images found on: %s', resolved_search_path);
else
    fprintf('Found %d images on: %s\n', num_files, resolved_search_path);
end

%% OUTPUT FOLDER
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
% Keep a slightly wider longitudinal span and a more compact output width:
% these settings produce a less distorted BEV on the provided test frames.
distAheadStart = 5;
distAheadEnd   = 40;
spaceToLeft    = 5;
spaceToRight   = 5;
area = [distAheadStart distAheadEnd -spaceToLeft spaceToRight];

%define BEV resolution
outImageSize = [NaN 400];

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
roi_bev_boundary = maskToBoundary(bev_mask);

test_idx = [1 10 20 30 40];
test_idx = test_idx(test_idx <= num_files);
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
    plot([roi_X roi_X(1)], [roi_Y roi_Y(1)], 'r-', 'LineWidth', roiLineWidth);
    hold off;

    subplot(1,2,2);
    imshow(BEV_gray);
    title(sprintf('BEV image %d', idx));
    hold on;
    plotLane(roi_bev_boundary, 'g', roiLineWidth);
    hold off;
end


%% OBSTACLE DETECTOR
YOLOScoreThreshold = 0.35;
[obstacleDetector, detectorMode] = initializeObstacleDetector();
allowedObstacleLabels = ["car","truck","bus","motorcycle","bicycle","person","vehicle"];


%% TEMPORARY MEMORY FOR LAST VALID LANES
maxMemoryFrames = 5; missedFrames = 0;
last_left_found = false; last_right_found = false;
last_locL = NaN; last_locR = NaN;
last_left_type = ""; last_right_type = "";
last_color_left = 'g'; last_color_right = 'g';
last_left_lane_bev = nan(0, 2);
last_right_lane_bev = nan(0, 2);

%% PIPELINE LANE DETECTION
for i=1:num_files
    I = imread(fullfile(files(i).folder, files(i).name));
    fprintf('Processing image %d/%d: %s\n', i, num_files, files(i).name);

    % 0. Obstacle detection on the original image
    [YOLO_bboxes, YOLO_scores, YOLO_labels] = detectObstacles( ...
        obstacleDetector, detectorMode, I, YOLOScoreThreshold);

    % Keep only obstacle classes inside the road ROI
    [YOLO_bboxes, YOLO_scores, YOLO_labels] = ...
        filterObstacleDetections(YOLO_bboxes, YOLO_scores, YOLO_labels, ...
        roi_poly, allowedObstacleLabels);



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

        % Some detector boxes can extend slightly outside the camera image.
        % Clamp them before projecting to vehicle coordinates.
        bbox_pts_img(:,1) = min(max(bbox_pts_img(:,1), 1), size(I,2));
        bbox_pts_img(:,2) = min(max(bbox_pts_img(:,2), 1), size(I,1));

        if size(unique(round(bbox_pts_img), 'rows'), 1) < 3
            continue;
        end

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

        % Projected obstacle polygons can land far outside the BEV image on
        % some frames. Clip them before rasterization so poly2mask does not
        % fail on extreme vertices.
        bbox_pts_bev(:,1) = min(max(bbox_pts_bev(:,1), 1), bev_w);
        bbox_pts_bev(:,2) = min(max(bbox_pts_bev(:,2), 1), bev_h);

        if size(unique(round(bbox_pts_bev), 'rows'), 1) < 3
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

    % 6. Find two separate thresholds, one for each half of the image
    split_col = round(bev_w / 2);
    left_half_mask = ROI_region_mask;
    left_half_mask(:, split_col+1:end) = false;
    right_half_mask = ROI_region_mask;
    right_half_mask(:, 1:split_col) = false;

    leftThreshold = computeIterativeThreshold(BEV_enhanced, left_half_mask, 50);
    rightThreshold = computeIterativeThreshold(BEV_enhanced, right_half_mask, 50);

    % 7. Binarization with half-specific thresholds
    BEV_binary_left = (BEV_enhanced >= leftThreshold) & left_half_mask;
    BEV_binary_right = (BEV_enhanced >= rightThreshold) & right_half_mask;
    BEV_binary = BEV_binary_left | BEV_binary_right;

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

    laneSearchHalfWidth = 35;

    % dashed lines _ _ _ _ _  or solid ______________ ??
    left_found = false;
    if maxL > dashedThreshold
        candidate_left_lane_bev = traceLaneFromBinary(BEV_binary, locL, laneSearchHalfWidth, [1 midpoint]);
        left_support_mask = isfinite(candidate_left_lane_bev(:,1));
        if nnz(left_support_mask(start_row:end)) >= dashedThreshold
            left_found = true;
            left_lane_bev = candidate_left_lane_bev;
            left_type = classifyLaneType(left_support_mask, start_row, solidThreshold);
            color_left = 'g';
            valid_left_bottom = left_lane_bev(start_row:end,1);
            locL = round(median(valid_left_bottom(isfinite(valid_left_bottom))));
        end
    end

    right_found = false;
    if maxR > dashedThreshold
        candidate_right_lane_bev = traceLaneFromBinary(BEV_binary, locR, laneSearchHalfWidth, [midpoint+1 bev_w]);
        right_support_mask = isfinite(candidate_right_lane_bev(:,1));
        if nnz(right_support_mask(start_row:end)) >= dashedThreshold
            right_found = true;
            right_lane_bev = candidate_right_lane_bev;
            right_type = classifyLaneType(right_support_mask, start_row, solidThreshold);
            color_right = 'g';
            valid_right_bottom = right_lane_bev(start_row:end,1);
            locR = round(median(valid_right_bottom(isfinite(valid_right_bottom))));
        end
    end

    %% MEMORY MANAGEMENT
    detected_now = left_found || right_found;
    if detected_now % I have detected lines in the current frame
        missedFrames = 0; % update memory with current valid detections
        last_left_found = left_found;
        last_right_found = right_found;
        if left_found
            last_locL = locL;
            last_left_type = left_type;
            last_color_left = color_left;
            last_left_lane_bev = left_lane_bev;
        end
        if right_found
            last_locR = locR;
            last_right_type = right_type;
            last_color_right = color_right;
            last_right_lane_bev = right_lane_bev;
        end
    else % I have not detected any line in the current frame
        missedFrames = missedFrames + 1;
    end

    % decide what to display:
    % use memory if I have not detected any lines AND  I am still in the memory window AND really exists a valid old line saved
    use_memory = ~detected_now && (missedFrames <= maxMemoryFrames) && ...
                 (last_left_found || last_right_found);

    if use_memory %no lanes fouund, visualize the last valid lanes found in the memory
        display_left_found = last_left_found;
        display_right_found = last_right_found;

        display_locL = last_locL;
        display_locR = last_locR;

        display_left_type = last_left_type;
        display_right_type = last_right_type;

        display_color_left = last_color_left;
        display_color_right = last_color_right;
        display_left_lane_bev = last_left_lane_bev;
        display_right_lane_bev = last_right_lane_bev;
    else % not using memory, visualize the current detections
        display_left_found = left_found;
        display_right_found = right_found;

        if left_found % if i have found a left line in current frame, I visualize it
            display_locL = locL;
            display_left_type = left_type;
            display_color_left = color_left;
            display_left_lane_bev = left_lane_bev;
        else %if not, I visualize nothing (NAN) and I will write "no lane found" on the image
            display_locL = NaN;
            display_left_type = "";
            display_color_left = 'g';
            display_left_lane_bev = nan(0, 2);
        end

        if right_found
            display_locR = locR;
            display_right_type = right_type;
            display_color_right = color_right;
            display_right_lane_bev = right_lane_bev;
        else
            display_locR = NaN;
            display_right_type = "";
            display_color_right = 'g';
            display_right_lane_bev = nan(0, 2);
        end
    end



    %% VISUALIZATION
    % Build lane points for visualization using display variables
    if display_left_found
        bev_points_left_display = display_left_lane_bev;
        image_points_left_display = bevToImagePoints(bev_points_left_display, birdsEye_object, sensor);
    end

    if display_right_found
        bev_points_right_display = display_right_lane_bev;
        image_points_right_display = bevToImagePoints(bev_points_right_display, birdsEye_object, sensor);
    end

    fig = figure(1); clf;
    set(fig, 'Position', [100 100 1400 700]);

    subplot(2,3,1);
    imshow(I);
    title('Original image');
    hold on;

    % Draw ROI on original image
    plot([roi_X roi_X(1)], [roi_Y roi_Y(1)], 'r-', 'LineWidth', roiLineWidth);

    % Draw left and right lanes on original image
    if display_left_found
        plotLane(image_points_left_display, display_color_left, laneLineWidth);
    end
    if display_right_found
        plotLane(image_points_right_display, display_color_right, laneLineWidth);
    end

    if ~display_left_found && ~display_right_found
        text(size(I,2)/2, size(I,1)-40, 'No lanes found', ...
            'Color', 'red', 'FontSize', 11, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'BackgroundColor', 'black');
    elseif use_memory
        text(size(I,2)/2, size(I,1)-40, ...
            sprintf('Using previous lanes (%d/%d)', missedFrames, maxMemoryFrames), ...
            'Color', 'yellow', 'FontSize', 11, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'BackgroundColor', 'black');
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

        obstacle_in_lane = obstacleInLane(thisBox, sensor, birdsEye_object, ...
            display_left_lane_bev, display_right_lane_bev);

        if obstacle_in_lane
            edgeColor = [1 0 0];
        else
            edgeColor = [1 1 0];
        end

        rectangle('Position', thisBox, 'EdgeColor', edgeColor, 'LineWidth', 2);

        labelText = formatDetectionLabel(YOLO_labels(d), YOLO_scores(d), ...
            distance_m, obstacle_in_lane);
        text(thisBox(1), max(15, thisBox(2)-10), labelText, ...
            'Color', 'yellow', 'FontSize', 9, 'FontWeight', 'bold', ...
            'BackgroundColor', 'black', 'Margin', 1);
    end


    hold off;

    subplot(2,3,2);
    imshow(BEV_gray);
    title('BEV grayscale');
    hold on;
    plotLane(roi_bev_boundary, 'c', roiLineWidth);

    if display_left_found
        plotLane(bev_points_left_display, display_color_left, laneLineWidth);
        text(display_locL + 8, 35, char(display_left_type), ...
            'Color', display_color_left, 'FontSize', 10, 'FontWeight', 'bold');
    end

    if display_right_found
        plotLane(bev_points_right_display, display_color_right, laneLineWidth);
        text(display_locR + 8, 70, char(display_right_type), ...
            'Color', display_color_right, 'FontSize', 10, 'FontWeight', 'bold');
    end

    if ~display_left_found && ~display_right_found
        text(30, 40, 'No lanes found', ...
            'Color', 'g', ...
            'FontSize', 12, ...
            'FontWeight', 'bold', ...
            'BackgroundColor', 'k');
    elseif use_memory
        text(30, 40, sprintf('Using previous lanes (%d/%d)', missedFrames, maxMemoryFrames), ...
            'Color', 'y', ...
            'FontSize', 11, ...
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
    plot(histogram_columns_sum, 'LineWidth', histLineWidth);
    title('Column histogram');
    grid on;
    hold on;
    if display_left_found,  xline(display_locL, 'g', 'LineWidth', peakLineWidth); end
    if display_right_found, xline(display_locR, 'g', 'LineWidth', peakLineWidth); end
    hold off;

    sgtitle(sprintf('Lane detection pipeline - %s', files(i).name), 'Interpreter', 'none');

    saveas(fig, fullfile(resultsDir, sprintf('result_%02d.png', i)));
    drawnow;
    
    
end

function resolved_path = resolveSearchPath(search_path, scriptDir)
    if isfolder(search_path) || startsWith(search_path, filesep) || ~isempty(regexp(search_path, '^[A-Za-z]:', 'once'))
        resolved_path = search_path;
        return;
    end

    candidate = fullfile(scriptDir, search_path);
    if ~isempty(dir(candidate))
        resolved_path = candidate;
    else
        resolved_path = search_path;
    end
end

function Th = computeIterativeThreshold(imageData, validMask, fallbackValue)
    valid_pixels = double(imageData(validMask));

    if isempty(valid_pixels)
        Th = fallbackValue;
        return;
    end

    Th = (max(valid_pixels) + min(valid_pixels)) / 2;
    Th_old = 0;

    while abs(Th - Th_old) > 0.5
        Th_old = Th;
        groupA = valid_pixels(valid_pixels >= Th);
        groupB = valid_pixels(valid_pixels < Th);

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

        Th = (meanA + meanB) / 2;
    end
end

function lane_points = traceLaneFromBinary(binaryImage, seedCol, halfWindow, colRange)
    imageHeight = size(binaryImage, 1);
    tracedCols = nan(imageHeight, 1);
    currentCol = seedCol;

    for row = imageHeight:-1:1
        col_min = max(colRange(1), round(currentCol - halfWindow));
        col_max = min(colRange(2), round(currentCol + halfWindow));

        if col_min > col_max
            continue;
        end

        activeCols = find(binaryImage(row, col_min:col_max)) + col_min - 1;
        if isempty(activeCols)
            continue;
        end

        splitIdx = [1, find(diff(activeCols) > 1) + 1, numel(activeCols) + 1];
        segmentCenters = zeros(numel(splitIdx) - 1, 1);
        for s = 1:numel(splitIdx) - 1
            segment = activeCols(splitIdx(s):splitIdx(s+1)-1);
            segmentCenters(s) = mean(segment);
        end

        [~, bestIdx] = min(abs(segmentCenters - currentCol));
        currentCol = segmentCenters(bestIdx);
        tracedCols(row) = currentCol;
    end

    supportMask = isfinite(tracedCols);
    supportMask = bwareaopen(supportMask, 8);
    tracedCols(~supportMask) = NaN;
    lane_points = [tracedCols, (1:imageHeight)'];
end

function lane_type = classifyLaneType(supportMask, startRow, solidThreshold)
    supportSlice = supportMask(startRow:end);

    if ~any(supportSlice)
        lane_type = "dashed";
        return;
    end

    supportLength = numel(supportSlice);
    coverageRatio = nnz(supportSlice) / supportLength;

    % Measure continuity instead of only counting total supporting rows:
    % dashed lanes can still have many active rows, but broken into short runs.
    paddedSlice = [false; supportSlice(:); false];
    runStarts = find(diff(paddedSlice) == 1);
    runEnds = find(diff(paddedSlice) == -1) - 1;
    runLengths = runEnds - runStarts + 1;
    longestRun = max(runLengths);

    minCoverageForSolid = 0.55;
    minLongestRunForSolid = max(solidThreshold, round(0.35 * supportLength));

    if coverageRatio >= minCoverageForSolid && longestRun >= minLongestRunForSolid
        lane_type = "solid";
    else
        lane_type = "dashed";
    end
end

function image_points = bevToImagePoints(bev_points, birdsEye_object, sensor)
    valid = all(isfinite(bev_points), 2);
    image_points = nan(size(bev_points));

    if ~any(valid)
        return;
    end

    vehicle_points = imageToVehicle(birdsEye_object, bev_points(valid, :));
    projected_points = vehicleToImage(sensor, vehicle_points);
    image_points(valid, :) = projected_points;
end

function plotLane(points, laneColor, lineWidth)
    if isempty(points) || size(points, 1) == 0
        return;
    end

    if ~any(all(isfinite(points), 2))
        return;
    end

    % Keep NaNs so MATLAB breaks the polyline into visible dashed segments
    % instead of connecting separate detections into one solid line.
    plot(points(:,1), points(:,2), 'Color', laneColor, 'LineWidth', lineWidth);
end

function boundary_points = maskToBoundary(binaryMask)
    boundaries = bwboundaries(binaryMask, 'noholes');

    if isempty(boundaries)
        boundary_points = nan(0, 2);
        return;
    end

    lengths = cellfun(@(b) size(b, 1), boundaries);
    [~, idx] = max(lengths);
    boundary_rc = boundaries{idx};
    boundary_points = [boundary_rc(:,2), boundary_rc(:,1)];
end

function [detector, mode] = initializeObstacleDetector()
    detector = struct();
    mode = "none";

    try
        detector.model = yolov4ObjectDetector("tiny-yolov4-coco");
        mode = "yolo";
        fprintf('Obstacle detector: YOLO v4 tiny COCO\n');
        return;
    catch ME
        warning(['YOLO disabled: %s\n', ...
            'Falling back to ACF vehicle/person detectors.\n', ...
            'Install the Computer Vision Toolbox Model for YOLO v4 Object Detection ', ...
            'support package from MATLAB Add-On Explorer to re-enable YOLO.'], ...
            ME.message);
    end

    try
        detector.vehicle = vehicleDetectorACF('front-rear-view');
        detector.person = peopleDetectorACF('caltech-50x21');
        mode = "acf";
        fprintf('Obstacle detector: ACF fallback (vehicles + people)\n');
    catch ME
        warning(['ACF fallback disabled: %s\n', ...
            'Obstacle detection will remain unavailable on this MATLAB setup.'], ...
            ME.message);
    end
end

function [bboxes, scores, labels] = detectObstacles(detector, mode, I, yoloThreshold)
    bboxes = zeros(0, 4);
    scores = zeros(0, 1);
    labels = strings(0, 1);

    switch string(mode)
        case "yolo"
            [bboxes, scores, labels] = detect(detector.model, I, ...
                'Threshold', yoloThreshold);

        case "acf"
            [vehicleBboxes, vehicleScores] = detect(detector.vehicle, I);
            if ~isempty(vehicleBboxes)
                vehicleBboxes = selectStrongestBbox(vehicleBboxes, vehicleScores, ...
                    'OverlapThreshold', 0.5);
                numVehicles = size(vehicleBboxes, 1);
                bboxes = [bboxes; vehicleBboxes];
                scores = [scores; ones(numVehicles, 1)];
                labels = [labels; repmat("vehicle", numVehicles, 1)];
            end

            [peopleBboxes, peopleScores] = detect(detector.person, I);
            if ~isempty(peopleBboxes)
                peopleBboxes = selectStrongestBbox(peopleBboxes, peopleScores, ...
                    'OverlapThreshold', 0.5);
                numPeople = size(peopleBboxes, 1);
                bboxes = [bboxes; peopleBboxes];
                scores = [scores; ones(numPeople, 1)];
                labels = [labels; repmat("person", numPeople, 1)];
            end

        otherwise
            % Leave outputs empty when no detector is available.
    end
end

function [filtered_bboxes, filtered_scores, filtered_labels] = ...
    filterObstacleDetections(bboxes, scores, labels, roi_poly, allowedLabels)

    if isempty(bboxes)
        filtered_bboxes = bboxes;
        filtered_scores = scores;
        filtered_labels = labels;
        return;
    end

    keep = false(size(bboxes,1),1);

    normalizedAllowedLabels = lower(string(allowedLabels));

    for k = 1:size(bboxes,1)
        box = bboxes(k, :);
        label = lower(strtrim(string(labels(k))));

        % YOLO boxes for distant obstacles often have a center above the ROI
        % even when the object is clearly on the road. Use bottom support
        % points instead of the box center to keep road users touching the road.
        probePoints = [
            box(1) + box(3) / 2, box(2) + box(4);   % bottom center
            box(1),              box(2) + box(4);   % bottom left
            box(1) + box(3),     box(2) + box(4)    % bottom right
        ];

        label_ok = any(strcmpi(label, normalizedAllowedLabels));
        in_roi = any(isinterior(roi_poly, probePoints(:,1), probePoints(:,2)));
        keep(k) = label_ok && in_roi;
    end

    filtered_bboxes = bboxes(keep, :);
    filtered_scores = scores(keep, :);
    filtered_labels = labels(keep, :);
end

function is_in_lane = obstacleInLane(bbox, sensor, birdsEye_object, ...
    left_lane_bev, right_lane_bev)

    is_in_lane = false;

    if isempty(left_lane_bev) || isempty(right_lane_bev)
        return;
    end

    if size(left_lane_bev, 1) == 0 || size(right_lane_bev, 1) == 0
        return;
    end

    bottom_center_img = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)];
    bottom_center_vehicle = imageToVehicle(sensor, bottom_center_img);
    if ~all(isfinite(bottom_center_vehicle))
        return;
    end

    bev_point = vehicleToImage(birdsEye_object, bottom_center_vehicle);
    if ~all(isfinite(bev_point))
        return;
    end

    lane_y = bev_point(2);
    left_x = interpolateLaneX(left_lane_bev, lane_y);
    right_x = interpolateLaneX(right_lane_bev, lane_y);

    if ~isfinite(left_x) || ~isfinite(right_x)
        return;
    end

    lane_margin = 10;
    is_in_lane = (bev_point(1) >= left_x - lane_margin) && ...
                 (bev_point(1) <= right_x + lane_margin);
end

function x_interp = interpolateLaneX(lane_points, targetY)
    x_interp = NaN;

    if isempty(lane_points) || size(lane_points,1) == 0
        return;
    end

    valid = all(isfinite(lane_points), 2);
    lane_points = lane_points(valid, :);

    if size(lane_points,1) < 2
        return;
    end

    uniqueY = lane_points(:,2);
    uniqueX = lane_points(:,1);
    [uniqueY, uniqueIdx] = unique(uniqueY);
    uniqueX = uniqueX(uniqueIdx);

    if targetY < min(uniqueY) || targetY > max(uniqueY)
        return;
    end

    x_interp = interp1(uniqueY, uniqueX, targetY, 'linear');
end

function labelText = formatDetectionLabel(label, score, distance_m, isInLane)
    if isnan(distance_m)
        baseText = sprintf('%s | %.2f', string(label), score);
    else
        baseText = sprintf('%s | %.2f | %.1f m', string(label), score, distance_m);
    end

    if isInLane
        labelText = sprintf('%s | IN LANE', baseText);
    else
        labelText = baseText;
    end
end
