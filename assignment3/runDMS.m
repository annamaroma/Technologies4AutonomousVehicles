clear; clc; close all;

% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System
%
% This script implements a webcam-based Driver Monitoring System (DMS)
% using MATLAB built-in detectors plus the provided rPPG material.
% Mandatory features implemented:
% - Webcam acquisition
% - Face / nose / eyes monitoring
% - Owl long and short distraction detection
% - Microsleep and sleep detection based on both-eye closure
% - Heart-rate estimation in BPM using live rPPG buffering and ICA
% - Video visualization with driver state and BPM overlays
%
% Optional lizard distraction is intentionally not implemented to preserve
% robustness of the mandatory features without external landmark models.

%% Paths
scriptDir = fileparts(mfilename("fullpath"));
addpath(scriptDir);
addpath(fullfile(scriptDir, "materiale utile", "remote_PPG"));
addpath(fullfile(scriptDir, "materiale utile", "ICA"));

%% Configurable parameters
config = struct();

% State timing thresholds [s]
config.longDistractionSec = 5;
config.shortAccumSec = 10;
config.shortWindowSec = 30;
config.returnToRoadSec = 2;
config.microsleepSec = 4;
config.sleepSec = 7;
config.eyesOpenResetSec = 2;

% Calibration and detection robustness
config.calibrationDurationSec = 3;
config.maxDetectionHoldSec = 0.75;
config.maxForwardBaselineStd = 0.10;
config.facePaddingFrac = 0.10;
config.eyeSearchTopFrac = 0.12;
config.eyeSearchHeightFrac = 0.42;
config.noseSearchTopFrac = 0.28;
config.noseSearchHeightFrac = 0.38;

% Gaze thresholds in normalized face coordinates
config.gazeThresholdX = 0.13;
config.gazeThresholdY = 0.18;
config.recenterAlpha = 0.02;

% Eye closure thresholds
config.eyeClosedRatio = 0.55;
config.eyeDarkPixelThreshold = 0.42;
config.eyeMinConfidence = 0.25;
config.eyeAdaptiveBlend = 0.01;

% rPPG configuration
config.hrWindowSec = 20;
config.hrMinWindowSec = 10;
config.hrUpdateIntervalSec = 1.0;
config.hrBandBpm = [45 165];
config.hrMinSamples = 180;
config.hrRoiTopFrac = 0.18;
config.hrRoiBottomFrac = 0.72;
config.hrRoiSideFrac = 0.20;
config.hrMinQuality = 0.05;

% Visualization
config.figureName = "Assignment 3 - Driver Monitoring System";
config.annotationFontSize = 20;
config.warningFontSize = 16;

%% Check webcam support
if exist("webcamlist", "file") ~= 2
    error(["MATLAB Support Package for USB Webcams is not installed. " ...
        "Install it from Add-On Explorer before running runDMS.m."]);
end

try
    availableCameras = webcamlist;
catch webcamListError
    error(["Unable to query USB webcams. Check that the MATLAB webcam " ...
        "support package is installed and the camera is not busy.\n%s"], ...
        webcamListError.message);
end

if isempty(availableCameras)
    error("No USB webcam detected. Connect a webcam or enable the laptop camera.");
end

%% Initialize webcam and cleanup
cam = webcam(1);
cleanupObj = onCleanup(@() localCleanup(cam, config.figureName));

%% Initialize detectors
faceDetector = vision.CascadeObjectDetector("FrontalFaceLBP");
faceDetector.MergeThreshold = 10;
faceDetector.ScaleFactor = 1.1;

leftEyeDetector = vision.CascadeObjectDetector("LeftEye");
leftEyeDetector.MergeThreshold = 8;

rightEyeDetector = vision.CascadeObjectDetector("RightEye");
rightEyeDetector.MergeThreshold = 8;

noseDetector = vision.CascadeObjectDetector("Nose");
noseDetector.MergeThreshold = 15;

%% Runtime state initialization
detectionState = struct( ...
    "lastFaceBox", [], ...
    "lastNoseBox", [], ...
    "lastLeftEyeBox", [], ...
    "lastRightEyeBox", [], ...
    "lastDetectionTime", -inf, ...
    "faceDetected", false, ...
    "message", "Calibrating... look forward with eyes open", ...
    "calibrated", false);

gazeBaselineSamples = [];
eyeBaselineSamples = [];

distractionState = struct( ...
    "lastTimestamp", NaN, ...
    "awayDuration", 0, ...
    "returnDuration", 0, ...
    "awayIntervals", zeros(0, 2), ...
    "longActive", false, ...
    "shortActive", false, ...
    "currentAwayStart", NaN);

drowsinessState = struct( ...
    "lastTimestamp", NaN, ...
    "closedDuration", 0, ...
    "openDuration", 0, ...
    "microsleepActive", false, ...
    "sleepActive", false);

rppgState = struct( ...
    "timestamps", [], ...
    "rgbMeans", zeros(0, 3), ...
    "lastBpm", NaN, ...
    "lastUpdateTime", -inf, ...
    "quality", 0);

currentStateLabel = "Focused on the road";
lastValidForward = true;

%% Create visualization window
viewer = figure("Name", config.figureName, ...
    "NumberTitle", "off", ...
    "MenuBar", "none", ...
    "ToolBar", "none", ...
    "Color", "k");

frame = snapshot(cam);
imageHandle = imshow(frame, "Border", "tight");
drawnow;

%% Initial calibration
calibrationStart = tic;
calibrationClockStart = tic;

while ishandle(viewer) && toc(calibrationClockStart) < config.calibrationDurationSec
    frame = snapshot(cam);
    nowSec = toc(calibrationStart);

    [faceBox, noseBox, leftEyeBox, rightEyeBox, faceDetected] = ...
        localDetectFaceFeatures(frame, faceDetector, noseDetector, leftEyeDetector, rightEyeDetector, config);

    if faceDetected
        detectionState.faceDetected = true;
        detectionState.lastDetectionTime = nowSec;
        detectionState.lastFaceBox = faceBox;
        detectionState.lastNoseBox = noseBox;
        detectionState.lastLeftEyeBox = leftEyeBox;
        detectionState.lastRightEyeBox = rightEyeBox;

        gazeMetrics = estimateGazeState(faceBox, noseBox, [], config);
        eyeMetrics = estimateEyeClosure(frame, leftEyeBox, rightEyeBox, [], config);

        gazeBaselineSamples(end+1, :) = gazeMetrics.noseCenterNorm; %#ok<SAGROW>
        if eyeMetrics.confidence > config.eyeMinConfidence
            eyeBaselineSamples(end+1, :) = [eyeMetrics.leftOpenness, eyeMetrics.rightOpenness]; %#ok<SAGROW>
        end

        detectionState.message = "Calibrating... keep looking forward";
    else
        detectionState.faceDetected = false;
        detectionState.message = "Face not detected during calibration";
    end

    overlayFrame = drawDMSOverlay(frame, "Focused on the road", NaN, detectionState, config);
    set(imageHandle, "CData", overlayFrame);
    drawnow limitrate;
end

if size(gazeBaselineSamples, 1) < 5
    error(["Calibration failed because the face, nose and eyes were not " ...
        "detected reliably. Keep the whole face visible and rerun the script."]);
end

gazeBaseline = median(gazeBaselineSamples, 1, "omitnan");
eyeBaseline = median(eyeBaselineSamples, 1, "omitnan");

if any(isnan(eyeBaseline))
    error(["Calibration failed because open-eye samples were insufficient. " ...
        "Keep both eyes visible and open during calibration."]);
end

detectionState.calibrated = true;
detectionState.message = "";

%% Main frame loop
mainClock = tic;
while ishandle(viewer)
    frame = snapshot(cam);
    nowSec = toc(mainClock);

    [faceBox, noseBox, leftEyeBox, rightEyeBox, faceDetected] = ...
        localDetectFaceFeatures(frame, faceDetector, noseDetector, leftEyeDetector, rightEyeDetector, config);

    if faceDetected
        detectionState.faceDetected = true;
        detectionState.lastDetectionTime = nowSec;
        detectionState.lastFaceBox = faceBox;
        detectionState.lastNoseBox = noseBox;
        detectionState.lastLeftEyeBox = leftEyeBox;
        detectionState.lastRightEyeBox = rightEyeBox;
        detectionState.message = "";
    else
        detectionState.faceDetected = false;
        if nowSec - detectionState.lastDetectionTime > config.maxDetectionHoldSec
            detectionState.message = "Face not detected";
        else
            faceBox = detectionState.lastFaceBox;
            noseBox = detectionState.lastNoseBox;
            leftEyeBox = detectionState.lastLeftEyeBox;
            rightEyeBox = detectionState.lastRightEyeBox;
        end
    end

    canUpdateLogic = ~isempty(faceBox) && ~isempty(noseBox) && ~isempty(leftEyeBox) && ...
        ~isempty(rightEyeBox) && (nowSec - detectionState.lastDetectionTime <= config.maxDetectionHoldSec);

    if canUpdateLogic
        gazeMetrics = estimateGazeState(faceBox, noseBox, gazeBaseline, config);
        eyeMetrics = estimateEyeClosure(frame, leftEyeBox, rightEyeBox, eyeBaseline, config);

        % Slowly adapt baseline only when the driver is likely looking forward.
        if gazeMetrics.isForward && eyeMetrics.confidence > config.eyeMinConfidence
            gazeBaseline = (1 - config.recenterAlpha) .* gazeBaseline + ...
                config.recenterAlpha .* gazeMetrics.noseCenterNorm;
            eyeBaseline = (1 - config.eyeAdaptiveBlend) .* eyeBaseline + ...
                config.eyeAdaptiveBlend .* [eyeMetrics.leftOpenness, eyeMetrics.rightOpenness];
        end

        lastValidForward = gazeMetrics.isForward;
        distractionState = updateDistractionTimers(nowSec, gazeMetrics.isAway, distractionState, config);
        drowsinessState = updateDrowsinessTimers(nowSec, eyeMetrics.bothEyesClosed, drowsinessState, config);

        if ~isempty(faceBox)
            [rgbMean, rppgState] = localUpdateRppgBuffer(frame, faceBox, leftEyeBox, rightEyeBox, nowSec, rppgState, config);
            if ~isempty(rgbMean)
                [bpm, quality, debugInfo] = estimateHeartRateRPPG(rppgState.timestamps, rppgState.rgbMeans, config, rppgState.lastBpm);
                rppgState.quality = quality;
                if debugInfo.updated
                    rppgState.lastBpm = bpm;
                    rppgState.lastUpdateTime = nowSec;
                end
            end
        end
    end

    if drowsinessState.sleepActive
        currentStateLabel = "Sleep";
    elseif drowsinessState.microsleepActive
        currentStateLabel = "Microsleep";
    elseif distractionState.longActive
        currentStateLabel = "Distracted (long)";
    elseif distractionState.shortActive
        currentStateLabel = "Distracted (short)";
    else
        currentStateLabel = "Focused on the road";
    end

    diagnostics = detectionState;
    diagnostics.faceBox = faceBox;
    diagnostics.noseBox = noseBox;
    diagnostics.leftEyeBox = leftEyeBox;
    diagnostics.rightEyeBox = rightEyeBox;

    if exist("gazeMetrics", "var") && canUpdateLogic
        diagnostics.gazeMetrics = gazeMetrics;
    else
        diagnostics.gazeMetrics = [];
    end

    if exist("eyeMetrics", "var") && canUpdateLogic
        diagnostics.eyeMetrics = eyeMetrics;
    else
        diagnostics.eyeMetrics = [];
    end

    diagnostics.distractionState = distractionState;
    diagnostics.drowsinessState = drowsinessState;

    annotatedFrame = drawDMSOverlay(frame, currentStateLabel, rppgState.lastBpm, diagnostics, config);
    set(imageHandle, "CData", annotatedFrame);
    drawnow limitrate;
end

%% Local helpers
function [faceBox, noseBox, leftEyeBox, rightEyeBox, faceDetected] = ...
        localDetectFaceFeatures(frame, faceDetector, noseDetector, leftEyeDetector, rightEyeDetector, config)
    faceBox = [];
    noseBox = [];
    leftEyeBox = [];
    rightEyeBox = [];
    faceDetected = false;

    grayFrame = rgb2gray(frame);
    detectedFaces = faceDetector(grayFrame);
    if isempty(detectedFaces)
        return;
    end

    [~, biggestIdx] = max(detectedFaces(:, 3) .* detectedFaces(:, 4));
    faceBox = round(detectedFaces(biggestIdx, :));
    faceBox = localClampBox(faceBox, size(grayFrame));
    if isempty(faceBox)
        return;
    end

    faceWidth = faceBox(3);
    faceHeight = faceBox(4);

    eyeSearch = faceBox;
    eyeSearch(1) = faceBox(1) + round(config.facePaddingFrac * faceWidth);
    eyeSearch(3) = round(faceWidth * (1 - 2 * config.facePaddingFrac));
    eyeSearch(2) = faceBox(2) + round(config.eyeSearchTopFrac * faceHeight);
    eyeSearch(4) = round(faceHeight * config.eyeSearchHeightFrac);
    eyeSearch = localClampBox(eyeSearch, size(grayFrame));

    noseSearch = faceBox;
    noseSearch(1) = faceBox(1) + round(0.18 * faceWidth);
    noseSearch(3) = round(faceWidth * 0.64);
    noseSearch(2) = faceBox(2) + round(config.noseSearchTopFrac * faceHeight);
    noseSearch(4) = round(faceHeight * config.noseSearchHeightFrac);
    noseSearch = localClampBox(noseSearch, size(grayFrame));

    leftEyeCandidates = leftEyeDetector(grayFrame, eyeSearch);
    rightEyeCandidates = rightEyeDetector(grayFrame, eyeSearch);
    noseCandidates = noseDetector(grayFrame, noseSearch);

    leftEyeBox = localSelectBestBox(leftEyeCandidates, faceBox, "leftEye");
    rightEyeBox = localSelectBestBox(rightEyeCandidates, faceBox, "rightEye");
    noseBox = localSelectBestBox(noseCandidates, faceBox, "nose");
    faceDetected = ~isempty(faceBox) && ~isempty(noseBox) && ~isempty(leftEyeBox) && ~isempty(rightEyeBox);
end

function bestBox = localSelectBestBox(candidates, faceBox, featureType)
    bestBox = [];
    if isempty(candidates)
        return;
    end

    faceCenterX = faceBox(1) + faceBox(3) / 2;
    faceCenterY = faceBox(2) + faceBox(4) / 2;

    scores = zeros(size(candidates, 1), 1);
    for idx = 1:size(candidates, 1)
        box = candidates(idx, :);
        centerX = box(1) + box(3) / 2;
        centerY = box(2) + box(4) / 2;

        switch featureType
            case "leftEye"
                refX = faceBox(1) + 0.33 * faceBox(3);
                refY = faceBox(2) + 0.32 * faceBox(4);
            case "rightEye"
                refX = faceBox(1) + 0.67 * faceBox(3);
                refY = faceBox(2) + 0.32 * faceBox(4);
            otherwise
                refX = faceCenterX;
                refY = faceBox(2) + 0.58 * faceBox(4);
        end

        distanceScore = hypot(centerX - refX, centerY - refY);
        areaScore = box(3) * box(4);
        scores(idx) = areaScore / max(distanceScore, 1);
    end

    [~, bestIdx] = max(scores);
    bestBox = round(candidates(bestIdx, :));
end

function box = localClampBox(box, imageSize)
    if isempty(box)
        return;
    end

    x = max(1, round(box(1)));
    y = max(1, round(box(2)));
    w = max(1, round(box(3)));
    h = max(1, round(box(4)));

    x2 = min(imageSize(2), x + w - 1);
    y2 = min(imageSize(1), y + h - 1);
    w = x2 - x + 1;
    h = y2 - y + 1;

    if w < 2 || h < 2
        box = [];
    else
        box = [x, y, w, h];
    end
end

function [rgbMean, rppgState] = localUpdateRppgBuffer(frame, faceBox, leftEyeBox, rightEyeBox, timestamp, rppgState, config)
    rgbMean = [];
    if isempty(faceBox)
        return;
    end

    faceX = faceBox(1);
    faceY = faceBox(2);
    faceW = faceBox(3);
    faceH = faceBox(4);

    roiX1 = faceX + round(config.hrRoiSideFrac * faceW);
    roiX2 = faceX + round((1 - config.hrRoiSideFrac) * faceW);
    roiY1 = faceY + round(config.hrRoiTopFrac * faceH);
    roiY2 = faceY + round(config.hrRoiBottomFrac * faceH);

    roiX1 = max(1, min(size(frame, 2), roiX1));
    roiX2 = max(roiX1 + 1, min(size(frame, 2), roiX2));
    roiY1 = max(1, min(size(frame, 1), roiY1));
    roiY2 = max(roiY1 + 1, min(size(frame, 1), roiY2));

    skinRoi = frame(roiY1:roiY2, roiX1:roiX2, :);
    mask = true(size(skinRoi, 1), size(skinRoi, 2));

    eyeBoxes = {leftEyeBox, rightEyeBox};
    for boxIdx = 1:numel(eyeBoxes)
        box = eyeBoxes{boxIdx};
        if isempty(box)
            continue;
        end
        ex1 = max(1, box(1) - roiX1 + 1);
        ex2 = min(size(mask, 2), box(1) + box(3) - roiX1);
        ey1 = max(1, box(2) - roiY1 + 1);
        ey2 = min(size(mask, 1), box(2) + box(4) - roiY1);
        if ex1 < ex2 && ey1 < ey2
            mask(ey1:ey2, ex1:ex2) = false;
        end
    end

    if nnz(mask) < 50
        return;
    end

    rgbMean = zeros(1, 3);
    for channelIdx = 1:3
        channel = double(skinRoi(:, :, channelIdx));
        rgbMean(channelIdx) = mean(channel(mask), "omitnan");
    end

    rppgState.timestamps(end+1, 1) = timestamp;
    rppgState.rgbMeans(end+1, :) = rgbMean;

    keepMask = rppgState.timestamps >= (timestamp - config.hrWindowSec);
    rppgState.timestamps = rppgState.timestamps(keepMask);
    rppgState.rgbMeans = rppgState.rgbMeans(keepMask, :);
end

function localCleanup(cam, figureName)
    if exist("cam", "var") && ~isempty(cam)
        clear cam;
    end
    figHandle = findall(groot, "Type", "figure", "Name", figureName);
    if ~isempty(figHandle)
        delete(figHandle);
    end
end
