clc;
close all;

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
config.calibrationDurationSec = 6;
config.calibrationPrintIntervalSec = 1.0;
config.maxDetectionHoldSec = 0.75;
config.maxForwardBaselineStd = 0.10;
config.hintCooldownSec = 2.5;
config.hintOverlayHoldSec = 2.5;
config.facePaddingFrac = 0.10;
config.eyeSearchTopFrac = 0.12;
config.eyeSearchHeightFrac = 0.42;
config.noseSearchTopFrac = 0.28;
config.noseSearchHeightFrac = 0.38;
config.minFaceWidthFrac = 0.20;
config.minFaceHeightFrac = 0.25;
config.faceBrightnessWarnMin = 0.12;
config.faceContrastWarnMin = 0.05;
config.preferredResolutions = ["1280x720", "1920x1080", "1600x1200", ...
    "1280x960", "1024x768", "800x600", "640x480"];
config.minPreferredCameraArea = 640 * 480;
config.minPreferredColorfulness = 0.03;
config.minCameraAspectRatio = 1.20;
config.minProbeBrightness = 0.15;
config.cameraReleasePauseSec = 0.35;
config.cameraProbeRetryCount = 2;
config.forceCameraIndex = [];

% Gaze thresholds in normalized face coordinates
config.gazeThresholdX = 0.08;
config.gazeThresholdY = 0.10;
config.gazeAspectRatioThreshold = 0.18;
config.recenterAlpha = 0.02;
config.printGazeDebug = true;
config.gazeDebugIntervalSec = 1.0;

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
    error("MATLAB Support Package for USB Webcams is not installed. Install it from Add-On Explorer before running runDMS.m.");
end

try
    availableCameras = webcamlist;
catch webcamListError
    error("Unable to query USB webcams. Check that the MATLAB webcam support package is installed and the camera is not busy.\n%s", ...
        webcamListError.message);
end

if isempty(availableCameras)
    error("No USB webcam detected. Connect a webcam or enable the laptop camera.");
end

%% Initialize webcam and cleanup
cam = [];
viewer = [];
stopRequested = false;
setappdata(0, "runDMSFigureName", config.figureName);
setappdata(0, "runDMSCam", []);
cleanupObj = onCleanup(@localCleanup);

fprintf("[DMS] Available cameras:\n");
for camIdx = 1:numel(availableCameras)
    fprintf("  %d. %s\n", camIdx, string(availableCameras(camIdx)));
end

[cam, cameraInfo] = localOpenPreferredCamera(availableCameras, config);
setappdata(0, "runDMSCam", cam);

fprintf("\n[DMS] Camera selected: %s\n", cameraInfo.name);
fprintf("[DMS] Resolution: %s\n", cameraInfo.resolution);
if ~isempty(cameraInfo.notes)
    fprintf("[DMS] Camera setup: %s\n", strjoin(cellstr(cameraInfo.notes), ", "));
end
if isfield(cameraInfo, "selectionReason") && strlength(cameraInfo.selectionReason) > 0
    fprintf("[DMS] Camera selection: %s\n", cameraInfo.selectionReason);
end
fprintf("[DMS] Calibration started for %.0f seconds.\n", config.calibrationDurationSec);
fprintf("[DMS] Look straight ahead, keep both eyes open, and stay centered.\n");
fprintf("[DMS] Tips: move closer if the face looks small, add front lighting, avoid strong backlight.\n\n");

%% Initialize detectors
faceDetectors = { ...
    vision.CascadeObjectDetector("FrontalFaceCART"), ...
    vision.CascadeObjectDetector("FrontalFaceLBP")};
faceDetectors{1}.MergeThreshold = 4;
faceDetectors{1}.ScaleFactor = 1.05;
faceDetectors{2}.MergeThreshold = 5;
faceDetectors{2}.ScaleFactor = 1.05;

leftEyeDetector = vision.CascadeObjectDetector("LeftEye");
leftEyeDetector.MergeThreshold = 4;

rightEyeDetector = vision.CascadeObjectDetector("RightEye");
rightEyeDetector.MergeThreshold = 4;

eyePairDetector = vision.CascadeObjectDetector("EyePairBig");
eyePairDetector.MergeThreshold = 4;

noseDetector = vision.CascadeObjectDetector("Nose");
noseDetector.MergeThreshold = 8;

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

hintState = struct( ...
    "lastMessage", "", ...
    "lastPrintTime", -inf, ...
    "overlayMessage", "", ...
    "overlayUntil", -inf);

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
    "Color", "k", ...
    "CloseRequestFcn", @localCloseViewer, ...
    "KeyPressFcn", @localHandleKeyPress);

frame = snapshot(cam);
frame = localEnsureRgbFrame(frame);
imageHandle = imshow(frame, "Border", "tight", "InitialMagnification", 100);
axis image off;
movegui(viewer, "center");
drawnow;

%% Initial calibration
calibrationStart = tic;
calibrationClockStart = tic;
lastCalibrationPrintSec = inf;
lastCalibrationMessage = "";
lastGazeDebugTime = -inf;

while ~stopRequested && ishandle(viewer) && toc(calibrationClockStart) < config.calibrationDurationSec
    frame = localEnsureRgbFrame(snapshot(cam));
    nowSec = toc(calibrationStart);
    remainingSec = max(0, config.calibrationDurationSec - toc(calibrationClockStart));

    [faceBox, noseBox, leftEyeBox, rightEyeBox, faceDetected, detectionHint, featureInfo] = ...
        localDetectFaceFeatures(frame, faceDetectors, noseDetector, leftEyeDetector, rightEyeDetector, eyePairDetector, config);

    if faceDetected
        detectionState.faceDetected = true;
        detectionState.lastDetectionTime = nowSec;
        detectionState.lastFaceBox = faceBox;
        detectionState.lastNoseBox = noseBox;
        detectionState.lastLeftEyeBox = leftEyeBox;
        detectionState.lastRightEyeBox = rightEyeBox;

        gazeMetrics = estimateGazeState(faceBox, noseBox, leftEyeBox, rightEyeBox, [], featureInfo, config);
        eyeMetrics = estimateEyeClosure(frame, leftEyeBox, rightEyeBox, [], config);

        gazeBaselineSamples(end+1, :) = [gazeMetrics.noseCenterNorm, gazeMetrics.eyeCenterNorm, gazeMetrics.faceAspectRatio]; %#ok<SAGROW>
        if eyeMetrics.confidence > config.eyeMinConfidence
            eyeBaselineSamples(end+1, :) = [eyeMetrics.leftOpenness, eyeMetrics.rightOpenness]; %#ok<SAGROW>
        end

        [hintState, detectionState.message, shouldPrintHint] = localUpdateHintState( ...
            sprintf("Calibrating... %ds left", max(1, ceil(remainingSec))), nowSec, hintState, 0, 0);
    else
        detectionState.faceDetected = false;
        [hintState, detectionState.message, shouldPrintHint] = localUpdateHintState( ...
            detectionHint.overlayMessage, nowSec, hintState, config.hintCooldownSec, config.hintOverlayHoldSec);
    end

    calibrationTerminalMessage = localBuildCalibrationTerminalMessage(remainingSec, faceDetected, detectionHint);
    currentCalibrationSec = ceil(remainingSec);
    shouldPrintCountdown = currentCalibrationSec ~= lastCalibrationPrintSec && ...
        mod(currentCalibrationSec, config.calibrationPrintIntervalSec) == 0;
    shouldPrintMessage = shouldPrintHint || ~strcmp(calibrationTerminalMessage, lastCalibrationMessage);
    if shouldPrintCountdown || shouldPrintMessage
        fprintf("[Calibration] %s\n", calibrationTerminalMessage);
        lastCalibrationPrintSec = currentCalibrationSec;
        lastCalibrationMessage = calibrationTerminalMessage;
    end

    overlayFrame = drawDMSOverlay(frame, "Focused on the road", NaN, detectionState, config);
    set(imageHandle, "CData", overlayFrame);
    drawnow limitrate;
end

if size(gazeBaselineSamples, 1) < 5
    fprintf(2, "[DMS] Calibration failed: face / eyes / nose were not detected reliably.\n");
    error("Calibration failed because the face, nose and eyes were not detected reliably. Keep the whole face visible and rerun the script.");
end

gazeBaselineMedian = median(gazeBaselineSamples, 1, "omitnan");
gazeBaseline = struct( ...
    "noseCenterNorm", gazeBaselineMedian(1:2), ...
    "eyeCenterNorm", gazeBaselineMedian(3:4), ...
    "faceAspectRatio", gazeBaselineMedian(5));
eyeBaseline = median(eyeBaselineSamples, 1, "omitnan");

if any(isnan(eyeBaseline))
    fprintf(2, "[DMS] Calibration failed: open-eye samples were insufficient.\n");
    error("Calibration failed because open-eye samples were insufficient. Keep both eyes visible and open during calibration.");
end

detectionState.calibrated = true;
detectionState.message = "";
fprintf("[DMS] Calibration completed. Live monitoring started.\n");

%% Main frame loop
mainClock = tic;
lastReportedStateLabel = "";
lastReportedHint = "";
while ~stopRequested && ishandle(viewer)
    if stopRequested
        break;
    end
    frame = localEnsureRgbFrame(snapshot(cam));
    nowSec = toc(mainClock);

    [faceBox, noseBox, leftEyeBox, rightEyeBox, faceDetected, detectionHint, featureInfo] = ...
        localDetectFaceFeatures(frame, faceDetectors, noseDetector, leftEyeDetector, rightEyeDetector, eyePairDetector, config);

    if faceDetected
        detectionState.faceDetected = true;
        detectionState.lastDetectionTime = nowSec;
        detectionState.lastFaceBox = faceBox;
        detectionState.lastNoseBox = noseBox;
        detectionState.lastLeftEyeBox = leftEyeBox;
        detectionState.lastRightEyeBox = rightEyeBox;
        [hintState, detectionState.message] = localUpdateHintState("", nowSec, hintState, config.hintCooldownSec, config.hintOverlayHoldSec);
    else
        detectionState.faceDetected = false;
        if nowSec - detectionState.lastDetectionTime > config.maxDetectionHoldSec
            [hintState, detectionState.message] = localUpdateHintState( ...
                detectionHint.overlayMessage, nowSec, hintState, config.hintCooldownSec, config.hintOverlayHoldSec);
        else
            faceBox = detectionState.lastFaceBox;
            noseBox = detectionState.lastNoseBox;
            leftEyeBox = detectionState.lastLeftEyeBox;
            rightEyeBox = detectionState.lastRightEyeBox;
            [hintState, detectionState.message] = localUpdateHintState("", nowSec, hintState, config.hintCooldownSec, config.hintOverlayHoldSec);
        end
    end

    canUpdateLogic = ~isempty(faceBox) && ~isempty(noseBox) && ~isempty(leftEyeBox) && ...
        ~isempty(rightEyeBox) && (nowSec - detectionState.lastDetectionTime <= config.maxDetectionHoldSec);

    if canUpdateLogic
        gazeMetrics = estimateGazeState(faceBox, noseBox, leftEyeBox, rightEyeBox, gazeBaseline, featureInfo, config);
        eyeMetrics = estimateEyeClosure(frame, leftEyeBox, rightEyeBox, eyeBaseline, config);

        % Slowly adapt baseline only when the driver is likely looking forward.
        if gazeMetrics.isForward && eyeMetrics.confidence > config.eyeMinConfidence
            gazeBaseline.noseCenterNorm = (1 - config.recenterAlpha) .* gazeBaseline.noseCenterNorm + ...
                config.recenterAlpha .* gazeMetrics.noseCenterNorm;
            if all(~isnan(gazeMetrics.eyeCenterNorm))
                gazeBaseline.eyeCenterNorm = (1 - config.recenterAlpha) .* gazeBaseline.eyeCenterNorm + ...
                    config.recenterAlpha .* gazeMetrics.eyeCenterNorm;
            end
            gazeBaseline.faceAspectRatio = (1 - config.recenterAlpha) .* gazeBaseline.faceAspectRatio + ...
                config.recenterAlpha .* gazeMetrics.faceAspectRatio;
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

        if config.printGazeDebug && (nowSec - lastGazeDebugTime >= config.gazeDebugIntervalSec)
            fprintf("[Gaze] signal=%s noseOffset=[%.3f %.3f] eyeOffset=[%.3f %.3f] aspectDelta=%.3f thresholds=[%.3f %.3f] isAway=%d\n", ...
                string(gazeMetrics.usedSignal), gazeMetrics.noseOffset(1), gazeMetrics.noseOffset(2), ...
                gazeMetrics.eyeOffset(1), gazeMetrics.eyeOffset(2), gazeMetrics.aspectDelta, ...
                gazeMetrics.thresholdX, gazeMetrics.thresholdY, gazeMetrics.isAway);
            lastGazeDebugTime = nowSec;
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

    if ~strcmp(currentStateLabel, lastReportedStateLabel)
        fprintf("[State] %s\n", currentStateLabel);
        lastReportedStateLabel = currentStateLabel;
    end

    hintToPrint = string(detectionState.message);
    if strlength(hintToPrint) > 0 && ~strcmp(hintToPrint, lastReportedHint)
        fprintf("[Hint] %s\n", hintToPrint);
        lastReportedHint = hintToPrint;
    elseif strlength(hintToPrint) == 0
        lastReportedHint = "";
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
function [faceBox, noseBox, leftEyeBox, rightEyeBox, faceDetected, detectionHint, featureInfo] = ...
        localDetectFaceFeatures(frame, faceDetectors, noseDetector, leftEyeDetector, rightEyeDetector, eyePairDetector, config)
    faceBox = [];
    noseBox = [];
    leftEyeBox = [];
    rightEyeBox = [];
    faceDetected = false;
    featureInfo = struct( ...
        "faceVisible", false, ...
        "leftEyeDetected", false, ...
        "rightEyeDetected", false, ...
        "noseDetected", false, ...
        "eyePairDetected", false, ...
        "noseEstimated", false, ...
        "eyeMirrored", false, ...
        "faceBrightness", NaN, ...
        "faceContrast", NaN);
    detectionHint = struct( ...
        "overlayMessage", "Center your face and improve lighting", ...
        "terminalMessage", "Face not detected. Center your face, move closer, and improve front lighting.");

    grayFrame = rgb2gray(frame);
    detectionFrame = localPrepareDetectionFrame(grayFrame);
    detectedFaces = localDetectFaces(faceDetectors, grayFrame, detectionFrame);
    if isempty(detectedFaces)
        return;
    end

    [~, biggestIdx] = max(detectedFaces(:, 3) .* detectedFaces(:, 4));
    faceBox = round(detectedFaces(biggestIdx, :));
    faceBox = localClampBox(faceBox, size(grayFrame));
    if isempty(faceBox)
        return;
    end

    featureInfo.faceVisible = true;
    [featureInfo.faceBrightness, featureInfo.faceContrast] = localComputeFaceLighting(grayFrame, faceBox);

    faceWidth = faceBox(3);
    faceHeight = faceBox(4);
    faceTooSmall = faceWidth < config.minFaceWidthFrac * size(grayFrame, 2) || ...
        faceHeight < config.minFaceHeightFrac * size(grayFrame, 1);

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

    leftEyeCandidates = localDetectInRegion(leftEyeDetector, detectionFrame, eyeSearch);
    rightEyeCandidates = localDetectInRegion(rightEyeDetector, detectionFrame, eyeSearch);
    eyePairCandidates = localDetectInRegion(eyePairDetector, detectionFrame, eyeSearch);
    noseCandidates = localDetectInRegion(noseDetector, detectionFrame, noseSearch);

    leftEyeBox = localSelectBestBox(leftEyeCandidates, faceBox, "leftEye");
    rightEyeBox = localSelectBestBox(rightEyeCandidates, faceBox, "rightEye");
    eyePairBox = localSelectBestBox(eyePairCandidates, faceBox, "eyePair");
    featureInfo.leftEyeDetected = ~isempty(leftEyeBox);
    featureInfo.rightEyeDetected = ~isempty(rightEyeBox);
    featureInfo.eyePairDetected = ~isempty(eyePairBox);
    [leftEyeBox, rightEyeBox] = localFillEyesFromPair(leftEyeBox, rightEyeBox, eyePairBox);
    [leftEyeBox, rightEyeBox, eyeMirrored] = localMirrorSingleEye(leftEyeBox, rightEyeBox, faceBox, size(grayFrame));
    featureInfo.eyeMirrored = eyeMirrored;
    noseBox = localSelectBestBox(noseCandidates, faceBox, "nose");
    featureInfo.noseDetected = ~isempty(noseBox);
    if isempty(noseBox)
        noseBox = localEstimateNoseBox(faceBox, leftEyeBox, rightEyeBox, size(grayFrame));
        featureInfo.noseEstimated = ~isempty(noseBox);
    end
    faceDetected = ~isempty(faceBox) && ~isempty(noseBox) && ~isempty(leftEyeBox) && ~isempty(rightEyeBox);

    if faceDetected
        detectionHint.overlayMessage = "";
        detectionHint.terminalMessage = "";
        return;
    end

    if faceTooSmall
        detectionHint.overlayMessage = "Move closer to the camera";
        detectionHint.terminalMessage = "Face detected but too small. Move closer to the camera and keep your head centered.";
        return;
    end

    missingParts = strings(0, 1);
    if isempty(leftEyeBox) || isempty(rightEyeBox)
        missingParts(end+1) = "eyes";
    end
    if isempty(noseBox)
        missingParts(end+1) = "nose";
    end

    poorLighting = featureInfo.faceBrightness < config.faceBrightnessWarnMin || ...
        featureInfo.faceContrast < config.faceContrastWarnMin;

    if isempty(missingParts)
        detectionHint.overlayMessage = "Hold still and keep looking forward";
        detectionHint.terminalMessage = "Face detected, but feature tracking is unstable. Hold still and keep looking forward.";
    else
        if poorLighting
            detectionHint.overlayMessage = "Too dark: face a brighter light";
            detectionHint.terminalMessage = "Face found, but the image is too dark to track features well. Use more front lighting and avoid backlight.";
        else
            detectionHint.overlayMessage = "Hold still and keep face frontal";
            detectionHint.terminalMessage = sprintf("Face detected, but %s tracking is unstable. Hold still for a moment and keep your face frontal.", ...
                strjoin(cellstr(missingParts), " and "));
        end
    end
end

function preparedFrame = localPrepareDetectionFrame(grayFrame)
    preparedFrame = imadjust(grayFrame);
    try
        preparedFrame = adapthisteq(preparedFrame);
    catch
    end
end

function detectedFaces = localDetectFaces(faceDetectors, grayFrame, detectionFrame)
    detectedFaces = zeros(0, 4);
    for detectorIdx = 1:numel(faceDetectors)
        detector = faceDetectors{detectorIdx};
        candidates = detector(grayFrame);
        if ~isempty(candidates)
            detectedFaces = [detectedFaces; candidates]; %#ok<AGROW>
        end
        enhancedCandidates = detector(detectionFrame);
        if ~isempty(enhancedCandidates)
            detectedFaces = [detectedFaces; enhancedCandidates]; %#ok<AGROW>
        end
    end
end

function [brightness, contrastValue] = localComputeFaceLighting(grayFrame, faceBox)
    x1 = faceBox(1);
    y1 = faceBox(2);
    x2 = min(size(grayFrame, 2), x1 + faceBox(3) - 1);
    y2 = min(size(grayFrame, 1), y1 + faceBox(4) - 1);
    facePatch = im2double(grayFrame(y1:y2, x1:x2));
    brightness = mean(facePatch(:), "omitnan");
    contrastValue = std(facePatch(:), 0, "omitnan");
end

function candidates = localDetectInRegion(detector, grayFrame, searchBox)
    candidates = zeros(0, 4);
    if isempty(searchBox)
        return;
    end

    x1 = searchBox(1);
    y1 = searchBox(2);
    x2 = x1 + searchBox(3) - 1;
    y2 = y1 + searchBox(4) - 1;
    roi = grayFrame(y1:y2, x1:x2);
    roiCandidates = detector(roi);
    if isempty(roiCandidates)
        return;
    end

    candidates = round(roiCandidates);
    candidates(:, 1) = candidates(:, 1) + x1 - 1;
    candidates(:, 2) = candidates(:, 2) + y1 - 1;
end

function rgbFrame = localEnsureRgbFrame(frame)
    if ndims(frame) == 2
        rgbFrame = repmat(frame, 1, 1, 3);
    elseif size(frame, 3) == 1
        rgbFrame = repmat(frame, 1, 1, 3);
    else
        rgbFrame = frame;
    end
end

function [hintState, overlayMessage, shouldPrint] = localUpdateHintState(requestedMessage, timestamp, hintState, cooldownSec, overlayHoldSec)
    shouldPrint = false;
    requestedMessage = string(requestedMessage);

    if strlength(requestedMessage) == 0
        if timestamp >= hintState.overlayUntil
            hintState.overlayMessage = "";
        end
        overlayMessage = hintState.overlayMessage;
        return;
    end

    if requestedMessage ~= hintState.lastMessage || (timestamp - hintState.lastPrintTime) >= cooldownSec
        hintState.lastMessage = requestedMessage;
        hintState.lastPrintTime = timestamp;
        hintState.overlayMessage = requestedMessage;
        hintState.overlayUntil = timestamp + overlayHoldSec;
        shouldPrint = true;
    elseif timestamp < hintState.overlayUntil
        hintState.overlayMessage = requestedMessage;
    end

    if timestamp >= hintState.overlayUntil && requestedMessage ~= hintState.overlayMessage
        hintState.overlayMessage = "";
    end

    overlayMessage = hintState.overlayMessage;
end

function [leftEyeBox, rightEyeBox] = localFillEyesFromPair(leftEyeBox, rightEyeBox, eyePairBox)
    if isempty(eyePairBox)
        return;
    end

    splitWidth = round(eyePairBox(3) / 2);
    leftFallback = [eyePairBox(1), eyePairBox(2), splitWidth, eyePairBox(4)];
    rightFallback = [eyePairBox(1) + splitWidth, eyePairBox(2), eyePairBox(3) - splitWidth, eyePairBox(4)];

    if isempty(leftEyeBox)
        leftEyeBox = leftFallback;
    end
    if isempty(rightEyeBox)
        rightEyeBox = rightFallback;
    end
end

function [leftEyeBox, rightEyeBox, eyeMirrored] = localMirrorSingleEye(leftEyeBox, rightEyeBox, faceBox, imageSize)
    eyeMirrored = false;
    faceCenterX = faceBox(1) + faceBox(3) / 2;

    if isempty(leftEyeBox) && ~isempty(rightEyeBox)
        mirrorCenterX = 2 * faceCenterX - (rightEyeBox(1) + rightEyeBox(3) / 2);
        leftEyeBox = round([mirrorCenterX - rightEyeBox(3) / 2, rightEyeBox(2), rightEyeBox(3), rightEyeBox(4)]);
        leftEyeBox = localClampBox(leftEyeBox, imageSize);
        eyeMirrored = true;
    elseif isempty(rightEyeBox) && ~isempty(leftEyeBox)
        mirrorCenterX = 2 * faceCenterX - (leftEyeBox(1) + leftEyeBox(3) / 2);
        rightEyeBox = round([mirrorCenterX - leftEyeBox(3) / 2, leftEyeBox(2), leftEyeBox(3), leftEyeBox(4)]);
        rightEyeBox = localClampBox(rightEyeBox, imageSize);
        eyeMirrored = true;
    end
end

function noseBox = localEstimateNoseBox(faceBox, leftEyeBox, rightEyeBox, imageSize)
    noseBox = [];
    if isempty(faceBox)
        return;
    end

    estimatedCenterX = faceBox(1) + 0.50 * faceBox(3);
    if ~isempty(leftEyeBox) && ~isempty(rightEyeBox)
        leftCenter = leftEyeBox(1) + leftEyeBox(3) / 2;
        rightCenter = rightEyeBox(1) + rightEyeBox(3) / 2;
        estimatedCenterX = 0.5 * (leftCenter + rightCenter);
    elseif ~isempty(leftEyeBox)
        estimatedCenterX = leftEyeBox(1) + leftEyeBox(3) / 2 + 0.16 * faceBox(3);
    elseif ~isempty(rightEyeBox)
        estimatedCenterX = rightEyeBox(1) + rightEyeBox(3) / 2 - 0.16 * faceBox(3);
    end

    noseBox = round([ ...
        estimatedCenterX - 0.13 * faceBox(3), ...
        faceBox(2) + 0.45 * faceBox(4), ...
        0.26 * faceBox(3), ...
        0.20 * faceBox(4)]);
    noseBox = localClampBox(noseBox, imageSize);
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
            case "eyePair"
                refX = faceCenterX;
                refY = faceBox(2) + 0.28 * faceBox(4);
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

function [cam, cameraInfo] = localOpenPreferredCamera(availableCameras, config)
    if ~isempty(config.forceCameraIndex)
        cam = localOpenCamera(config.forceCameraIndex, config);
        cameraInfo = localConfigureCamera(cam, config);
        cameraInfo.selectionReason = sprintf("forced camera index %d from config", config.forceCameraIndex);
        return;
    end

    sortedCameras = localSortCameraNames(availableCameras);
    bestCameraName = string(sortedCameras(1));
    bestScore = -inf;
    bestInfo = struct();
    bestAcceptableFound = false;
    probeFailures = strings(0, 1);

    for camIdx = 1:numel(sortedCameras)
        cameraName = string(sortedCameras(camIdx));
        try
            [tempInfo, metrics] = localProbeSingleCamera(cameraName, config);
            isAcceptable = localIsAcceptableCamera(metrics, config);
            score = metrics.score;

            fprintf("[DMS] Camera %d probe: %s, %s, brightness %.2f, color %.3f, score %.0f\n", ...
                camIdx, cameraName, tempInfo.resolution, ...
                metrics.brightness, metrics.colorfulness, score);

            if isAcceptable && (~bestAcceptableFound || score > bestScore)
                bestScore = score;
                bestCameraName = cameraName;
                bestInfo = tempInfo;
                bestInfo.selectionReason = sprintf("picked %s after RGB/resolution probe", cameraName);
                bestAcceptableFound = true;
            elseif ~bestAcceptableFound && score > bestScore
                bestScore = score;
                bestCameraName = cameraName;
                bestInfo = tempInfo;
                bestInfo.selectionReason = sprintf("fallback pick %s because no fully acceptable camera succeeded", cameraName);
            end
        catch probeError
            fprintf(2, "[DMS] Camera %d probe failed: %s\n", camIdx, probeError.message);
            probeFailures(end+1, 1) = cameraName; %#ok<AGROW>
        end
    end

    if ~bestAcceptableFound && ~isempty(probeFailures)
        pause(config.cameraReleasePauseSec);
        for failureIdx = 1:numel(probeFailures)
            cameraName = probeFailures(failureIdx);
            try
                [tempInfo, metrics] = localProbeSingleCamera(cameraName, config);
                if localIsAcceptableCamera(metrics, config)
                    bestCameraName = cameraName;
                    bestInfo = tempInfo;
                    bestInfo.selectionReason = sprintf("picked %s after retrying a busy camera", cameraName);
                    break;
                end
            catch
            end
        end
    end

    if ~bestAcceptableFound
        error(["No suitable RGB webcam could be opened. The detected devices look grayscale/low-resolution " ...
            "or are busy. Close other apps using the webcam or set config.forceCameraIndex to the correct RGB camera."]);
    end

    cam = localOpenCamera(char(bestCameraName), config);
    cameraInfo = localConfigureCamera(cam, config);
    if isempty(fieldnames(bestInfo))
        cameraInfo.selectionReason = sprintf("defaulted to %s", bestCameraName);
    elseif isfield(bestInfo, "selectionReason")
        cameraInfo.selectionReason = bestInfo.selectionReason;
    else
        cameraInfo.selectionReason = sprintf("picked %s", bestCameraName);
    end
end

function [cameraInfo, metrics] = localProbeSingleCamera(cameraName, config)
    lastError = [];
    for attemptIdx = 1:config.cameraProbeRetryCount
        probeCam = [];
        try
            probeCam = webcam(char(cameraName));
            cameraInfo = localConfigureCamera(probeCam, config);
            frame = localEnsureRgbFrame(snapshot(probeCam));
            metrics = localEvaluateCameraFrame(frame, config);
            probeCam = [];
            pause(config.cameraReleasePauseSec);
            return;
        catch probeError
            lastError = probeError;
            probeCam = [];
            pause(config.cameraReleasePauseSec);
        end
    end

    rethrow(lastError);
end

function cam = localOpenCamera(cameraIdentifier, config)
    lastError = [];
    for attemptIdx = 1:config.cameraProbeRetryCount
        try
            cam = webcam(cameraIdentifier);
            pause(config.cameraReleasePauseSec);
            return;
        catch openError
            lastError = openError;
            pause(config.cameraReleasePauseSec);
        end
    end

    rethrow(lastError);
end

function metrics = localEvaluateCameraFrame(frame, config)
    if isa(frame, "uint8") || isa(frame, "uint16")
        frameDouble = im2double(frame);
    else
        frameDouble = double(frame);
        if max(frameDouble(:)) > 1
            frameDouble = frameDouble ./ 255;
        end
    end

    metrics = struct( ...
        "height", size(frameDouble, 1), ...
        "width", size(frameDouble, 2), ...
        "brightness", 0, ...
        "colorfulness", 0, ...
        "aspectRatio", size(frameDouble, 2) / max(size(frameDouble, 1), 1), ...
        "score", 0);

    metrics.brightness = mean(frameDouble(:), "omitnan");
    pixelArea = size(frameDouble, 1) * size(frameDouble, 2);

    if ndims(frameDouble) >= 3 && size(frameDouble, 3) >= 3
        redChannel = frameDouble(:, :, 1);
        greenChannel = frameDouble(:, :, 2);
        blueChannel = frameDouble(:, :, 3);
        metrics.colorfulness = mean(abs(redChannel(:) - greenChannel(:)), "omitnan") + ...
            mean(abs(greenChannel(:) - blueChannel(:)), "omitnan") + ...
            mean(abs(redChannel(:) - blueChannel(:)), "omitnan");
    end

    brightnessScore = 1 - min(abs(metrics.brightness - 0.55), 0.55) / 0.55;
    resolutionScore = pixelArea / (640 * 480);
    colorScore = min(metrics.colorfulness / 0.08, 1.5);

    metrics.score = 1000 * resolutionScore + 400 * brightnessScore + 600 * colorScore;

    if pixelArea < config.minPreferredCameraArea
        metrics.score = metrics.score - 500;
    end
    if metrics.colorfulness < config.minPreferredColorfulness
        metrics.score = metrics.score - 250;
    end
    if metrics.brightness < 0.15
        metrics.score = metrics.score - 250;
    end
    if metrics.aspectRatio < config.minCameraAspectRatio
        metrics.score = metrics.score - 400;
    end
end

function isAcceptable = localIsAcceptableCamera(metrics, config)
    isAcceptable = metrics.colorfulness >= config.minPreferredColorfulness && ...
        metrics.height * metrics.width >= config.minPreferredCameraArea && ...
        metrics.aspectRatio >= config.minCameraAspectRatio && ...
        metrics.brightness >= config.minProbeBrightness;
end

function sortedNames = localSortCameraNames(cameraNames)
    cameraNames = string(cameraNames);
    priorities = zeros(numel(cameraNames), 1);
    for idx = 1:numel(cameraNames)
        tokens = regexp(cameraNames(idx), "video(\d+)$", "tokens", "once");
        if isempty(tokens)
            priorities(idx) = 1e6 + idx;
        else
            priorities(idx) = str2double(tokens{1});
        end
    end
    [~, order] = sort(priorities, "ascend");
    sortedNames = cameraNames(order);
end

function cameraInfo = localConfigureCamera(cam, config)
    cameraInfo = struct("name", string(cam.Name), "resolution", "unknown", "notes", strings(0, 1));

    if isprop(cam, "AvailableResolutions")
        availableResolutions = string(cam.AvailableResolutions);
        chosenResolution = localChooseResolution(availableResolutions, config.preferredResolutions);
        try
            cam.Resolution = char(chosenResolution);
        catch
        end
    end

    if isprop(cam, "Resolution")
        cameraInfo.resolution = string(cam.Resolution);
    end

    cameraInfo.notes = strings(0, 1);
    if isprop(cam, "FocusMode")
        try
            cam.FocusMode = "auto";
            cameraInfo.notes(end+1) = "autofocus enabled";
        catch
        end
    end
    if isprop(cam, "ExposureMode")
        try
            cam.ExposureMode = "auto";
            cameraInfo.notes(end+1) = "auto exposure enabled";
        catch
        end
    end
    if isprop(cam, "WhiteBalanceMode")
        try
            cam.WhiteBalanceMode = "auto";
            cameraInfo.notes(end+1) = "auto white balance enabled";
        catch
        end
    end
end

function chosenResolution = localChooseResolution(availableResolutions, preferredResolutions)
    if isempty(availableResolutions)
        chosenResolution = "";
        return;
    end

    chosenResolution = availableResolutions(1);

    for prefIdx = 1:numel(preferredResolutions)
        matchIdx = find(availableResolutions == preferredResolutions(prefIdx), 1);
        if ~isempty(matchIdx)
            chosenResolution = availableResolutions(matchIdx);
            return;
        end
    end

    bestArea = -inf;
    for resIdx = 1:numel(availableResolutions)
        tokens = regexp(availableResolutions(resIdx), "(\d+)x(\d+)", "tokens", "once");
        if isempty(tokens)
            continue;
        end
        area = str2double(tokens{1}) * str2double(tokens{2});
        if area > bestArea
            bestArea = area;
            chosenResolution = availableResolutions(resIdx);
        end
    end
end

function message = localBuildCalibrationTerminalMessage(remainingSec, faceDetected, detectionHint)
    if faceDetected
        message = sprintf("%ds left. Good pose: keep looking forward with both eyes open.", ...
            max(1, ceil(remainingSec)));
    else
        message = sprintf("%ds left. %s", max(1, ceil(remainingSec)), detectionHint.terminalMessage);
    end
end

function localHandleKeyPress(~, event)
    if strcmp(event.Key, "escape")
        stopRequested = true;
        if ~isempty(viewer) && ishandle(viewer)
            delete(viewer);
        end
    end
end

function localCloseViewer(src, ~)
    stopRequested = true;
    delete(src);
end

function localCleanup
    figureName = "";
    if isappdata(0, "runDMSFigureName")
        figureName = getappdata(0, "runDMSFigureName");
    end
    if strlength(figureName) > 0
        figHandle = findall(groot, "Type", "figure", "Name", figureName);
        if ~isempty(figHandle)
            delete(figHandle);
        end
    end

    if isappdata(0, "runDMSCam")
        camObj = getappdata(0, "runDMSCam");
        rmappdata(0, "runDMSCam");
        if ~isempty(camObj)
            clear camObj;
        end
    end
    if isappdata(0, "runDMSFigureName")
        rmappdata(0, "runDMSFigureName");
    end
end

