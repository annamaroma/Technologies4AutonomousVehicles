function gazeMetrics = estimateGazeState(faceBox, noseBox, leftEyeBox, rightEyeBox, gazeBaseline, featureInfo, config)
%ESTIMATEGAZESTATE Estimate forward or away head pose from face features.
%   The primary cue is the nose center normalized inside the face box.
%   When the nose is unreliable, the eye-center position and feature-loss
%   cues provide a fallback so clear head turns are still detected.

gazeMetrics = struct( ...
    "noseCenterNorm", [NaN, NaN], ...
    "eyeCenterNorm", [NaN, NaN], ...
    "offset", [NaN, NaN], ...
    "noseOffset", [NaN, NaN], ...
    "eyeOffset", [NaN, NaN], ...
    "faceAspectRatio", NaN, ...
    "aspectDelta", NaN, ...
    "usedSignal", "none", ...
    "featureLossAway", false, ...
    "isForward", false, ...
    "isAway", false, ...
    "confidence", 0, ...
    "thresholdX", config.gazeThresholdX, ...
    "thresholdY", config.gazeThresholdY);

if isempty(faceBox)
    return;
end

faceCenter = [faceBox(1) + faceBox(3) / 2, faceBox(2) + faceBox(4) / 2];
gazeMetrics.faceAspectRatio = faceBox(3) / max(faceBox(4), eps);

if ~isempty(noseBox)
    noseCenter = [noseBox(1) + noseBox(3) / 2, noseBox(2) + noseBox(4) / 2];
    gazeMetrics.noseCenterNorm = [(noseCenter(1) - faceCenter(1)) / faceBox(3), ...
        (noseCenter(2) - faceCenter(2)) / faceBox(4)];
end

eyeCenters = zeros(0, 2);
if ~isempty(leftEyeBox)
    eyeCenters(end+1, :) = [leftEyeBox(1) + leftEyeBox(3) / 2, leftEyeBox(2) + leftEyeBox(4) / 2]; %#ok<AGROW>
end
if ~isempty(rightEyeBox)
    eyeCenters(end+1, :) = [rightEyeBox(1) + rightEyeBox(3) / 2, rightEyeBox(2) + rightEyeBox(4) / 2]; %#ok<AGROW>
end
if ~isempty(eyeCenters)
    eyeCenter = mean(eyeCenters, 1);
    gazeMetrics.eyeCenterNorm = [(eyeCenter(1) - faceCenter(1)) / faceBox(3), ...
        (eyeCenter(2) - faceCenter(2)) / faceBox(4)];
end

if isempty(gazeBaseline)
    referenceNose = [0, 0];
    referenceEye = [0, 0];
    referenceAspect = gazeMetrics.faceAspectRatio;
elseif isstruct(gazeBaseline)
    referenceNose = gazeBaseline.noseCenterNorm;
    referenceEye = gazeBaseline.eyeCenterNorm;
    referenceAspect = gazeBaseline.faceAspectRatio;
else
    referenceNose = gazeBaseline;
    referenceEye = gazeBaseline;
    referenceAspect = gazeMetrics.faceAspectRatio;
end

gazeMetrics.noseOffset = gazeMetrics.noseCenterNorm - referenceNose;
gazeMetrics.eyeOffset = gazeMetrics.eyeCenterNorm - referenceEye;
gazeMetrics.aspectDelta = abs(gazeMetrics.faceAspectRatio - referenceAspect);

signalOffset = gazeMetrics.noseOffset;
if all(~isnan(signalOffset))
    gazeMetrics.usedSignal = "nose";
elseif all(~isnan(gazeMetrics.eyeOffset))
    signalOffset = gazeMetrics.eyeOffset;
    gazeMetrics.usedSignal = "eyes";
else
    signalOffset = [NaN, NaN];
end
gazeMetrics.offset = signalOffset;

featureLossAway = false;
if ~isempty(featureInfo)
    leftEyeReliable = featureInfo.leftEyeDetected || featureInfo.eyePairDetected;
    rightEyeReliable = featureInfo.rightEyeDetected || featureInfo.eyePairDetected;
    missingEyeCount = double(~leftEyeReliable) + double(~rightEyeReliable);
    featureLossAway = featureInfo.faceVisible && featureInfo.faceBrightness >= config.faceBrightnessWarnMin && ...
        (missingEyeCount >= 1 || featureInfo.eyeMirrored || ...
        (featureInfo.noseEstimated && missingEyeCount >= 1));
end
gazeMetrics.featureLossAway = featureLossAway;

distanceRatio = 0;
if all(~isnan(signalOffset))
    normX = abs(signalOffset(1)) / max(config.gazeThresholdX, eps);
    normY = abs(signalOffset(2)) / max(config.gazeThresholdY, eps);
    distanceRatio = max(normX, normY);
end

aspectAway = gazeMetrics.aspectDelta > config.gazeAspectRatioThreshold;
offsetAway = all(~isnan(signalOffset)) && (abs(signalOffset(1)) > config.gazeThresholdX || ...
    abs(signalOffset(2)) > config.gazeThresholdY);

gazeMetrics.confidence = min(1, max([distanceRatio, ...
    0.70 * double(featureLossAway), ...
    0.80 * double(aspectAway)]));

gazeMetrics.isAway = offsetAway || aspectAway || featureLossAway;
gazeMetrics.isForward = ~gazeMetrics.isAway;
end
