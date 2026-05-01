function eyeMetrics = estimateEyeClosure(frame, leftEyeBox, rightEyeBox, eyeBaseline, config)
%ESTIMATEEYECLOSURE Estimate whether both eyes are closed.
%   Without dense landmarks, openness is approximated from the apparent
%   vertical structure and darkness distribution inside the eye ROIs.

eyeMetrics = struct( ...
    "leftOpenness", NaN, ...
    "rightOpenness", NaN, ...
    "leftRatio", NaN, ...
    "rightRatio", NaN, ...
    "confidence", 0, ...
    "bothEyesClosed", false);

if isempty(leftEyeBox) || isempty(rightEyeBox)
    return;
end

[leftOpenness, leftConfidence] = localEyeOpenness(frame, leftEyeBox, config);
[rightOpenness, rightConfidence] = localEyeOpenness(frame, rightEyeBox, config);

eyeMetrics.leftOpenness = leftOpenness;
eyeMetrics.rightOpenness = rightOpenness;
eyeMetrics.confidence = min(leftConfidence, rightConfidence);

if any(isnan(eyeBaseline)) || any(eyeBaseline <= 0)
    return;
end

eyeMetrics.leftRatio = leftOpenness / eyeBaseline(1);
eyeMetrics.rightRatio = rightOpenness / eyeBaseline(2);

leftClosed = eyeMetrics.leftRatio < config.eyeClosedRatio;
rightClosed = eyeMetrics.rightRatio < config.eyeClosedRatio;

eyeMetrics.bothEyesClosed = leftClosed && rightClosed && ...
    eyeMetrics.confidence >= config.eyeMinConfidence;

function [openness, confidence] = localEyeOpenness(frame, eyeBox, config)
    grayFrame = rgb2gray(frame);

    x1 = max(1, eyeBox(1));
    y1 = max(1, eyeBox(2));
    x2 = min(size(grayFrame, 2), eyeBox(1) + eyeBox(3) - 1);
    y2 = min(size(grayFrame, 1), eyeBox(2) + eyeBox(4) - 1);
    eyePatch = grayFrame(y1:y2, x1:x2);

    if numel(eyePatch) < 25
        openness = NaN;
        confidence = 0;
        return;
    end

    eyePatch = im2double(eyePatch);
    eyePatch = imgaussfilt(eyePatch, 0.8);

    rowContrast = std(eyePatch, 0, 2);
    [~, peakRow] = max(rowContrast);

    rowThreshold = max(rowContrast) * 0.50;
    activeRows = find(rowContrast >= rowThreshold);
    if isempty(activeRows)
        activeRows = peakRow;
    end

    structuralHeight = (max(activeRows) - min(activeRows) + 1) / size(eyePatch, 1);

    intensityThreshold = graythresh(eyePatch);
    darkMask = eyePatch < max(intensityThreshold, config.eyeDarkPixelThreshold);
    darkRows = any(darkMask, 2);
    darkHeight = nnz(darkRows) / size(eyePatch, 1);

    openness = 0.7 * structuralHeight + 0.3 * darkHeight;
    confidence = min(1, max(rowContrast) / 0.12);

