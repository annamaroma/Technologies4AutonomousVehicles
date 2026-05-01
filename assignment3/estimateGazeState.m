function gazeMetrics = estimateGazeState(faceBox, noseBox, gazeBaseline, config)
%ESTIMATEGAZESTATE Estimate forward or away gaze from nose position.
%   Nose center is normalized inside the face bounding box so the
%   thresholds remain approximately scale-invariant across distances.

gazeMetrics = struct( ...
    "noseCenterNorm", [NaN, NaN], ...
    "offset", [NaN, NaN], ...
    "isForward", false, ...
    "isAway", false, ...
    "confidence", 0);

if isempty(faceBox) || isempty(noseBox)
    return;
end

faceCenter = [faceBox(1) + faceBox(3) / 2, faceBox(2) + faceBox(4) / 2];
noseCenter = [noseBox(1) + noseBox(3) / 2, noseBox(2) + noseBox(4) / 2];

noseCenterNorm = [(noseCenter(1) - faceCenter(1)) / faceBox(3), ...
    (noseCenter(2) - faceCenter(2)) / faceBox(4)];
gazeMetrics.noseCenterNorm = noseCenterNorm;

if isempty(gazeBaseline)
    reference = [0, 0];
else
    reference = gazeBaseline;
end

offset = noseCenterNorm - reference;
gazeMetrics.offset = offset;

normX = abs(offset(1)) / config.gazeThresholdX;
normY = abs(offset(2)) / config.gazeThresholdY;
distanceRatio = max(normX, normY);
gazeMetrics.confidence = min(1, distanceRatio);

gazeMetrics.isAway = abs(offset(1)) > config.gazeThresholdX || ...
    abs(offset(2)) > config.gazeThresholdY;
gazeMetrics.isForward = ~gazeMetrics.isAway;

