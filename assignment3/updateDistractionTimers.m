function distractionState = updateDistractionTimers(timestamp, isAway, distractionState, config)
%UPDATEDISTRACTIONTIMERS Update long/short owl distraction timers.

if isnan(distractionState.lastTimestamp)
    distractionState.lastTimestamp = timestamp;
    if isAway
        distractionState.currentAwayStart = timestamp;
    end
    return;
end

dt = max(0, timestamp - distractionState.lastTimestamp);
distractionState.lastTimestamp = timestamp;

if isAway
    distractionState.awayDuration = distractionState.awayDuration + dt;
    distractionState.returnDuration = 0;
    if isnan(distractionState.currentAwayStart)
        distractionState.currentAwayStart = max(0, timestamp - dt);
    end
else
    if ~isnan(distractionState.currentAwayStart)
        distractionState.awayIntervals(end+1, :) = [distractionState.currentAwayStart, timestamp]; %#ok<AGROW>
        distractionState.currentAwayStart = NaN;
    end
    distractionState.awayDuration = 0;
    distractionState.returnDuration = distractionState.returnDuration + dt;
end

if distractionState.awayDuration >= config.longDistractionSec
    distractionState.longActive = true;
end

if distractionState.longActive && ~isAway
    distractionState.longActive = false;
end

windowStart = max(0, timestamp - config.shortWindowSec);
intervals = distractionState.awayIntervals;
intervals = intervals(intervals(:, 2) > windowStart, :);
distractionState.awayIntervals = intervals;

cumulativeAway = 0;
for idx = 1:size(intervals, 1)
    overlapStart = max(intervals(idx, 1), windowStart);
    overlapEnd = min(intervals(idx, 2), timestamp);
    cumulativeAway = cumulativeAway + max(0, overlapEnd - overlapStart);
end

if ~isnan(distractionState.currentAwayStart)
    cumulativeAway = cumulativeAway + max(0, timestamp - max(distractionState.currentAwayStart, windowStart));
end

if cumulativeAway >= config.shortAccumSec
    distractionState.shortActive = true;
end

if distractionState.shortActive && ~isAway && distractionState.returnDuration >= config.returnToRoadSec
    distractionState.shortActive = false;
    distractionState.awayIntervals = zeros(0, 2);
end

