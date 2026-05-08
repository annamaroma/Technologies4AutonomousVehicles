% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% compute mean RGB values in the face region defined by landmarks
function mean_rgb = getFaceMeanRGB(frame, lm)
    mean_rgb = [NaN NaN NaN];
    if isempty(frame) || isempty(lm) || size(lm,1) < 455
        return;
    end

    [h, w, ~] = size(frame);
    faceOval = [10 338 297 332 284 251 389 356 454 323 361 288 397 365 ...
                379 378 400 377 152 148 176 149 150 136 172 58 132 93 ...
                234 127 162 21 54 103 67 109] + 1;
    xy = lm(faceOval, 1:2) .* [w h];

    x1 = max(1, floor(min(xy(:,1))));
    x2 = min(w, ceil(max(xy(:,1))));
    y1 = max(1, floor(min(xy(:,2))));
    y2 = min(h, ceil(max(xy(:,2))));

    if x2 <= x1 || y2 <= y1
        return;
    end

    roiW = x2 - x1 + 1;
    roiH = y2 - y1 + 1;
    x1 = max(1, round(x1 + 0.18 * roiW));
    x2 = min(w, round(x2 - 0.18 * roiW));
    y1 = max(1, round(y1 + 0.18 * roiH));
    y2 = min(h, round(y2 - 0.30 * roiH));

    if x2 <= x1 || y2 <= y1
        return;
    end

    roi = double(frame(y1:y2, x1:x2, :));
    mean_rgb = squeeze(mean(mean(roi, 1), 2))';
end

