% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% visualization functions
function out = drawLandmarks(frame, lm)
    out = frame;
    if isempty(frame) || isempty(lm) || size(lm,1) < 478
        return;
    end

    [h, w, ~] = size(frame);
    leftEye  = [362 382 381 380 374 373 390 249 263 466 388 387 386 385 384 398] + 1;
    rightEye = [33 7 163 144 145 153 154 155 133 173 157 158 159 160 161 246] + 1;
    leftIris = [473 474 475 476 477] + 1;
    rightIris= [468 469 470 471 472] + 1;
    noseTip  = [45 4 275] + 1;
    faceOval = [10 338 297 332 284 251 389 356 454 323 361 288 397 365 ...
                379 378 400 377 152 148 176 149 150 136 172 58 132 93 ...
                234 127 162 21 54 103 67 109] + 1;
    facePts = lm(faceOval, 1:2) .* [w h];
    minXY = min(facePts, [], 1);
    maxXY = max(facePts, [], 1);
    rect = [minXY(1), minXY(2), maxXY(1)-minXY(1), maxXY(2)-minXY(2)];
    leftEyeCircles  = [lm(leftEye, 1:2) .* [w h], repmat(2, numel(leftEye), 1)];
    rightEyeCircles = [lm(rightEye,1:2) .* [w h], repmat(2, numel(rightEye),1)];
    leftIrisCircles = [lm(leftIris,1:2) .* [w h], repmat(2, numel(leftIris),1)];
    rightIrisCircles= [lm(rightIris,1:2).* [w h], repmat(2, numel(rightIris),1)];
    noseCircles     = [lm(noseTip, 1:2) .* [w h], repmat(2, numel(noseTip), 1)];
    out = insertShape(out, 'Rectangle', rect, 'Color', [0 148 192], 'LineWidth', 2);
    out = insertShape(out, 'FilledCircle', leftEyeCircles,   'Color', [180 21 0],   'Opacity', 1);
    out = insertShape(out, 'FilledCircle', rightEyeCircles,  'Color', [180 21 0],   'Opacity', 1);
    out = insertShape(out, 'FilledCircle', leftIrisCircles,  'Color', [46 139 87], 'Opacity', 1);
    out = insertShape(out, 'FilledCircle', rightIrisCircles, 'Color', [46 139 87], 'Opacity', 1);
    out = insertShape(out, 'FilledCircle', noseCircles,      'Color', [25 25 112],  'Opacity', 1);


    % [0 148 192] azzurrino 
    % [46 139 87] verde 
    % [25 25 112] blu
    % [180 21 0] rosso
end
