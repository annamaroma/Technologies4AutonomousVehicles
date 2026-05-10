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
    leftEye = [362 382 381 380 374 373 390 249 263 466 388 387 386 385 384 398] + 1;
    rightEye = [33 7 163 144 145 153 154 155 133 173 157 158 159 160 161 246] + 1;
    mouth = [61 146 91 181 84 17 314 405 321 375 291 308 324 318 402 317 ...
             14 87 178 88 95 185 40 39 37 0 267 269 270 409 415 310 311 ...
             312 13 82 81 42 183 78] + 1;
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

    faceMask = poly2mask(xy(:,1), xy(:,2), h, w);
    leftEyeXY = lm(leftEye, 1:2) .* [w h];
    rightEyeXY = lm(rightEye, 1:2) .* [w h];
    mouthXY = lm(mouth, 1:2) .* [w h];
    faceMask = faceMask & ~poly2mask(leftEyeXY(:,1), leftEyeXY(:,2), h, w);
    faceMask = faceMask & ~poly2mask(rightEyeXY(:,1), rightEyeXY(:,2), h, w);
    faceMask = faceMask & ~poly2mask(mouthXY(:,1), mouthXY(:,2), h, w);

    roiMask = false(h, w);
    roiMask(y1:y2, x1:x2) = true;
    skinMask = faceMask & roiMask;
    if ~any(skinMask(:))
        return;
    end

    rgb = double(frame);
    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);
    skinRule = R > 60 & G > 35 & B > 20 & (max(max(R, G), B) - min(min(R, G), B)) > 12 & R > G & R > B;
    skinMask = skinMask & skinRule;
    if nnz(skinMask) < 200
        skinMask = faceMask & roiMask;
    end
    if nnz(skinMask) < 200
        return;
    end

    mean_rgb = [mean(R(skinMask)), mean(G(skinMask)), mean(B(skinMask))];
end
