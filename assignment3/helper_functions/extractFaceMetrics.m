% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% compute subject-independent geometric features from face landmarks
function [m, isValid] = extractFaceMetrics(lm, RE_IDX, LE_IDX, NOSE_IDX, ...
    R_IRIS_IDX, L_IRIS_IDX, R_EYE_OUT, R_EYE_IN, L_EYE_OUT, L_EYE_IN)

    m = struct( ...
        'ear_r', NaN, 'ear_l', NaN, 'ear', NaN, ...
        'nose_rel_x', NaN, 'nose_rel_y', NaN, ...
        'r_iris_h', NaN, 'l_iris_h', NaN, 'r_iris_v', NaN, 'l_iris_v', NaN, ...
        'face_area', NaN);
    isValid = false;

    req = unique([RE_IDX(:); LE_IDX(:); NOSE_IDX; R_IRIS_IDX; L_IRIS_IDX; ...
                  R_EYE_OUT; R_EYE_IN; L_EYE_OUT; L_EYE_IN]);
    if isempty(lm) || size(lm,1) < max(req)
        return;
    end

    pts = lm(req, 1:2);
    if any(~isfinite(pts(:))) || any(pts(:) < -0.05) || any(pts(:) > 1.05)
        return;
    end

    re = lm(RE_IDX, 1:2);
    le = lm(LE_IDX, 1:2);
    m.ear_r = computeEAR(re);
    m.ear_l = computeEAR(le);
    m.ear = mean([m.ear_r, m.ear_l], 'omitnan');
    if ~isfinite(m.ear)
        return;
    end

    r_eye_outer = lm(R_EYE_OUT, 1:2);
    r_eye_inner = lm(R_EYE_IN, 1:2);
    l_eye_outer = lm(L_EYE_OUT, 1:2);
    l_eye_inner = lm(L_EYE_IN, 1:2);
    r_eye_center = (r_eye_outer + r_eye_inner) / 2;
    l_eye_center = (l_eye_outer + l_eye_inner) / 2;
    eye_mid = (r_eye_center + l_eye_center) / 2;
    interocular = norm(l_eye_center - r_eye_center);
    if interocular < 0.06
        return;
    end

    nose_xy = lm(NOSE_IDX, 1:2);
    m.nose_rel_x = (nose_xy(1) - eye_mid(1)) / interocular;
    m.nose_rel_y = (nose_xy(2) - eye_mid(2)) / interocular;

    r_eye_width = max(norm(r_eye_inner - r_eye_outer), eps);
    l_eye_width = max(norm(l_eye_inner - l_eye_outer), eps);
    r_eye_upper = mean(lm(RE_IDX([2 3]), 1:2), 1);
    r_eye_lower = mean(lm(RE_IDX([5 6]), 1:2), 1);
    l_eye_upper = mean(lm(LE_IDX([2 3]), 1:2), 1);
    l_eye_lower = mean(lm(LE_IDX([5 6]), 1:2), 1);
    r_eye_center_y = mean([r_eye_upper(2), r_eye_lower(2)]);
    l_eye_center_y = mean([l_eye_upper(2), l_eye_lower(2)]);
    r_eye_height = max(norm(r_eye_upper - r_eye_lower), 0.008);
    l_eye_height = max(norm(l_eye_upper - l_eye_lower), 0.008);

    r_iris_xy = lm(R_IRIS_IDX, 1:2);
    l_iris_xy = lm(L_IRIS_IDX, 1:2);
    m.r_iris_h = (r_iris_xy(1) - r_eye_center(1)) / r_eye_width;
    m.l_iris_h = (l_iris_xy(1) - l_eye_center(1)) / l_eye_width;
    m.r_iris_v = (r_iris_xy(2) - r_eye_center_y) / r_eye_height;
    m.l_iris_v = (l_iris_xy(2) - l_eye_center_y) / l_eye_height;

    r_eye_x = sort([r_eye_outer(1), r_eye_inner(1)]);
    l_eye_x = sort([l_eye_outer(1), l_eye_inner(1)]);
    r_iris_ratio_x = (r_iris_xy(1) - r_eye_x(1)) / max(r_eye_x(2) - r_eye_x(1), eps);
    l_iris_ratio_x = (l_iris_xy(1) - l_eye_x(1)) / max(l_eye_x(2) - l_eye_x(1), eps);
    r_iris_ratio_y = (r_iris_xy(2) - r_eye_upper(2)) / max(r_eye_lower(2) - r_eye_upper(2), 0.008);
    l_iris_ratio_y = (l_iris_xy(2) - l_eye_upper(2)) / max(l_eye_lower(2) - l_eye_upper(2), 0.008);

    faceOval = [10 338 297 332 284 251 389 356 454 323 361 288 397 365 ...
                379 378 400 377 152 148 176 149 150 136 172 58 132 93 ...
                234 127 162 21 54 103 67 109] + 1;
    facePts = lm(faceOval, 1:2);
    faceSize = max(facePts, [], 1) - min(facePts, [], 1);
    m.face_area = faceSize(1) * faceSize(2);

    ratioChecks = [r_iris_ratio_x, l_iris_ratio_x];
    if m.ear > 0.20
        ratioChecks = [ratioChecks, r_iris_ratio_y, l_iris_ratio_y];
    end
    if any(~isfinite(ratioChecks)) || any(ratioChecks < -0.45) || any(ratioChecks > 1.45)
        return;
    end
    if ~isfinite(m.face_area) || m.face_area < 0.03
        return;
    end

    isValid = true;
end
