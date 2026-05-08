% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% compute eye aspect ratio (EAR) for a given eye
function ear = computeEAR(eyePts)
    ear = NaN;
    if size(eyePts,1) ~= 6
        return;
    end

    d14 = norm(eyePts(1,:) - eyePts(4,:));
    if d14 <= eps
        return;
    end

    d26 = norm(eyePts(2,:) - eyePts(6,:));
    d35 = norm(eyePts(3,:) - eyePts(5,:));
    ear = (d26 + d35) / (2 * d14);
end