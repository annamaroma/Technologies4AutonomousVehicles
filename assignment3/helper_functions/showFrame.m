% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% display frame with landmarks (if provided)
function hImg = showFrame(hImg, frame, hAx)
    if nargin < 3; hAx = gca; end
    if isempty(hImg)
        hImg = imshow(frame, 'Parent', hAx);
        set(hAx, 'Position', [0 0 1 1]);
    else
        set(hImg, 'CData', frame);
    end
end
