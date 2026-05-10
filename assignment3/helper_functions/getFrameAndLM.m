% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% capture a video frame and detect facial landmarks using the Python bridge
function [frame, h, w, lm] = getFrameAndLM(bridge)
    lm = [];  frame = [];  h = 0;  w = 0;
    persistent lastWarnTic

    try
        py_result = cell(bridge.capture_frame());
        py_bytes  = py_result{1};
        h = double(py_result{2});
        w = double(py_result{3});
        if h == 0 || w == 0; return; end

        vec = uint8(py.array.array('B', py_bytes));
        frame = permute(reshape(vec, [3, w, h]), [3, 2, 1]);

        % Landmarks
        py_lm = bridge.detect_landmarks(py_bytes, int32(h), int32(w));
        coords = double(py.array.array('d', py_lm));
        if isempty(coords); return; end
        lm = reshape(coords, 3, [])';   % 478 x 3  [x y z]
    catch ME
        if isempty(lastWarnTic) || toc(lastWarnTic) > 5
            warning('getFrameAndLM:BridgeError', ...
                'MediaPipe bridge error: %s', ME.message);
            lastWarnTic = tic;
        end
        lm = [];
    end
end
