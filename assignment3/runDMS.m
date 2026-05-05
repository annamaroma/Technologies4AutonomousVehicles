clc; close all;

% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System


addpath("libraries");
%% Paths
scriptDir  = fileparts(mfilename('fullpath'));
libDir   = fullfile(scriptDir, 'libraries', 'remote_PPG');
addpath(scriptDir);
addpath(libDir);
addpath(fullfile(scriptDir, 'libraries', 'ICA'));
addpath("libraries");

%% Parameters
CALIB_SEC        = 3;     % seconds to look forward for baseline calibration
NOSE_THRESH      = 0.060; % normalised deviation (x or y) triggering head-away detection
IRIS_THRESH      = 0.005; % normalised iris offset for lizard distraction
EAR_THRESH       = 0.300; % Eye Aspect Ratio below this = eyes closed (calibrated live)
LONG_OWL_SEC     = 5.0;   % continuous head-away time for long owl alarm
SHORT_OWL_WIN    = 30.0;  % sliding-window length for short owl [s]
SHORT_OWL_CUM    = 10.0;  % cumulative away time inside window [s]
SHORT_OWL_RESET  = 2.0;   % focused time needed to clear short owl alarm [s]
MICROSLEEP_SEC   = 4.0;   % eye-closure threshold for microsleep
SLEEP_SEC        = 7.0;   % eye-closure threshold for sleep
EYE_RESET_SEC    = 2.0;   % eye-open time to clear microsleep/sleep alarm
RPPG_BUF_SEC     = 20;    % seconds of RGB history for rPPG
RPPG_UPDATE_SEC  = 5;     % update heart rate every N seconds
BPM_RANGE        = [45 165];
RECORD           = true;  % set false to skip MP4 saving (live preview only)
OUT_FILE         = fullfile(scriptDir, 'results', 'DMS_output.mp4');

%% Landmark index maps  
% Right eye EAR: p1 outer, p2 up-out, p3 up-in, p4 inner, p5 lo-in, p6 lo-out
RE_IDX = [33 160 158 133 153 144] + 1;
% Left eye EAR
LE_IDX = [263 387 385 362 373 380] + 1;
% Nose tip (used for head direction)
NOSE_IDX  = 4 + 1;       % landmark 4
% Right iris centre, left iris centre
R_IRIS_IDX = 468 + 1;
L_IRIS_IDX = 473 + 1;
% Eye corners for iris offset reference
R_EYE_OUT  = 33  + 1;  R_EYE_IN = 133 + 1;
L_EYE_OUT  = 263 + 1;  L_EYE_IN = 362 + 1;

%% Initialise Python bridge 
fprintf('Initialising Python MediaPipe bridge...\n');
try
    bridge = py.importlib.import_module('mediapipe_bridge');
    ok = bridge.initialize(0);
    if ~ok; error('Cannot open webcam in Python.'); end
catch ME
    error(['Python bridge failed: ' ME.message ...
           '\nCheck pyenv and that mediapipe/cv2 are installed.']);
end

%% Video writer
if RECORD
    if isfile(OUT_FILE)
        try
            delete(OUT_FILE);
        catch
            [p, n, e] = fileparts(OUT_FILE);
            OUT_FILE = fullfile(p, sprintf('%s_%s%s', n, char(datetime('now','Format','yyyyMMdd_HHmmss')), e));
            warning('DMS_output.mp4 locked, writing to %s instead.', OUT_FILE);
        end
    end
    vw = VideoWriter(OUT_FILE, 'MPEG-4');
    vw.FrameRate = 20;
    open(vw);
end

%% Display figure 
hFig = figure('Name','DMS - press any key to stop','NumberTitle','off', ...
              'KeyPressFcn', @(~,~) setappdata(gcf,'stop',true));
setappdata(hFig,'stop',false);
% Image axes
hAx = axes('Parent', hFig, 'Position', [0 0 1 1], ...
           'Visible', 'off', 'XTick', [], 'YTick', []);
hAxOv = axes('Parent', hFig, 'Position', [0 0 1 1], ...
             'Color', 'none', 'XLim', [0 1], 'YLim', [0 1], ...
             'Visible', 'off', 'HitTest', 'off', 'PickableParts', 'none');
uistack(hAxOv, 'top');
hNoFaceTxt = text(0.5, 0.5, 'No face detected', 'Parent', hAxOv, ...
    'Units', 'data', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 20, 'FontWeight', 'bold', ...
    'Color', 'white', 'BackgroundColor', [0.78 0 0], 'Visible', 'off');
hImg = [];

%% Calibration phase 
fprintf('Calibration: look forward for %.0f seconds...\n', CALIB_SEC);
nose_cx  = 0.5;   % default centre (normalised x)
nose_cy  = 0.5;   % default centre (normalised y)
calib_xs = zeros(1, ceil(CALIB_SEC * 60));   % preallocate (60 fps upper bound)
calib_ys = zeros(1, ceil(CALIB_SEC * 60));
n_calib  = 0;
t_calib  = tic;

while toc(t_calib) < CALIB_SEC
    [frame, h, w, lm] = getFrameAndLM(bridge);
    if ~isempty(lm)
        n_calib = n_calib + 1;
        calib_xs(n_calib) = lm(NOSE_IDX, 1);
        calib_ys(n_calib) = lm(NOSE_IDX, 2);
        frame = insertText(frame, [w/2, 30], ...
            sprintf('Calibrating... %.1f s', CALIB_SEC-toc(t_calib)), ...
            'FontSize',16,'BoxColor',[0 0 0],'BoxOpacity',0.6,'TextColor','white', ...
            'AnchorPoint','CenterTop');
    end
    hImg = showFrame(hImg, frame, hAx);
    drawnow limitrate;
end
calib_xs = calib_xs(1:n_calib);   % trim unused slots
calib_ys = calib_ys(1:n_calib);
if ~isempty(calib_xs)
    nose_cx = mean(calib_xs);
    nose_cy = mean(calib_ys);
end
fprintf('Baseline nose x = %.3f, y = %.3f\n', nose_cx, nose_cy);

%% State variables

% Owl
head_away_start = NaN;    
long_alarm      = false;
short_log       = zeros(0,2);
short_alarm     = false;
short_focus_start = NaN;  

% Lizard 
lizard_away_start = NaN;
lizard_alarm_long = false;
lizard_short_log  = zeros(0,2);
lizard_alarm_short= false;
lizard_focus_start= NaN;

% Eyes 
eye_close_start = NaN;    
eye_open_start  = NaN;     
eye_alarm       = 0;       

% rPPG 
rgb_buf  = zeros(0,3);
t_buf    = zeros(0,1);
hr_bpm   = NaN;
t_last_hr = -Inf;

state   = 'Focused on the road';
t_start = tic;

%% Main Loop
fprintf('DMS running. Press any key in the figure window to stop.\n');

while ~getappdata(hFig,'stop')
    t_now = toc(t_start);

    %% Get frame and landmarks
    [frame, h, w, lm] = getFrameAndLM(bridge);
    if isempty(frame); continue; end

    %% Face not detected
    if isempty(lm)
        out = insertText(frame, [w/2, h/2], 'No face detected', ...
            'FontSize', 18, 'BoxColor', [200 0 0], 'BoxOpacity', 0.6, ...
            'TextColor', 'white', 'AnchorPoint', 'CenterCenter');
        if RECORD; writeVideo(vw, out); end
        hImg = showFrame(hImg, frame, hAx);
        set(hNoFaceTxt, 'Visible', 'on');
        drawnow limitrate;
        continue;
    end
    set(hNoFaceTxt, 'Visible', 'off');   

    %% Accumulate rPPG signal 
    mean_rgb = getFaceMeanRGB(frame, lm);
    if ~any(isnan(mean_rgb))
        rgb_buf(end+1,:) = mean_rgb;
        t_buf(end+1)     = t_now;
        % Keep only last RPPG_BUF_SEC
        keep = t_buf >= t_now - RPPG_BUF_SEC;
        rgb_buf = rgb_buf(keep,:);
        t_buf   = t_buf(keep);
    end

    %% Update heart rate 
    if (t_now - t_last_hr) >= RPPG_UPDATE_SEC && size(rgb_buf,1) >= 60
        hr_bpm    = estimateHR(rgb_buf, t_buf, BPM_RANGE);
        t_last_hr = t_now;
    end

    %% Eye Aspect Ratio 
    re = lm(RE_IDX, 1:2) .* [w h];
    le = lm(LE_IDX, 1:2) .* [w h];
    ear_r = computeEAR(re);
    ear_l = computeEAR(le);
    ear   = (ear_r + ear_l) / 2;
    eyes_closed = ear < EAR_THRESH;

    % Eye alarm state machine
    if eyes_closed
        if isnan(eye_close_start); eye_close_start = t_now; end
        eye_open_start = NaN;
        dur = t_now - eye_close_start;
        if dur >= SLEEP_SEC
            eye_alarm = 2;
        elseif dur >= MICROSLEEP_SEC
            eye_alarm = 1;
        end
    else
        eye_close_start = NaN;
        if eye_alarm > 0
            if isnan(eye_open_start); eye_open_start = t_now; end
            if (t_now - eye_open_start) >= EYE_RESET_SEC
                eye_alarm      = 0;
                eye_open_start = NaN;
            end
        else
            eye_open_start = NaN;
        end
    end

    %% Nose-tip gaze (Owl distraction)
    nose_x  = lm(NOSE_IDX, 1);
    nose_y  = lm(NOSE_IDX, 2);
    nose_dx = nose_x - nose_cx;
    nose_dy = nose_y - nose_cy;
    % Suppress owl detection while eye-closure counter is running: priority
    % is on microsleep/sleep; don't accumulate distraction during closure.
    head_away = ~eyes_closed && ...
        (abs(nose_dx) > NOSE_THRESH || abs(nose_dy) > NOSE_THRESH);

    % Long owl
    if head_away
        if isnan(head_away_start); head_away_start = t_now; end
        if (t_now - head_away_start) >= LONG_OWL_SEC
            long_alarm = true;
        end
    else
        head_away_start = NaN;
        long_alarm = false;   
    end

    % Short owl (sliding window)
    short_log(end+1,:) = [t_now, double(head_away)];
    short_log = short_log(short_log(:,1) >= t_now - SHORT_OWL_WIN, :);
    cumul_away = 0;
    if size(short_log,1) >= 2
        dt_s = diff(short_log(:,1));
        cumul_away = sum(dt_s .* short_log(1:end-1,2));
    end
    if short_alarm
        if ~head_away
            if isnan(short_focus_start); short_focus_start = t_now; end
            if (t_now - short_focus_start) >= SHORT_OWL_RESET
                short_alarm = false;
                short_focus_start = NaN;
                short_log = zeros(0,2);  % fresh window after recovery
            end
        else
            short_focus_start = NaN;
        end
    else
        if cumul_away >= SHORT_OWL_CUM
            short_alarm       = true;
            short_focus_start = NaN;
        end
    end

    %% Iris gaze in Lizard distraction
    r_iris_x = lm(R_IRIS_IDX, 1);
    l_iris_x = lm(L_IRIS_IDX, 1);
    r_iris_y = lm(R_IRIS_IDX, 2);
    l_iris_y = lm(L_IRIS_IDX, 2);
    r_eye_cx = (lm(R_EYE_OUT,1) + lm(R_EYE_IN,1)) / 2;
    l_eye_cx = (lm(L_EYE_OUT,1) + lm(L_EYE_IN,1)) / 2;
    % Vertical eye centre from EAR up/down landmarks (normalised y)
    r_eye_cy = (lm(RE_IDX(2),2) + lm(RE_IDX(3),2) + lm(RE_IDX(5),2) + lm(RE_IDX(6),2)) / 4;
    l_eye_cy = (lm(LE_IDX(2),2) + lm(LE_IDX(3),2) + lm(LE_IDX(5),2) + lm(LE_IDX(6),2)) / 4;
    r_offset   = r_iris_x - r_eye_cx;
    l_offset   = l_iris_x - l_eye_cx;
    r_offset_y = r_iris_y - r_eye_cy;
    l_offset_y = l_iris_y - l_eye_cy;

    lizard_away = ~head_away && ~eyes_closed && ...
        (abs(r_offset) > IRIS_THRESH || abs(l_offset) > IRIS_THRESH || ...
         abs(r_offset_y) > IRIS_THRESH || abs(l_offset_y) > IRIS_THRESH);

    % Lizard long
    if lizard_away
        if isnan(lizard_away_start); lizard_away_start = t_now; end
        if (t_now - lizard_away_start) >= LONG_OWL_SEC
            lizard_alarm_long = true;
        end
    else
        lizard_away_start = NaN;
        lizard_alarm_long = false;
    end

    % Lizard short
    lizard_short_log(end+1,:) = [t_now, double(lizard_away)];
    lizard_short_log = lizard_short_log(lizard_short_log(:,1) >= t_now - SHORT_OWL_WIN, :);
    cumul_liz = 0;
    if size(lizard_short_log,1) >= 2
        dt_l    = diff(lizard_short_log(:,1));
        cumul_liz = sum(dt_l .* lizard_short_log(1:end-1,2));
    end
    if lizard_alarm_short
        if ~lizard_away
            if isnan(lizard_focus_start); lizard_focus_start = t_now; end
            if (t_now - lizard_focus_start) >= SHORT_OWL_RESET
                lizard_alarm_short = false;
                lizard_focus_start = NaN;
                lizard_short_log   = zeros(0,2);  % flush to prevent immediate re-trigger
            end
        else
            lizard_focus_start = NaN;
        end
    else
        if cumul_liz >= SHORT_OWL_CUM
            lizard_alarm_short = true;
            lizard_focus_start = NaN;
        end
    end

    %% Overall driver state (Sleep > Microsleep > Long > Short > Focused) 
    if eye_alarm == 2
        state = 'Sleep';
    elseif eye_alarm == 1
        state = 'Microsleep';
    elseif long_alarm || lizard_alarm_long
        state = 'Distracted (long)';
    elseif short_alarm || lizard_alarm_short
        state = 'Distracted (short)';
    else
        state = 'Focused on the road';
    end

    %%  Compose output frame 
    out = drawLandmarks(frame, lm);   % eye contours (red), iris (green), nose (blue)
    out = insertText(out, [w/2, 5], ...
        sprintf(['EAR %.3f | irisR %+.3f/%+.3f  irisL %+.3f/%+.3f | ' ...
                 'nose %.3f/%.3f (d %+.3f/%+.3f)'], ...
                ear, r_offset, r_offset_y, l_offset, l_offset_y, ...
                nose_x, nose_y, nose_dx, nose_dy), ...
        'FontSize', 12, 'BoxColor', [0 0 0], 'BoxOpacity', 0.5, ...
        'TextColor', 'white', 'AnchorPoint', 'CenterTop');
    out = addOverlay(out, state, hr_bpm);
    if RECORD; writeVideo(vw, out); end
    hImg = showFrame(hImg, out, hAx);
    drawnow limitrate;
end

%% Cleanup 
if RECORD
    close(vw);
    fprintf('Stopped. Video saved to: %s\n', OUT_FILE);
else
    fprintf('Stopped. Recording disabled (RECORD=false).\n'); 
end
bridge.release();
try close(hFig); catch; end

%% helper functions

function [frame, h, w, lm] = getFrameAndLM(bridge)

lm = [];  frame = [];  h = 0;  w = 0;

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
catch

end
end

function hImg = showFrame(hImg, frame, hAx)
if nargin < 3; hAx = gca; end
if isempty(hImg)
    hImg = imshow(frame, 'Parent', hAx);
    set(hAx, 'Position', [0 0 1 1]);
else
    set(hImg, 'CData', frame);
end
end
