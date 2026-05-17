clc; close all; clear all;

% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% Paths
scriptDir  = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(fullfile(scriptDir, 'libraries'));
addpath(fullfile(scriptDir, 'helper_functions'));

%% Parameters
CALIB_SEC        = 3;     % target seconds of valid forward-looking calibration data
CALIB_MAX_SEC    = 20;    % maximum wall-clock time allowed for calibration
CALIB_MIN_FRAMES = 40;    % minimum valid frames required to lock the baselines
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
EAR_THRESH_RANGE = [0.18 0.34];
NOSE_THRESH_MIN  = [0.10 0.12]; % [x y] minimum threshold on nose-vs-eyes relative pose
IRIS_THRESH_MIN  = [0.12 0.18]; % [x y] minimum threshold on normalized iris motion
RECORD           = true;  % set false to skip MP4 saving (live preview only)
GRID_COLOR       = [0 0 0];
CENTER_BOX_OK    = [154 205 50]; % verde oliva chiaro
CENTER_BOX_WARN  = [180 21 0]; % rosso
FACE_GUIDE_MSG   = 'Keep your face in the centre';
timestamp        = char(datetime('now','Format','yyyyMMdd_HHmmss'));
OUT_FILE         = fullfile(scriptDir, 'results', sprintf('DMS_output_%s.mp4', timestamp));

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

pythonExe = resolvePythonExecutable(scriptDir);
pe = pyenv;
if strcmp(string(pe.Status), "NotLoaded")
    if strlength(pythonExe) == 0
        error('runDMS:PythonNotFound', ...
            ['No usable Python interpreter found. Set DMS_PYTHON or install a Python ' ...
             'environment with mediapipe and opencv-python.']);
    end
    pyenv("Version", char(pythonExe));
    pe = pyenv;
elseif strlength(pythonExe) > 0 && ~strcmp(string(pe.Version), string(pythonExe))
    fprintf(['MATLAB Python already loaded from %s; continuing with that interpreter ' ...
             '(requested: %s).\n'], pe.Version, pythonExe);
end

py.importlib.invalidate_caches();
sysPathObj = py.sys.path();
sysPath = cell(sysPathObj);
sysPath = cellfun(@char, sysPath, 'UniformOutput', false);
if ~any(strcmp(sysPath, scriptDir))
    sysPathObj.insert(int32(0), scriptDir);
end

bridge = py.importlib.import_module('mediapipe_bridge');
bridge = py.importlib.reload(bridge);


ok = logical(bridge.initialize(int32(0)));
if ~ok
    error('runDMS:PythonInitFailed', ...
          'Failed to initialize mediapipe_bridge. Check webcam access and Python dependencies.');
end






%% Video writer
fprintf('Initialising video writer...\n');
if RECORD
    try
        vw = VideoWriter(OUT_FILE, 'MPEG-4');
    catch ME
        if contains(ME.message, 'profile is not valid', 'IgnoreCase', true)
            [p, n, ~] = fileparts(OUT_FILE);
            OUT_FILE = fullfile(p, sprintf('%s.avi', n));
            if isfile(OUT_FILE)
                try
                    delete(OUT_FILE);
                catch
                    OUT_FILE = fullfile(p, sprintf('%s_%s.avi', n, char(datetime('now','Format','yyyyMMdd_HHmmss'))));
                end
            end
            fprintf(['MPEG-4 not available in this MATLAB installation. ' ...
                     'Falling back to Motion JPEG AVI: %s\n'], OUT_FILE);
            vw = VideoWriter(OUT_FILE, 'Motion JPEG AVI');
        else
            rethrow(ME);
        end
    end
    vw.FrameRate = 20;
    open(vw);
end

%% Display figure 
fprintf('Initialising display...\n');
hFig = figure('Name','DMS - press C to recalibrate, S to stop','NumberTitle','off', ...
              'KeyPressFcn', @handleKeyPress);
setappdata(hFig,'stop',false);
setappdata(hFig,'recalibrate',false);
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

need_recalibration = true;
t_start = tic;

while ~getappdata(hFig,'stop')
    if need_recalibration
        fprintf('Calibration: look forward for %.0f seconds of valid face data...\n', CALIB_SEC);

        calibCap = max(CALIB_MIN_FRAMES + 20, ceil(CALIB_MAX_SEC * 25));
        calib_nose_x = zeros(calibCap, 1);
        calib_nose_y = zeros(calibCap, 1);
        calib_ears   = zeros(calibCap, 1);
        calib_rh     = zeros(calibCap, 1);
        calib_lh     = zeros(calibCap, 1);
        calib_rv     = zeros(calibCap, 1);
        calib_lv     = zeros(calibCap, 1);
        n_calib = 0;
        t_calib = tic;
        setappdata(hFig, 'recalibrate', false);

        while n_calib < CALIB_MIN_FRAMES && toc(t_calib) < CALIB_MAX_SEC
            if getappdata(hFig,'stop')
                break;
            end
            if getappdata(hFig,'recalibrate')
                setappdata(hFig, 'recalibrate', false);
                n_calib = 0;
                t_calib = tic;
            end

            [frame, h, w, lm] = getFrameAndLM(bridge);
            if isempty(frame)
                drawnow limitrate;
                continue;
            end

            [metrics, face_valid] = extractFaceMetrics(lm, RE_IDX, LE_IDX, NOSE_IDX, ...
                R_IRIS_IDX, L_IRIS_IDX, R_EYE_OUT, R_EYE_IN, L_EYE_OUT, L_EYE_IN);
            face_in_center = false;
            if face_valid
                face_in_center = isFaceInsideCenter(lm, w, h);
            end

            if face_valid && face_in_center
                n_calib = n_calib + 1;
                calib_nose_x(n_calib) = metrics.nose_rel_x;
                calib_nose_y(n_calib) = metrics.nose_rel_y;
                calib_ears(n_calib)   = metrics.ear;
                calib_rh(n_calib)     = metrics.r_iris_h;
                calib_lh(n_calib)     = metrics.l_iris_h;
                calib_rv(n_calib)     = metrics.r_iris_v;
                calib_lv(n_calib)     = metrics.l_iris_v;
                frame = drawLandmarks(frame, lm);
                msg = sprintf('Calibrating... %d/%d valid frames', n_calib, CALIB_MIN_FRAMES);
                boxColor = [0 120 0];
            elseif face_valid
                frame = drawLandmarks(frame, lm);
                msg = sprintf('Calibration: keep face in the centre (%d/%d)', n_calib, CALIB_MIN_FRAMES);
                boxColor = [180 80 0];
            else
                msg = sprintf('Calibration: keep both eyes and nose visible (%d/%d)', n_calib, CALIB_MIN_FRAMES);
                boxColor = [180 80 0];
            end

            frame = drawAttentionGrid(frame, w, h, face_in_center, GRID_COLOR, CENTER_BOX_OK, CENTER_BOX_WARN);
            frame = insertText(frame, [w/2, 30], msg, ...
                'FontSize', 16, 'BoxColor', boxColor, 'BoxOpacity', 0.65, 'TextColor', 'white', ...
                'AnchorPoint', 'CenterTop');
            hImg = showFrame(hImg, frame, hAx);
            drawnow limitrate;
        end

        if getappdata(hFig,'stop')
            break;
        end

        if n_calib < CALIB_MIN_FRAMES
            warning('runDMS:CalibrationRetry', ...
                'Calibration incomplete. Press C to retry or keep your face centered and wait.');
            continue;
        end

        calib_nose_x = calib_nose_x(1:n_calib);
        calib_nose_y = calib_nose_y(1:n_calib);
        calib_ears   = calib_ears(1:n_calib);
        calib_rh     = calib_rh(1:n_calib);
        calib_lh     = calib_lh(1:n_calib);
        calib_rv     = calib_rv(1:n_calib);
        calib_lv     = calib_lv(1:n_calib);

        nose_cx = median(calib_nose_x);
        nose_cy = median(calib_nose_y);
        ear_open = median(calib_ears);
        EAR_THRESH = min(EAR_THRESH_RANGE(2), max(EAR_THRESH_RANGE(1), ...
            ear_open - max([0.04, 3 * robustStd(calib_ears), 0.22 * ear_open])));
        NOSE_THRESH = max(NOSE_THRESH_MIN, ...
            [4 * robustStd(calib_nose_x), 4 * robustStd(calib_nose_y)]);
        r_iris_base = [median(calib_rh), median(calib_rv)];
        l_iris_base = [median(calib_lh), median(calib_lv)];
        IRIS_THRESH = max(IRIS_THRESH_MIN, ...
            [4 * max(robustStd(calib_rh), robustStd(calib_lh)), ...
             4 * max(robustStd(calib_rv), robustStd(calib_lv))]);

        fprintf(['Calibration locked: nose=[%.3f %.3f], EARopen=%.3f, EARth=%.3f, ' ...
                 'noseTh=[%.3f %.3f], irisTh=[%.3f %.3f]\n'], ...
                nose_cx, nose_cy, ear_open, EAR_THRESH, ...
                NOSE_THRESH(1), NOSE_THRESH(2), IRIS_THRESH(1), IRIS_THRESH(2));

        fprintf('Entering main loop...\n');
        % Owl
        head_away_start = NaN;    
        long_alarm      = false;
        owl_short_segments = zeros(0,2);
        short_alarm     = false;
        owl_focus_start = NaN;  
        owl_short_alarm_since = NaN;

        % Lizard 
        lizard_away_start = NaN;
        lizard_alarm_long = false;
        lizard_short_segments = zeros(0,2);
        lizard_alarm_short= false;
        lizard_focus_start= NaN;
        lizard_short_alarm_since = NaN;

        % Eyes 
        eye_close_start = NaN;    
        eye_open_start  = NaN;     
        eye_alarm       = 0;       

        % rPPG 
        rgb_buf  = zeros(0,3);
        t_buf    = zeros(0,1);
        hr_bpm   = NaN;
        t_last_hr = -Inf;
        hr_hist  = zeros(0,1);

        state   = 'Focused on the road';
        state_alarm_sec = 0;
        t_start = tic;
        need_recalibration = false;
    end

    t_now = toc(t_start);

    if getappdata(hFig,'recalibrate')
        setappdata(hFig, 'recalibrate', false);
        need_recalibration = true;
        continue;
    end

    %% Get frame and landmarks
    [frame, h, w, lm] = getFrameAndLM(bridge);
    if isempty(frame); continue; end
    [metrics, face_valid] = extractFaceMetrics(lm, RE_IDX, LE_IDX, NOSE_IDX, ...
        R_IRIS_IDX, L_IRIS_IDX, R_EYE_OUT, R_EYE_IN, L_EYE_OUT, L_EYE_IN);
    if ~face_valid
        lm = [];
    end

    %% Face not detected
    if isempty(lm)
        out = drawAttentionGrid(frame, w, h, false, GRID_COLOR, CENTER_BOX_OK, CENTER_BOX_WARN);
        out = insertText(out, [w/2, h/2], 'No face detected', ...
            'FontSize', 18, 'BoxColor', [200 0 0], 'BoxOpacity', 0.6, ...
            'TextColor', 'white', 'AnchorPoint', 'CenterCenter');
        out = addOverlay(out, 'Warning', hr_bpm, 0);
        if RECORD; writeVideo(vw, out); end
        hImg = showFrame(hImg, out, hAx);
        set(hNoFaceTxt, 'Visible', 'on');
        drawnow limitrate;
        continue;
    end
    set(hNoFaceTxt, 'Visible', 'off');   

    face_in_center = isFaceInsideCenter(lm, w, h);
    if ~face_in_center
        out = drawLandmarks(frame, lm);
        out = drawAttentionGrid(out, w, h, false, GRID_COLOR, CENTER_BOX_OK, CENTER_BOX_WARN);
        out = insertText(out, [w/2, 40], FACE_GUIDE_MSG, ...
            'FontSize', 18, 'BoxColor', [200 0 0], 'BoxOpacity', 0.75, ...
            'TextColor', 'white', 'AnchorPoint', 'CenterTop');
        out = addOverlay(out, 'Warning', hr_bpm, 0);
        if RECORD; writeVideo(vw, out); end
        hImg = showFrame(hImg, out, hAx);
        drawnow limitrate;
        continue;
    end

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
        [hr_raw, hr_quality] = estimateHR(rgb_buf, t_buf, BPM_RANGE);
        if isfinite(hr_raw) && hr_quality > 0.10
            if isempty(hr_hist) || abs(hr_raw - median(hr_hist(max(1,end-2):end))) <= 18 || hr_quality >= 0.22
                hr_hist(end+1,1) = hr_raw;
                hr_hist = hr_hist(max(1, end-4):end);
                hr_bpm = median(hr_hist);
            end
        end
        t_last_hr = t_now;
    end

    %% Eye Aspect Ratio 
    ear_r = metrics.ear_r;
    ear_l = metrics.ear_l;
    ear   = metrics.ear;
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
    nose_x  = metrics.nose_rel_x;
    nose_y  = metrics.nose_rel_y;
    nose_dx = nose_x - nose_cx;
    nose_dy = nose_y - nose_cy;
    % Owl is based on head pose relative to the eyes, not absolute image position.
    head_away = ~eyes_closed && ...
        (abs(nose_dx) > NOSE_THRESH(1) || abs(nose_dy) > NOSE_THRESH(2));

    % Long owl + short owl episode tracking
    owl_short_window_start = t_now - SHORT_OWL_WIN;
    owl_short_segments = owl_short_segments(owl_short_segments(:,2) >= owl_short_window_start, :);
    owl_completed_short_sec = cumulativeWindowDuration(owl_short_segments, owl_short_window_start, t_now);
    owl_current_short_sec = 0;

    if head_away
        if isnan(head_away_start); head_away_start = t_now; end
        owl_away_sec = t_now - head_away_start;
        if owl_away_sec >= LONG_OWL_SEC
            long_alarm = true;
            if short_alarm && owl_completed_short_sec < SHORT_OWL_CUM
                short_alarm = false;
                owl_focus_start = NaN;
                owl_short_alarm_since = NaN;
            end
        else
            owl_current_short_sec = owl_away_sec;
        end
    else
        if ~isnan(head_away_start)
            owl_away_sec = t_now - head_away_start;
            if owl_away_sec > 0 && owl_away_sec < LONG_OWL_SEC
                owl_short_segments(end+1,:) = [head_away_start, t_now];
            end
        end
        head_away_start = NaN;
        long_alarm = false;   
    end

    % Short owl: cumulative duration of only short away episodes in 30 s
    cumul_away = owl_completed_short_sec + owl_current_short_sec;
    if short_alarm
        if ~head_away
            if isnan(owl_focus_start); owl_focus_start = t_now; end
            if (t_now - owl_focus_start) >= SHORT_OWL_RESET
                short_alarm = false;
                owl_focus_start = NaN;
                owl_short_alarm_since = NaN;
                owl_short_segments = zeros(0,2);  % fresh window after recovery
            end
        else
            owl_focus_start = NaN;
        end
    else
        if cumul_away >= SHORT_OWL_CUM && ~long_alarm
            short_alarm       = true;
            owl_focus_start   = NaN;
            owl_short_alarm_since = t_now;
        end
    end

    %% Iris gaze in Lizard distraction
    r_offset   = metrics.r_iris_h - r_iris_base(1);
    l_offset   = metrics.l_iris_h - l_iris_base(1);
    r_offset_y = metrics.r_iris_v - r_iris_base(2);
    l_offset_y = metrics.l_iris_v - l_iris_base(2);

    lizard_away = ~head_away && ~eyes_closed && ...
        (abs(r_offset) > IRIS_THRESH(1) || abs(l_offset) > IRIS_THRESH(1) || ...
         abs(r_offset_y) > IRIS_THRESH(2) || abs(l_offset_y) > IRIS_THRESH(2));

    % Lizard long + short episode tracking
    lizard_short_window_start = t_now - SHORT_OWL_WIN;
    lizard_short_segments = lizard_short_segments(lizard_short_segments(:,2) >= lizard_short_window_start, :);
    lizard_completed_short_sec = cumulativeWindowDuration(lizard_short_segments, lizard_short_window_start, t_now);
    lizard_current_short_sec = 0;

    if lizard_away
        if isnan(lizard_away_start); lizard_away_start = t_now; end
        lizard_away_sec = t_now - lizard_away_start;
        if lizard_away_sec >= LONG_OWL_SEC
            lizard_alarm_long = true;
            if lizard_alarm_short && lizard_completed_short_sec < SHORT_OWL_CUM
                lizard_alarm_short = false;
                lizard_focus_start = NaN;
                lizard_short_alarm_since = NaN;
            end
        else
            lizard_current_short_sec = lizard_away_sec;
        end
    else
        if ~isnan(lizard_away_start)
            lizard_away_sec = t_now - lizard_away_start;
            if lizard_away_sec > 0 && lizard_away_sec < LONG_OWL_SEC
                lizard_short_segments(end+1,:) = [lizard_away_start, t_now];
            end
        end
        lizard_away_start = NaN;
        lizard_alarm_long = false;
    end

    % Lizard short: cumulative duration of only short away episodes in 30 s
    cumul_liz = lizard_completed_short_sec + lizard_current_short_sec;
    if lizard_alarm_short
        if ~lizard_away
            if isnan(lizard_focus_start); lizard_focus_start = t_now; end
            if (t_now - lizard_focus_start) >= SHORT_OWL_RESET
                lizard_alarm_short = false;
                lizard_focus_start = NaN;
                lizard_short_alarm_since = NaN;
                lizard_short_segments = zeros(0,2);  % flush to prevent immediate re-trigger
            end
        else
            lizard_focus_start = NaN;
        end
    else
        if cumul_liz >= SHORT_OWL_CUM && ~lizard_alarm_long
            lizard_alarm_short = true;
            lizard_focus_start = NaN;
            lizard_short_alarm_since = t_now;
        end
    end

    %% Overall driver state (Sleep > Microsleep > Long > Short > Focused) 
    state_alarm_sec = 0;
    if eye_alarm == 2
        state = 'Sleep';
        state_alarm_sec = max(0, t_now - eye_close_start - SLEEP_SEC);
    elseif eye_alarm == 1
        state = 'Microsleep';
    elseif long_alarm
        state = 'Distracted (long owl)';
        if ~isnan(head_away_start)
            state_alarm_sec = max(0, t_now - head_away_start - LONG_OWL_SEC);
        end
    elseif lizard_alarm_long
        state = 'Distracted (long lizard)';
        if ~isnan(lizard_away_start)
            state_alarm_sec = max(0, t_now - lizard_away_start - LONG_OWL_SEC);
        end
    elseif short_alarm && lizard_alarm_short
        if lizard_away
            state = 'Distracted (short lizard)';
        elseif head_away
            state = 'Distracted (short owl)';
        elseif owl_short_alarm_since >= lizard_short_alarm_since
            state = 'Distracted (short owl)';
        else
            state = 'Distracted (short lizard)';
        end
    elseif short_alarm
        state = 'Distracted (short owl)';
    elseif lizard_alarm_short
        state = 'Distracted (short lizard)';
    else
        state = 'Focused on the road';
    end

    %%  Compose output frame 
    out = drawLandmarks(frame, lm);   % eye contours (red), iris (green), nose (blue)
    out = drawAttentionGrid(out, w, h, true, GRID_COLOR, CENTER_BOX_OK, CENTER_BOX_WARN);
    out = insertText(out, [w/2, 5], ...
        sprintf(['EAR %.3f<th %.3f | irisR %+.3f/%+.3f  irisL %+.3f/%+.3f | ' ...
                 'noseRel %.3f/%.3f (d %+.3f/%+.3f)'], ...
                ear, EAR_THRESH, ...
                r_offset, r_offset_y, l_offset, l_offset_y, ...
                nose_x, nose_y, nose_dx, nose_dy), ...
        'FontSize', 12, 'BoxColor', [0 0 0], 'BoxOpacity', 0.5, ...
        'TextColor', 'white', 'AnchorPoint', 'CenterTop');
    out = addOverlay(out, state, hr_bpm, state_alarm_sec);
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

function total_sec = cumulativeWindowDuration(segments, winStart, winEnd)
    total_sec = 0;
    if isempty(segments)
        return;
    end

    segStarts = max(segments(:,1), winStart);
    segEnds   = min(segments(:,2), winEnd);
    total_sec = sum(max(0, segEnds - segStarts));
end

function out = drawAttentionGrid(frame, w, h, faceInCenter, gridColor, okColor, warnColor)
    out = frame;

    x1 = round(w*0.25);
    x2 = round(w*0.75);
    y1 = round(h*0.25);
    y2 = round(h*0.75);

    boxColor = okColor;
    if ~faceInCenter
        boxColor = warnColor;
    end

    centerRect = [x1 y1 max(1, x2 - x1) max(1, y2 - y1)];
    dashLen = max(24, round(min(w, h) * 0.05));
    gapLen = max(18, round(min(w, h) * 0.035));

    out = drawDashedLine(out, [x1 1], [x1 h], dashLen, gapLen, gridColor, 1);
    out = drawDashedLine(out, [x2 1], [x2 h], dashLen, gapLen, gridColor, 1);
    out = drawDashedLine(out, [1 y1], [w y1], dashLen, gapLen, gridColor, 1);
    out = drawDashedLine(out, [1 y2], [w y2], dashLen, gapLen, gridColor, 1);
    out = insertShape(out, 'Rectangle', centerRect, 'Color', boxColor, 'LineWidth', 3);
end

function out = drawDashedLine(frame, startPt, endPt, dashLen, gapLen, color, lineWidth)
    out = frame;
    totalLen = hypot(endPt(1) - startPt(1), endPt(2) - startPt(2));
    if totalLen <= 0
        return;
    end

    direction = (endPt - startPt) / totalLen;
    stepLen = dashLen + gapLen;
    dashStarts = 0:stepLen:totalLen;

    for i = 1:numel(dashStarts)
        segStart = startPt + direction * dashStarts(i);
        segEnd = startPt + direction * min(dashStarts(i) + dashLen, totalLen);
        out = insertShape(out, 'Line', [segStart segEnd], ...
            'Color', color, 'LineWidth', lineWidth);
    end
end

function isInside = isFaceInsideCenter(lm, w, h)
    faceOval = [10 338 297 332 284 251 389 356 454 323 361 288 397 365 ...
                379 378 400 377 152 148 176 149 150 136 172 58 132 93 ...
                234 127 162 21 54 103 67 109] + 1;
    facePts = lm(faceOval, 1:2) .* [w h];

    faceCenter = mean([min(facePts, [], 1); max(facePts, [], 1)], 1);
    xMin = w*0.25;
    xMax = w*0.75;
    yMin = h*0.25;
    yMax = h*0.75;

    isInside = faceCenter(1) >= xMin && faceCenter(1) <= xMax && ...
        faceCenter(2) >= yMin && faceCenter(2) <= yMax;
end

function sigma = robustStd(x)
    x = x(isfinite(x));
    if isempty(x)
        sigma = 0;
        return;
    end
    sigma = 1.4826 * median(abs(x - median(x)));
end

function pythonExe = resolvePythonExecutable(scriptDir)
    pythonExe = "";
    candidates = strings(0,1);

    envPython = string(strtrim(getenv('DMS_PYTHON')));
    if strlength(envPython) > 0
        candidates(end+1,1) = envPython;
    end

    homeDir = string(getenv('HOME'));
    if strlength(homeDir) > 0
        candidates(end+1,1) = fullfile(homeDir, "dms_env", "bin", "python");
    end
    candidates(end+1,1) = fullfile(string(scriptDir), "venv", "bin", "python");

    [statusPy3, cmdoutPy3] = system('command -v python3');
    if statusPy3 == 0
        candidates(end+1,1) = string(strtrim(cmdoutPy3));
    end

    [statusPy, cmdoutPy] = system('command -v python');
    if statusPy == 0
        candidates(end+1,1) = string(strtrim(cmdoutPy));
    end

    candidates = unique(candidates(candidates ~= ""));
    for k = 1:numel(candidates)
        candidate = candidates(k);
        if contains(candidate, string(filesep) + ".pyenv" + string(filesep) + "shims" + string(filesep))
            continue;
        end
        if isUsablePython(candidate)
            pythonExe = candidates(k);
            return;
        end
    end
end

function tf = isUsablePython(pythonPath)
    tf = false;
    if strlength(pythonPath) == 0 || ~isfile(pythonPath)
        return;
    end

    pythonPath = string(pythonPath);
    escapedPath = strrep(char(pythonPath), '''', '''''');
    cmd = sprintf('''%s'' -c "import mediapipe, cv2; print(''ok'')"', escapedPath);
    [status, cmdout] = system(cmd);
    tf = (status == 0) && contains(string(cmdout), "ok");
end

function handleKeyPress(src, evt)
    if nargin < 2 || isempty(evt) || ~isprop(evt, 'Key')
        return;
    end

    switch lower(string(evt.Key))
        case "c"
            setappdata(src, 'recalibrate', true);
        case "s"
            setappdata(src, 'stop', true);
    end
end
