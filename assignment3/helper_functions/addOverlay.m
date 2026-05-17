% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% overlay functions
function out = addOverlay(frame, state, hr_bpm, state_alarm_sec)
    out = frame;
    [h, w, ~] = size(frame);

    switch state
        case 'Focused on the road'
            stateColor = [40 205 50]; %lime green
        case {'Distracted (short owl)', 'Distracted (short lizard)'}
            stateColor = [253 117 1]; %orange
        case {'Distracted (long owl)', 'Distracted (long lizard)'}
            stateColor = [180 21 0];
        case 'Microsleep'
            stateColor = [220 120 0];
        case 'Sleep'
            stateColor = [180 21 0];
        case 'Warning'
            stateColor = [180 21 0];
        otherwise
            stateColor = [40 40 40];
    end

    stateFont = 18;
    if strcmp(state, 'Sleep') || contains(state, 'Distracted (long')
        stateFont = min(42, 18 + 4 * floor(max(0, state_alarm_sec)));
    end

    if isnan(hr_bpm)
        hrText = 'HR: -- bpm';
    else
        hrText = sprintf('HR: %.0f bpm', hr_bpm);
    end

    %add HR text bottom left
    out = insertText(out, [10, h - 10], hrText, ...
        'FontSize', 16, 'BoxColor', [0 0 0], 'BoxOpacity', 0.55, ...
        'TextColor', 'white', 'AnchorPoint', 'LeftBottom');

    %add state text bottom right
    out = insertText(out, [w - 10, h - 10], state, ...
        'FontSize', stateFont, 'BoxColor', stateColor, 'BoxOpacity', 0.70, ...
        'TextColor', 'white', 'AnchorPoint', 'RightBottom');

end
