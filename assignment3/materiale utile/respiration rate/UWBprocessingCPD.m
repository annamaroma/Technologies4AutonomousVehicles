%% Magnitude-Only Peak Tracking for Breathing Rate Estimation
clear; clc; close all;

doPlot = 1;
doSimChild = 1;

% --- 1. Simulation parameters ---
fs = 40;                % Frame rate (Hz)
duration = 60;          % Seconds
t = 0:1/fs:duration-1/fs;
numRanges = 200;
rangeAxis = linspace(0.5, 1.50, numRanges);   % Range bins in meters

if doSimChild == 1
    % Simulate breathing motion: range 0.75 m subject A
    f_breath_subA = 35/60;        % Breathing frequency (Hz) -> 35 breaths/min
    displacement = 0.005 * sin(2*pi*f_breath_subA*t); % Abdominal displacement 5mm
    trueRange_subA = 0.75 + displacement;
else
    trueRange_subA = zeros(length(t));
end

% --- 2. Generate simulated range profiles ---
profiles = zeros(numRanges, length(t));
for i = 1:length(t)
    echo_A = exp(-((rangeAxis - trueRange_subA(i)).^2) / (2*(0.01)^2)); % Gaussian echo
    noise = 0.05 * randn(1, numRanges);
    profiles(:, i) = echo_A+noise;
end

% --- 3. Peak tracking over time ---
[~, peakIdx] = max(profiles, [], 1);
rangeEst = rangeAxis(peakIdx);

if mean(rangeEst) < (0.9*0.75) || mean(rangeEst) > (1.1*0.75)
    disp 'No Child in the radar field of view'
else
    rangeEst = rangeEst - mean(rangeEst); % Remove DC offset

    % --- 4. Band-pass filter (5–20 bpm respiration band) ---
    lowCut = 30/60; highCut = 60/60;
    [b, a] = butter(4, [lowCut, highCut]/(fs/2), 'bandpass');
    filtered = filtfilt(b, a, rangeEst);

    % --- 5. Estimate breathing rate via Welch PSD ---
    [pxx, f] = pwelch(filtered, hamming(fs*30), fs*15, [], fs);
    freqBand = (f >= lowCut) & (f <= highCut);
    %[~, idx] = max(pxx(freqBand));
    %f_resp = f(freqBand);
    %breathingRate_bpm = 60 * f_resp(idx);

    [m, ~] = max(pxx(freqBand));
    [~, idx] = findpeaks(pxx(freqBand), 'MinPeakHeight',m/5,'MinPeakDistance',2);
    f_resp = f(freqBand);
    breathingRate_bpm = 60 * f_resp(idx);

    fprintf('Estimated Breathing Rate: %.1f breaths/min\n', breathingRate_bpm);

    % --- 6. Plot results ---
    if doPlot == 1
        figure;
        subplot(2,1,1);
        plot(t, rangeEst*1000, 'b');
        title('Tracked Chest Displacement (Magnitude-Only Peak Tracking)');
        xlabel('Time (s)');
        ylabel('Displacement (mm)');
        grid on;

        subplot(2,1,2);
        semilogy(f(1:60)*60, pxx(1:60), 'LineWidth', 1.2);
        hold on;
        % for i=1:length(idx)
        %     xline(f_resp(idx(i)), 'r--', sprintf('%.1f bpm', breathingRate_bpm(i)));
        % end
        title('Power Spectral Density (Welch)');
        xlabel('RR (bpm)');
        ylabel('Power');
        grid on;
    end
end
