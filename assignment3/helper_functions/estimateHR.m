% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% estimate heart rate (HR) in beats per minute (bpm) from RGB and timestamp buffers
function [hr_bpm, quality] = estimateHR(rgb_buf, t_buf, bpmRange)
    hr_bpm = NaN;
    quality = 0;
    if size(rgb_buf,1) < 60 || numel(t_buf) < 60
        return;
    end

    t_buf = t_buf(:);
    dt = median(diff(t_buf));
    if ~isfinite(dt) || dt <= 0
        return;
    end
    fs = 1 / dt;
    if ~isfinite(fs) || fs <= 0
        return;
    end

    t_uniform = (t_buf(1):dt:t_buf(end))';
    if numel(t_uniform) < 60
        return;
    end

    rgb_uniform = interp1(t_buf, double(rgb_buf), t_uniform, 'linear', 'extrap');
    rgb_uniform = rgb_uniform ./ max(mean(rgb_uniform, 1), eps) - 1;
    rgb_uniform = detrend(rgb_uniform);

    loHz = bpmRange(1) / 60;
    hiHz = bpmRange(2) / 60;
    if hiHz >= fs / 2
        hiHz = 0.95 * (fs / 2);
    end
    if loHz <= 0 || hiHz <= loHz
        return;
    end

    [b, a] = butter(3, [loHz hiHz] / (fs / 2), 'bandpass');
    rgb_filt = filtfilt(b, a, rgb_uniform);

    % Use the suggested ICA approach on the filtered RGB traces.
    comp = [];
    try
        comp = fastica(rgb_filt', 'numOfIC', 3, 'approach', 'symm', ...
                       'g', 'tanh', 'stabilization', 'on', ...
                       'verbose', 'off', 'displayMode', 'off');
        comp = comp';
    catch
    end

    candidates = rgb_filt;
    if ~isempty(comp)
        candidates = [candidates, comp];
    end

    nfft = max(1024, 2^nextpow2(numel(t_uniform)));
    f = (0:floor(nfft/2))' * (fs / nfft);
    bpm = 60 * f;
    keep = bpm >= bpmRange(1) & bpm <= bpmRange(2);
    if ~any(keep)
        return;
    end

    bestScore = -Inf;
    bestBpm = NaN;

    for k = 1:size(candidates, 2)
        sig = candidates(:,k);
        sig = sig - mean(sig);
        sig = filtfilt(b, a, sig);
        if any(~isfinite(sig))
            continue;
        end

        win = 0.5 - 0.5 * cos(2 * pi * (0:numel(sig)-1)' / max(numel(sig)-1, 1));
        sig = sig .* win;
        Y = fft(sig, nfft);
        P = abs(Y(1:floor(nfft/2)+1)).^2;
        Pkeep = P(keep);
        if ~any(isfinite(Pkeep)) || all(Pkeep <= 0)
            continue;
        end

        Pkeep = smoothdata(Pkeep, 'movmean', 5);
        [peakPower, idx] = max(Pkeep);
        totalPower = sum(Pkeep) + eps;
        bpmCand = bpm(keep);
        thisBpm = bpmCand(idx);
        localFloor = median(Pkeep(max(1, idx-10):min(numel(Pkeep), idx+10))) + eps;
        peakProminence = peakPower / localFloor;
        thisScore = 0.65 * (peakPower / totalPower) + 0.35 * min(peakProminence, 10);

        if thisScore > bestScore
            bestScore = thisScore;
            bestBpm = thisBpm;
        end
    end

    if isfinite(bestBpm) && bestScore > 0.10
        hr_bpm = bestBpm;
        quality = bestScore;
    end
end
