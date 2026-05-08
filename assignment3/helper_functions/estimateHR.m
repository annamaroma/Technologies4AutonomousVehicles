% Anna Roma - s345819
% Assignment 3 - Driver Monitoring System

%% estimate heart rate (HR) in beats per minute (bpm) from RGB and timestamp buffers
function hr_bpm = estimateHR(rgb_buf, t_buf, bpmRange)
    hr_bpm = NaN;
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
    rgb_uniform = rgb_uniform ./ max(mean(rgb_uniform, 1), eps);
    rgb_uniform = detrend(rgb_uniform);

    sig = rgb_uniform(:,2) - 0.5 * (rgb_uniform(:,1) + rgb_uniform(:,3));
    sig = sig - mean(sig);
    win = 0.5 - 0.5 * cos(2 * pi * (0:numel(sig)-1)' / max(numel(sig)-1, 1));
    sig = sig .* win;

    nfft = 2^nextpow2(numel(sig));
    Y = fft(sig, nfft);
    P = abs(Y(1:floor(nfft/2)+1)).^2;
    f = (0:floor(nfft/2))' * (fs / nfft);
    bpm = 60 * f;

    keep = bpm >= bpmRange(1) & bpm <= bpmRange(2);
    if ~any(keep)
        return;
    end

    [~, idx] = max(P(keep));
    bpmCand = bpm(keep);
    hr_bpm = bpmCand(idx);
end
