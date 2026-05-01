function [bpm, quality, debugInfo] = estimateHeartRateRPPG(timestamps, rgbMeans, config, previousBpm)
%ESTIMATEHEARTRATERPPG Estimate BPM from buffered RGB traces.
%   The preferred path uses the provided fastICA implementation. If ICA
%   fails, the function falls back to the green channel spectrum so the
%   system remains robust in real time.

if nargin < 4
    previousBpm = NaN;
end

bpm = previousBpm;
quality = 0;
debugInfo = struct("updated", false, "method", "hold", "sampleRate", NaN);

if numel(timestamps) < 2 || size(rgbMeans, 1) < 2
    return;
end

durationSec = timestamps(end) - timestamps(1);
if durationSec < config.hrMinWindowSec
    return;
end

medianDt = median(diff(timestamps), "omitnan");
if ~isfinite(medianDt) || medianDt <= 0
    return;
end

fs = 1 / medianDt;
debugInfo.sampleRate = fs;

if size(rgbMeans, 1) < max(config.hrMinSamples, 3 * fs)
    return;
end

uniformTime = (timestamps(1):medianDt:timestamps(end)).';
if numel(uniformTime) < max(config.hrMinSamples, 3 * fs)
    return;
end

resampled = zeros(numel(uniformTime), 3);
for channelIdx = 1:3
    resampled(:, channelIdx) = interp1(timestamps, rgbMeans(:, channelIdx), uniformTime, "linear", "extrap");
end

resampled = detrend(resampled, "linear");
resampled = resampled - mean(resampled, 1, "omitnan");
resampled = resampled ./ max(std(resampled, 0, 1, "omitnan"), eps);

bandHz = config.hrBandBpm / 60;
[bBand, aBand] = butter(3, bandHz / (fs / 2), "bandpass");
filtered = filtfilt(bBand, aBand, resampled);

components = [];
try
    [icasig, ~, ~] = fastica(filtered.', ...
        'numOfIC', 3, ...
        'verbose', 'off', ...
        'displayMode', 'off', ...
        'approach', 'symm');
    components = icasig.';
    debugInfo.method = "fastICA";
catch
    components = filtered;
    debugInfo.method = "green_fallback";
end

numSamples = size(components, 1);
nfft = 2 ^ nextpow2(numSamples);
freqHz = (0:(nfft / 2))' * fs / nfft;
freqMask = freqHz >= bandHz(1) & freqHz <= bandHz(2);
if ~any(freqMask)
    return;
end

bestPeakPower = -inf;
bestFreq = NaN;
for componentIdx = 1:size(components, 2)
    signal = components(:, componentIdx);
    signal = signal - mean(signal, "omitnan");
    spectrum = abs(fft(signal .* hann(numSamples), nfft)).^2;
    oneSided = spectrum(1:(nfft / 2 + 1));
    bandPower = oneSided(freqMask);
    [peakPower, peakIdx] = max(bandPower);
    if isempty(peakPower) || peakPower <= 0
        continue;
    end

    localFreqs = freqHz(freqMask);
    localFreq = localFreqs(peakIdx);
    normalizedPower = peakPower / max(sum(bandPower), eps);

    if normalizedPower > bestPeakPower
        bestPeakPower = normalizedPower;
        bestFreq = localFreq;
    end
end

quality = max(0, bestPeakPower);
if isfinite(bestFreq) && quality >= config.hrMinQuality
    bpm = 60 * bestFreq;
    debugInfo.updated = true;
end
