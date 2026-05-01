%% ICA numerical example: 2 hidden sources, 2 mixtures, FastICA recovery
clear; close all; clc;

rng(0);

%% 1) Create two hidden independent non-Gaussian sources
T = 60;
Fs = 40;
N = T*Fs;
t = linspace(0, T, N);


% Source 1: PPG-like waveform
%s1 = sawtooth_manual(t, 1.0);
s1 = simulatePPG65(Fs,T);

% Source 2: sawtooth-like waveform
s2 = vibration(0.15,t, 2.0);
%s2 = sign(sin(2*pi*1.7*t));

% Stack sources as rows
S = [s1; s2];

% Add small noise
S = S + 0.3 * randn(size(S));

% Standardize each source
S = S - mean(S, 2);
S = S ./ std(S, 0, 2);

%% 2) Mix the sources
A = [1.0 0.6;
     0.5 1.2];

X = A * S;   % observed mixtures

%% 3) Whiten the observed mixtures
[Z, V] = whiten_rows(X);

%% 4) Apply FastICA (deflation version)
num_components = 2;
max_iter = 1000;
tol = 1e-7;

[S_est, W] = fastica_deflation(Z, num_components, max_iter, tol);

%% 5) Match recovered components to true sources
C = corr_rows(S, S_est);
absC = abs(C);

if absC(1,1) + absC(2,2) >= absC(1,2) + absC(2,1)
    perm = [1 2];
else
    perm = [2 1];
end

S_rec = S_est(perm, :);

% Fix sign ambiguity
C2 = corr_rows(S, S_rec);
for k = 1:2
    if C2(k,k) < 0
        S_rec(k,:) = -S_rec(k,:);
    end
end

C_final = corr_rows(S, S_rec);

%% 6) Display numerical results
disp('True mixing matrix A:');
disp(A);

disp('Correlation between true sources and recovered components:');
disp(C_final);

%% 7) Plots

figure;
plot(t, S(1,:), 'LineWidth', 1.2);
title('Hidden source 1');
xlabel('Time');
ylabel('Amplitude');
grid on;

figure;
plot(t, S(2,:), 'LineWidth', 1.2);
title('Hidden source 2');
xlabel('Time');
ylabel('Amplitude');
grid on;

figure;
plot(t, X(1,:), 'LineWidth', 1.2);
title('Observed mixture 1');
xlabel('Time');
ylabel('Amplitude');
grid on;

figure;
plot(t, X(2,:), 'LineWidth', 1.2);
title('Observed mixture 2');
xlabel('Time');
ylabel('Amplitude');
grid on;

figure;
plot(t, S_rec(1,:), 'LineWidth', 1.2);
title('Recovered component 1');
xlabel('Time');
ylabel('Amplitude');
grid on;

figure;
plot(t, S_rec(2,:), 'LineWidth', 1.2);
title('Recovered component 2');
xlabel('Time');
ylabel('Amplitude');
grid on;

figure;
scatter(X(1,:), X(2,:), 8, 'filled');
title('Scatter of observed mixtures');
xlabel('Mixture 1');
ylabel('Mixture 2');
grid on;

figure;
scatter(S_rec(1,:), S_rec(2,:), 8, 'filled');
title('Scatter after ICA');
xlabel('Recovered component 1');
ylabel('Recovered component 2');
grid on;

%% 8) Overlay comparison: true vs recovered
figure;
subplot(2,1,1);
plot(t, S(1,:), 'LineWidth', 1.2); hold on;
plot(t, S_rec(1,:), '--', 'LineWidth', 1.2);
title('Source 1 vs recovered component 1');
xlabel('Time');
ylabel('Amplitude');
legend('True source 1', 'Recovered 1');
grid on;

subplot(2,1,2);
plot(t, S(2,:), 'LineWidth', 1.2); hold on;
plot(t, S_rec(2,:), '--', 'LineWidth', 1.2);
title('Source 2 vs recovered component 2');
xlabel('Time');
ylabel('Amplitude');
legend('True source 2', 'Recovered 2');
grid on;

%% extract HR

ppg = S_rec(1,:);
df = Fs/length(ppg);
freq=0:df:(length(ppg)-1)*df;
figure;plot(ppg);
f_ppg=fft(ppg);
psd=abs(f_ppg);
figure;plot(freq(1:(210/60)/df)*60,psd(1:(210/60)/df));
title('Power spectrum of estimated rPPG');
xlabel('HR [BPM]');
[pk,loc]=max(psd(1:(210/60)/df));
fprintf("Estimated HR: %.2f\n", 60*freq(loc));
xline(60*freq(loc),'Color','r');
str = sprintf('Estimated HR: %.2f',60*freq(loc));
text(68,1000, str);
%% ---------- Local functions ----------

function y = vibration(A, t, freq)
    x = freq * t;
    y = A*sin(2*pi*x);
end

function [Z, V] = whiten_rows(X)
    % X is [m x N], rows = signals
    Xc = X - mean(X, 2);
    C = cov(Xc');                 % covariance across rows
    [E, D] = eig(C);

    % sort eigenvalues descending
    [d, idx] = sort(diag(D), 'descend');
    E = E(:, idx);
    D = diag(d);

    V = diag(1 ./ sqrt(diag(D))) * E';
    Z = V * Xc;
end

function [S_est, W] = fastica_deflation(Z, n_components, max_iter, tol)
    % Z is whitened data, size [m x N]
    [m, N] = size(Z);
    W = zeros(n_components, m);

    for p = 1:n_components
        w = randn(m,1);
        w = w / norm(w);

        for iter = 1:max_iter
            w_old = w;

            y = w' * Z;
            g = tanh(y);
            gp = 1 - g.^2;

            % FastICA fixed-point update
            w = (Z * g') / N - mean(gp) * w_old;

            % Orthogonalize against previous components
            if p > 1
                w = w - W(1:p-1,:)' * (W(1:p-1,:) * w);
            end

            % Normalize
            w = w / norm(w);

            % Convergence test
            if abs(abs(w' * w_old) - 1) < tol
                break;
            end
        end

        W(p,:) = w';
    end

    S_est = W * Z;
end

function C = corr_rows(A, B)
    % Correlation matrix between rows of A and rows of B
    na = size(A,1);
    nb = size(B,1);
    C = zeros(na, nb);

    for i = 1:na
        a = A(i,:) - mean(A(i,:));
        for j = 1:nb
            b = B(j,:) - mean(B(j,:));
            C(i,j) = (a * b') / sqrt((a*a') * (b*b'));
        end
    end
end

function [ppg, t] = simulatePPG65(Fs, T)
%SIMULATEPPG65 Simulate a synthetic PPG signal for 65 bpm
%
%   [ppg, t] = simulatePPG65(Fs, T)
%
% Inputs:
%   Fs : sampling frequency in Hz
%   T  : duration in seconds
%
% Outputs:
%   ppg : simulated PPG signal
%   t   : time vector
%
% Example:
%   [ppg, t] = simulatePPG65(100, 20);
%   plot(t, ppg); grid on;
%   xlabel('Time [s]'); ylabel('Amplitude');
%   title('Simulated PPG at 65 bpm');

    if nargin < 1
        Fs = 100;
    end
    if nargin < 2
        T = 20;
    end

    HR_bpm = 65;
    f_hr = HR_bpm / 60;

    t = 0:1/Fs:T-1/Fs;

    % Fundamental + harmonics
    ppg1 = sin(2*pi*f_hr*t);
    ppg2 = 0.4 * sin(2*pi*2*f_hr*t - pi/4);
    ppg3 = 0.15 * sin(2*pi*3*f_hr*t - pi/2);

    ppg_clean = ppg1 + ppg2 + ppg3;
    ppg_clean = ppg_clean - min(ppg_clean);

    % Baseline drift
    f_resp = 0.25;
    baseline = 0.2 * sin(2*pi*f_resp*t);

    % Noise
    noise = 0.03 * randn(size(t));

    % Final signal
    ppg = ppg_clean + baseline + noise;
end