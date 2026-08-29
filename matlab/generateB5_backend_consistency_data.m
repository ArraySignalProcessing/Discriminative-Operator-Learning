% generateB5_backend_consistency_data.m
% B5 data for grid/root backend consistency.
% Ideal independent-source setting to isolate grid/root backend consistency.

clear; close all;
tic;
rng(42);

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B5');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

M            = 10;
K            = 2;
theta_true   = [-9.2, 13.8];
SNR_dB       = 20;
T            = 1000;
signal_power = 1;
N_mc         = 1000;

steer_vec = @(theta, N) exp(1j*pi*sin(deg2rad(theta))*(0:N-1).');
noise_power = signal_power * 10^(-SNR_dB/10);
sigma_n = sqrt(noise_power);

A = zeros(M, K);
for k = 1:K
    A(:, k) = steer_vec(theta_true(k), M);
end

r_sam = zeros(M, M, 2, N_mc);
true_angles = repmat(theta_true(:), 1, N_mc);

fprintf('Generating B5 backend-compatibility data: SNR=%d dB, T=%d, N_mc=%d...\n', ...
    SNR_dB, T, N_mc);

for mc = 1:N_mc
    S = (randn(K, T) + 1j * randn(K, T)) / sqrt(2);
    Noise = sigma_n * (randn(M, T) + 1j * randn(M, T)) / sqrt(2);
    Y = A * S + Noise;
    Ry = (Y * Y') / T;
    r_sam(:, :, 1, mc) = real(Ry);
    r_sam(:, :, 2, mc) = imag(Ry);
end

fname = fullfile(data_dir, 'B5_BackendCompat.h5');
if exist(fname, 'file')
    delete(fname);
end

h5create(fname, '/sam', size(r_sam));
h5write(fname, '/sam', r_sam);
h5create(fname, '/angles', size(true_angles));
h5write(fname, '/angles', true_angles);

h5writeatt(fname, '/', 'M', M);
h5writeatt(fname, '/', 'K', K);
h5writeatt(fname, '/', 'theta_true', theta_true);
h5writeatt(fname, '/', 'SNR_dB', SNR_dB);
h5writeatt(fname, '/', 'T', T);
h5writeatt(fname, '/', 'N_mc', N_mc);
h5writeatt(fname, '/', 'scenario', 'ideal_independent_backend_compatibility');

fprintf('Saved B5 data to %s\n', fname);
fprintf('Done (%.2f min).\n', toc / 60);
