% generateB3_operator_response_data.m
% B3: composite-perturbation angular-separation scan for learned-Q
% spectrum-structure analysis.
%
% Data flow:
%   MATLAB: generate test samples and EVD Pn -> data/B3/B3_OperatorResponse_QPn.h5
%   Python: export learned Q              -> data/B3/B3_OperatorResponse_Q.h5
%   MATLAB: plot paper figure             -> figures/plotB3_operator_response.*

clear; close all;
tic;
rng(42);

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B3');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

fname = fullfile(data_dir, 'B3_OperatorResponse_QPn.h5');
if exist(fname, 'file')
    delete(fname);
end

% ================= Adjustable scenario parameters =================
M             = 10;
K             = 2;
T             = 50;
SNR_dB        = 0;
signal_power  = 1;
N_mc          = 1000;

theta_center    = 5;
delta_theta_vec = 2:0.1:60;
angle_scan      = -50:0.05:60;

% Source coherence. Set source_rho = 0 for independent sources.
source_rho = 0.8;

% Array imperfections. Set array_error_enable = false for nominal ULA.
array_error_enable = true;
array_error_rho    = 0.50;    % 0: nominal, 1: full strength below
pos_std_max        = 0.05;    % wavelength
gain_std_max       = 0.10;    % relative standard deviation
phase_std_max_deg  = 10;      % degrees
coupling_mag1_max  = 0.20;    % first-neighbor magnitude
coupling_phase_deg = 45;      % degrees
coupling_decay     = 0.50;
coupling_neighbors = 2;

% ================= Metadata =================
h5create(fname, '/delta_theta_vec', size(delta_theta_vec));
h5write(fname, '/delta_theta_vec', delta_theta_vec);
h5create(fname, '/angle_scan', size(angle_scan));
h5write(fname, '/angle_scan', angle_scan);
h5writeatt(fname, '/', 'M', M);
h5writeatt(fname, '/', 'K', K);
h5writeatt(fname, '/', 'T', T);
h5writeatt(fname, '/', 'SNR_dB', SNR_dB);
h5writeatt(fname, '/', 'N_mc', N_mc);
h5writeatt(fname, '/', 'theta_center', theta_center);
h5writeatt(fname, '/', 'source_rho', source_rho);
h5writeatt(fname, '/', 'array_error_enable', double(array_error_enable));
h5writeatt(fname, '/', 'array_error_rho', array_error_rho);
h5writeatt(fname, '/', 'pos_std_max', pos_std_max);
h5writeatt(fname, '/', 'gain_std_max', gain_std_max);
h5writeatt(fname, '/', 'phase_std_max_deg', phase_std_max_deg);
h5writeatt(fname, '/', 'coupling_mag1_max', coupling_mag1_max);
h5writeatt(fname, '/', 'coupling_phase_deg', coupling_phase_deg);
h5writeatt(fname, '/', 'coupling_decay', coupling_decay);
h5writeatt(fname, '/', 'coupling_neighbors', coupling_neighbors);
h5writeatt(fname, '/', 'scenario', 'B3_operator_response_composite_perturbation');

noise_power = signal_power * 10^(-SNR_dB / 10);
sigma_n = sqrt(noise_power);

fprintf('Generating B3 composite data: SNR=%g dB, T=%d, N_mc=%d\n', ...
    SNR_dB, T, N_mc);
fprintf('source_rho=%.3f, array_error_enable=%d, array_error_rho=%.3f\n', ...
    source_rho, array_error_enable, array_error_rho);

for sep_idx = 1:numel(delta_theta_vec)
    Delta = delta_theta_vec(sep_idx);
    theta_true = [theta_center - Delta / 2, theta_center + Delta / 2];
    grp = sprintf('/Delta_%sdeg', strrep(sprintf('%.1f', Delta), '.', 'p'));

    fprintf('[%2d/%2d] Delta=%.1f deg, theta=[%.2f, %.2f]\n', ...
        sep_idx, numel(delta_theta_vec), Delta, theta_true(1), theta_true(2));

    r_sam = zeros(M, M, 2, N_mc, 'single');
    true_angles = repmat(theta_true(:), 1, N_mc);
    Pn_real = zeros(M, M, N_mc, 'single');
    Pn_imag = zeros(M, M, N_mc, 'single');

    for mc = 1:N_mc
        A_true = build_true_steering(theta_true, M, array_error_enable, array_error_rho, ...
            pos_std_max, gain_std_max, phase_std_max_deg, coupling_mag1_max, ...
            coupling_phase_deg, coupling_decay, coupling_neighbors);

        S = generate_sources(K, T, source_rho);
        X = A_true * S;
        Noise = sigma_n * (randn(M, T) + 1j * randn(M, T)) / sqrt(2);
        Y = X + Noise;
        Ry = (Y * Y') / T;

        [U, D] = eig(Ry, 'vector');
        [~, order] = sort(real(D), 'descend');
        U = U(:, order);
        Un = U(:, K+1:end);
        Pn = Un * Un';
        Pn = Pn * (M / real(trace(Pn)));

        r_sam(:, :, 1, mc) = single(real(Ry));
        r_sam(:, :, 2, mc) = single(imag(Ry));
        Pn_real(:, :, mc) = single(real(Pn));
        Pn_imag(:, :, mc) = single(imag(Pn));
    end

    h5create(fname, [grp '/sam'], size(r_sam), 'Datatype', 'single');
    h5write(fname, [grp '/sam'], r_sam);
    h5create(fname, [grp '/angles'], size(true_angles));
    h5write(fname, [grp '/angles'], true_angles);
    h5create(fname, [grp '/Pn_real'], size(Pn_real), 'Datatype', 'single');
    h5write(fname, [grp '/Pn_real'], Pn_real);
    h5create(fname, [grp '/Pn_imag'], size(Pn_imag), 'Datatype', 'single');
    h5write(fname, [grp '/Pn_imag'], Pn_imag);
    h5writeatt(fname, grp, 'Delta', Delta);
end

fprintf('Saved B3 data to %s\n', fname);
fprintf('Done (%.2f min).\n', toc / 60);

function S = generate_sources(K, T, rho)
    if K ~= 2
        error('B3 generator currently expects K=2.');
    end
    s1 = (randn(1, T) + 1j * randn(1, T)) / sqrt(2);
    u = (randn(1, T) + 1j * randn(1, T)) / sqrt(2);
    rho = min(max(real(rho), 0), 0.999999);
    s2 = rho * s1 + sqrt(max(0, 1 - rho^2)) * u;
    S = [s1; s2];
end

function A = build_true_steering(theta_vec, M, enable_err, err_rho, ...
    pos_std_max, gain_std_max, phase_std_max_deg, coupling_mag1_max, ...
    coupling_phase_deg, coupling_decay, coupling_neighbors)

    nominal_pos = (0:M-1).' * 0.5;
    if enable_err
        pos = nominal_pos + err_rho * pos_std_max * randn(M, 1);
        gain = 1 + err_rho * gain_std_max * randn(M, 1);
        phase = deg2rad(err_rho * phase_std_max_deg * randn(M, 1));
        G = diag(gain .* exp(1j * phase));
        C = eye(M);
        base_c = err_rho * coupling_mag1_max * exp(1j * deg2rad(coupling_phase_deg));
        for lag = 1:coupling_neighbors
            c_lag = base_c * (coupling_decay ^ (lag - 1));
            C = C + diag(c_lag * ones(M-lag, 1), lag) + ...
                    diag(conj(c_lag) * ones(M-lag, 1), -lag);
        end
    else
        pos = nominal_pos;
        G = eye(M);
        C = eye(M);
    end

    A_nom = zeros(M, numel(theta_vec));
    for k = 1:numel(theta_vec)
        A_nom(:, k) = exp(1j * 2 * pi * pos * sin(deg2rad(theta_vec(k))));
    end
    A = C * G * A_nom;
end
