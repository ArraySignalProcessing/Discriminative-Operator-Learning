% generateB6_k123_composite_spectrum_data.m
% B6: K=1/2/3 composite-robustness data for source-count generalization
% spectrum analysis.
%
% Data flow:
%   MATLAB: generate test samples and EVD Pn -> data/B6/B6_K123_CompositeSpectrum_QPn_K*.h5
%   Python: export learned Q from a K=3 model -> data/B6/B6_K123_CompositeSpectrum_Q_K*.h5
%   MATLAB: plot K=1/2/3 spectra -> figures/plotB6_k123_composite_spectrum_K*.*

clear; close all;
tic;
rng(42);

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B6');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

% ================= Adjustable scenario parameters =================
M             = 10;
K_vec         = 1:3;
T             = 500;
SNR_dB        = 0;
signal_power  = 1;
N_mc          = 1000;

theta_center    = 5;
theta_vec       = -50:0.5:50;   % K=1 true-angle scan
delta_theta_k2  = 2:0.1:30;     % K=2 separation scan: center +/- Delta/2
delta_theta_k3  = 2:0.1:20;     % K=3 spacing scan: center + [-Delta, 0, Delta]
angle_scan_k1   = -60:0.05:60;
angle_scan_k2   = -20:0.05:30;
angle_scan_k3   = -25:0.05:35;

% Source coherence. For K>1, a common latent waveform plus independent
% residuals is used to keep the pairwise source correlation high.
source_rho = 0.8;

% Composite array imperfections.
array_error_enable = true;
array_error_rho    = 0.50;    % 0: nominal, 1: full strength below
pos_std_max        = 0.05;    % wavelength
gain_std_max       = 0.10;    % relative standard deviation
phase_std_max_deg  = 10;      % degrees
coupling_mag1_max  = 0.20;    % first-neighbor magnitude
coupling_phase_deg = 45;      % degrees
coupling_decay     = 0.50;
coupling_neighbors = 2;

noise_power = signal_power * 10^(-SNR_dB / 10);
sigma_n = sqrt(noise_power);

fprintf('Generating B6 K=1/2/3 composite data: SNR=%g dB, T=%d, N_mc=%d\n', ...
    SNR_dB, T, N_mc);
fprintf('source_rho=%.3f, array_error_enable=%d, array_error_rho=%.3f\n', ...
    source_rho, array_error_enable, array_error_rho);

for k_idx = 1:numel(K_vec)
    K = K_vec(k_idx);
    fname = fullfile(data_dir, sprintf('B6_K123_CompositeSpectrum_QPn_K%d.h5', K));
    if exist(fname, 'file')
        delete(fname);
    end

    if K == 1
        scan_vec = theta_vec;
        angle_scan = angle_scan_k1;
        h5create(fname, '/theta_vec', size(theta_vec));
        h5write(fname, '/theta_vec', theta_vec);
    elseif K == 2
        scan_vec = delta_theta_k2;
        angle_scan = angle_scan_k2;
        h5create(fname, '/delta_theta_vec', size(delta_theta_k2));
        h5write(fname, '/delta_theta_vec', delta_theta_k2);
    else
        scan_vec = delta_theta_k3;
        angle_scan = angle_scan_k3;
        h5create(fname, '/delta_theta_vec', size(delta_theta_k3));
        h5write(fname, '/delta_theta_vec', delta_theta_k3);
    end

    h5create(fname, '/angle_scan', size(angle_scan));
    h5write(fname, '/angle_scan', angle_scan);
    h5writeatt(fname, '/', 'M', M);
    h5writeatt(fname, '/', 'K', K);
    h5writeatt(fname, '/', 'K_vec', K_vec);
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
    h5writeatt(fname, '/', 'scenario', 'B6_k123_composite_source_count_generalization');

    for scan_idx = 1:numel(scan_vec)
        scan_value = scan_vec(scan_idx);
        theta_true = make_true_angles(theta_center, K, scan_value);
        grp = make_group(K, scan_value);

        if K == 1
            fprintf('[K %d, %3d/%3d] theta=%.1f deg\n', ...
                K, scan_idx, numel(scan_vec), scan_value);
        else
            fprintf('[K %d, %3d/%3d] Delta=%.1f deg, theta=%s\n', ...
                K, scan_idx, numel(scan_vec), scan_value, mat2str(theta_true, 3));
        end

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
        h5create(fname, [grp '/theta_true'], size(theta_true));
        h5write(fname, [grp '/theta_true'], theta_true);
        h5create(fname, [grp '/Pn_real'], size(Pn_real), 'Datatype', 'single');
        h5write(fname, [grp '/Pn_real'], Pn_real);
        h5create(fname, [grp '/Pn_imag'], size(Pn_imag), 'Datatype', 'single');
        h5write(fname, [grp '/Pn_imag'], Pn_imag);
        h5writeatt(fname, grp, 'K', K);
        if K == 1
            h5writeatt(fname, grp, 'theta', scan_value);
        else
            h5writeatt(fname, grp, 'Delta', scan_value);
        end
    end

    fprintf('Saved B6 K=%d data to %s\n', K, fname);
end

fprintf('Done (%.2f min).\n', toc / 60);

function theta = make_true_angles(theta_center, K, scan_value)
    if K == 1
        theta = scan_value;
    elseif K == 2
        theta = [theta_center - scan_value / 2, theta_center + scan_value / 2];
    else
        theta = [theta_center - scan_value, theta_center, theta_center + scan_value];
    end
end

function grp = make_group(K, scan_value)
    if K == 1
        if scan_value < 0
            prefix = 'min';
        else
            prefix = '';
        end
        tag = strrep(sprintf('%.1f', abs(scan_value)), '.', 'p');
        grp = ['/Theta_' prefix tag 'deg'];
    else
        tag = strrep(sprintf('%.1f', scan_value), '.', 'p');
        grp = ['/Delta_' tag 'deg'];
    end
end

function S = generate_sources(K, T, rho)
    rho = min(max(real(rho), 0), 0.999999);
    if K == 1
        S = (randn(K, T) + 1j * randn(K, T)) / sqrt(2);
        return;
    end
    common = (randn(1, T) + 1j * randn(1, T)) / sqrt(2);
    independent = (randn(K, T) + 1j * randn(K, T)) / sqrt(2);
    S = sqrt(rho) * repmat(common, K, 1) + sqrt(max(0, 1 - rho)) * independent;
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
