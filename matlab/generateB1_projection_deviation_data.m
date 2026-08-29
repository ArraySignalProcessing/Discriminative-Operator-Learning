% generateB1_projection_deviation_data.m
% 生成 SNR、快拍数、阵列误差扫描的样本协方差数据
% 添加 waitbar 图形化进度条

clear; close all; clc; rng(42);
tic;

%% 路径
script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(fileparts(script_dir), 'data', 'B1');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
fname_main = fullfile(out_dir, 'B1_ProjectionDeviation.h5');
if exist(fname_main, 'file'); delete(fname_main); end

%% 公共参数
M = 10;
K = 1;
theta_true = 0;
d = 0.5;
signal_power = 1;

%% 扫描参数
% SNR 扫描 (固定 T=1000, 无误差)
SNR_vec = -15:1:20;
T_for_SNR = 100;
N_mc_snr = 500;

% 快拍数 T 扫描 (固定 SNR=0 dB, 无误差)
T_vec = [2:1:8, 10:10:800, 1000:200:2000];
SNR_for_T = -5;
N_mc_T = 500;

% 阵列误差强度扫描 (固定 SNR=0 dB, T=500)
rho_err_vec = 0:0.03:0.33;
SNR_for_ERR = 5;
T_for_ERR = 300;
N_mc_err = 500;

%% 1. SNR 扫描
fprintf('\n--- 生成 SNR 扫描数据 ---\n');
total = length(SNR_vec);
wb = waitbar(0, 'SNR 扫描生成中...', 'Name', 'B1 数据生成');
for ii = 1:total
    snr_tag = make_snr_tag(SNR_vec(ii));
    grp = ['/SNR/SNR_' snr_tag];
    [r_sam, angles] = generate_cov_data(M, K, theta_true, T_for_SNR, SNR_vec(ii), ...
                                         N_mc_snr, signal_power, d, 0);
    write_h5_group(fname_main, grp, r_sam, angles);
    
    waitbar(ii/total, wb, sprintf('SNR 扫描: %d dB (%.1f%%)', SNR_vec(ii), ii/total*100));
    drawnow;
end
close(wb);

%% 2. 快拍数 T 扫描
fprintf('\n--- 生成快拍数 T 扫描数据 ---\n');
total = length(T_vec);
wb = waitbar(0, '快拍数扫描生成中...', 'Name', 'B1 数据生成');
for ii = 1:total
    grp = sprintf('/T/T_%d', T_vec(ii));
    [r_sam, angles] = generate_cov_data(M, K, theta_true, T_vec(ii), SNR_for_T, ...
                                         N_mc_T, signal_power, d, 0);
    write_h5_group(fname_main, grp, r_sam, angles);
    
    waitbar(ii/total, wb, sprintf('快拍数扫描: T=%d (%.1f%%)', T_vec(ii), ii/total*100));
    drawnow;
end
close(wb);

%% 3. 阵列误差扫描
fprintf('\n--- 生成阵列误差扫描数据 ---\n');
total = length(rho_err_vec);
wb = waitbar(0, '阵列误差扫描生成中...', 'Name', 'B1 数据生成');
for ii = 1:total
    rho_tag = make_rho_tag(rho_err_vec(ii));
    grp = ['/ERR/' rho_tag];
    [r_sam, angles] = generate_cov_data(M, K, theta_true, T_for_ERR, SNR_for_ERR, ...
                                         N_mc_err, signal_power, d, rho_err_vec(ii));
    write_h5_group(fname_main, grp, r_sam, angles);
    
    waitbar(ii/total, wb, sprintf('阵列误差扫描: ρ=%.3f (%.1f%%)', rho_err_vec(ii), ii/total*100));
    drawnow;
end
close(wb);

%% 保存参数
param_file = fullfile(out_dir, 'B1_ProjectionDeviation_params.mat');
save(param_file, ...
    'M','K','theta_true','d','signal_power',...
    'SNR_vec','T_for_SNR','N_mc_snr',...
    'T_vec','SNR_for_T','N_mc_T',...
    'rho_err_vec','SNR_for_ERR','T_for_ERR','N_mc_err');

fprintf('数据生成完成。耗时 %.2f 秒\n', toc);

%% ================= 局部函数 =================
function [r_sam, true_angles] = generate_cov_data(M, K, theta_true, T, SNR, N_mc, signal_power, d, rho_err)
    r_sam = zeros(M, M, 3, N_mc);
    true_angles = repmat(theta_true(:), 1, N_mc);
    noise_power = signal_power * 10^(-SNR/10);
    for mc = 1:N_mc
        A = build_array_manifold(M, K, theta_true, d, rho_err);
        S = sqrt(signal_power) * (randn(K, T) + 1j*randn(K, T)) / sqrt(2);
        X = A * S;
        Noise = sqrt(noise_power) * (randn(M, T) + 1j*randn(M, T)) / sqrt(2);
        Y = X + Noise;
        R_sam = (Y * Y') / T;
        R_sam = (R_sam + R_sam') / 2;
        r_sam(:, :, 1, mc) = real(R_sam);
        r_sam(:, :, 2, mc) = imag(R_sam);
        r_sam(:, :, 3, mc) = angle(R_sam);
    end
end

function A = build_array_manifold(M, K, theta_true, d, rho_err)
    pos_ideal = (0:M-1).' * d;
    if rho_err > 0
        pos_err = rho_err * randn(M, 1);
        gain_err = rho_err * randn(M, 1);
        phase_err = pi * rho_err * randn(M, 1);
        pos_actual = pos_ideal + pos_err;
        amp_phase = (1 + gain_err) .* exp(1j * phase_err);
    else
        pos_actual = pos_ideal;
        amp_phase = ones(M, 1);
    end
    A = zeros(M, K);
    for k = 1:K
        a = exp(1j * 2*pi * pos_actual * sin(deg2rad(theta_true(k))));
        A(:, k) = amp_phase .* a;
    end
end

function write_h5_group(fname, grp, r_sam, true_angles)
    h5create(fname, [grp '/sam'], size(r_sam));
    h5write(fname, [grp '/sam'], r_sam);
    h5create(fname, [grp '/angles'], size(true_angles));
    h5write(fname, [grp '/angles'], true_angles);
end

function tag = make_snr_tag(snr_val)
    if snr_val < 0
        tag = sprintf('min%ddB', abs(snr_val));
    else
        tag = sprintf('%ddB', snr_val);
    end
end

function tag = make_rho_tag(rho_val)
    tag = sprintf('rho_%.3f', rho_val);
    tag = strrep(tag, '.', 'p');
end
