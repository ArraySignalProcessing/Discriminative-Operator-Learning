% generateB2_coherence_gap_data.m
% 生成相干源数据，添加图形化进度条

clear; close all; clc;
tic;
rng(14);

%% 路径
script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(fileparts(script_dir), 'data', 'B2');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
fname_main = fullfile(out_dir, 'B2_CoherenceGap.h5');
if exist(fname_main, 'file'); delete(fname_main); end

%% 实验参数
M = 10; K = 2; theta_true = [-10, 10]; d = 0.5;
SNR = 10; T = 500; N_mc = 500;
signal_power = 1; noise_power = signal_power * 10^(-SNR/10);
rho_s_vec = [0:0.05:0.8, 0.9:0.03:0.99, 0.991:0.001:1.0];

%% 理想 ULA 流形
steer_vec = @(theta, N) exp(1j * 2*pi*d * sin(deg2rad(theta)) * (0:N-1).');
A = zeros(M, K);
for k = 1:K; A(:, k) = steer_vec(theta_true(k), M); end
true_angles = repmat(theta_true(:), 1, N_mc);

fprintf('\n========== B2 数据生成开始 ==========\n');

%% 逐 rho_s 生成数据，带进度条
total = length(rho_s_vec);
wb = waitbar(0, '生成数据中...', 'Name', 'B2 数据生成');
for ii = 1:total
    rho_s = rho_s_vec(ii);
    rho_tag = make_rho_tag(rho_s);
    grp = ['/RHO/' rho_tag];
    
    % 源协方差矩阵
    Rs = signal_power * [1, rho_s; rho_s, 1];
    R_theory = A * Rs * A' + noise_power * eye(M);
    R_theory = (R_theory + R_theory') / 2;
    r_theory = zeros(M, M, 3);
    r_theory(:, :, 1) = real(R_theory);
    r_theory(:, :, 2) = imag(R_theory);
    r_theory(:, :, 3) = angle(R_theory);
    
    % 构造相关源平方根矩阵
    [U_rs, D_rs] = eig(Rs);
    eig_rs = real(diag(D_rs)); eig_rs(eig_rs < 0) = 0;
    Rs_sqrt = U_rs * diag(sqrt(eig_rs)) * U_rs';
    
    r_sam = zeros(M, M, 3, N_mc);
    for mc = 1:N_mc
        W = (randn(K, T) + 1j*randn(K, T)) / sqrt(2);
        S = Rs_sqrt * W;
        X = A * S;
        Noise = sqrt(noise_power) * (randn(M, T) + 1j*randn(M, T)) / sqrt(2);
        Y = X + Noise;
        R_sam = (Y * Y') / T; R_sam = (R_sam + R_sam') / 2;
        r_sam(:, :, 1, mc) = real(R_sam);
        r_sam(:, :, 2, mc) = imag(R_sam);
        r_sam(:, :, 3, mc) = angle(R_sam);
    end
    
    % 写入 HDF5
    h5create(fname_main, [grp '/sam'], size(r_sam));
    h5write(fname_main, [grp '/sam'], r_sam);
    h5create(fname_main, [grp '/theory'], size(r_theory));
    h5write(fname_main, [grp '/theory'], r_theory);
    h5create(fname_main, [grp '/angles'], size(true_angles));
    h5write(fname_main, [grp '/angles'], true_angles);
    
    waitbar(ii/total, wb, sprintf('rho_s = %.4f (%.1f%%)', rho_s, ii/total*100));
    drawnow;
end
close(wb);

%% 保存参数
param_file = fullfile(out_dir, 'B2_CoherenceGap_params.mat');
save(param_file, 'M','K','theta_true','d','SNR','T','N_mc','signal_power','noise_power','rho_s_vec');
fprintf('数据生成完成。耗时 %.2f 秒\n', toc);

%% 局部函数
function tag = make_rho_tag(rho_val)
    tag = sprintf('rho_%.3f', rho_val);
    tag = strrep(tag, '.', 'p');
end
