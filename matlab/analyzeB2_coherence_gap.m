% analyzeB2_coherence_gap.m
% 计算相干源条件下的谱间隙和子空间偏差，保存结果，不绘图

clear; close all; clc;

%% 路径
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(fileparts(script_dir), 'data', 'B2');
if ~exist(data_dir, 'dir'); data_dir = fullfile(script_dir, 'data', 'B2'); end
fname = fullfile(data_dir, 'B2_CoherenceGap.h5');
param_file = fullfile(data_dir, 'B2_CoherenceGap_params.mat');
if ~exist(fname, 'file'); error('数据文件不存在，请先运行 genB2_coherence_gap.m'); end
if ~exist(param_file, 'file'); error('参数文件不存在'); end
load(param_file);

%% 理想噪声子空间投影矩阵
steer_vec = @(theta, N) exp(1j * 2*pi*d * sin(deg2rad(theta)) * (0:N-1).');
A_ideal = zeros(M, K);
for k = 1:K; A_ideal(:, k) = steer_vec(theta_true(k), M); end
P_n_ideal = eye(M) - A_ideal * ((A_ideal' * A_ideal) \ A_ideal');
P_n_ideal = (P_n_ideal + P_n_ideal') / 2;

%% 预分配
n_rho = length(rho_s_vec);
gamma_norm_theory = zeros(1, n_rho);
gamma_norm_sample_mean = zeros(1, n_rho);
gamma_norm_sample_std = zeros(1, n_rho);
mean_dn = zeros(1, n_rho);
std_dn = zeros(1, n_rho);
all_gamma_sample = cell(1, n_rho);
all_dn = cell(1, n_rho);

fprintf('\n========== B2 分析开始 ==========\n');

%% 逐 rho_s 分析，带进度条
wb = waitbar(0, '分析中...', 'Name', 'B2 分析');
for ii = 1:n_rho
    rho_s = rho_s_vec(ii);
    rho_tag = make_rho_tag(rho_s);
    grp = ['/RHO/' rho_tag];
    
    % 理论协方差
    r_theory = h5read(fname, [grp '/theory']);
    R_theory = r_theory(:, :, 1) + 1j * r_theory(:, :, 2);
    R_theory = (R_theory + R_theory') / 2;
    eig_theory = sort(real(eig(R_theory)), 'descend');
    gamma_theory = eig_theory(K) - eig_theory(K+1);
    gamma_norm_theory(ii) = gamma_theory / max(eig_theory(1), eps);
    
    % 样本数据
    r_sam = h5read(fname, [grp '/sam']);
    R_sam_all = squeeze(r_sam(:, :, 1, :) + 1j * r_sam(:, :, 2, :));
    if ndims(R_sam_all) == 2; R_sam_all = reshape(R_sam_all, M, M, 1); end
    N_cur = size(R_sam_all, 3);
    
    gamma_mc = zeros(N_cur, 1);
    dn_mc = zeros(N_cur, 1);
    for mc = 1:N_cur
        R_hat = R_sam_all(:, :, mc);
        R_hat = (R_hat + R_hat') / 2;
        [V_est, D_est] = eig(R_hat);
        eig_vals = real(diag(D_est));
        [eig_sorted, eig_idx] = sort(eig_vals, 'descend');
        gamma_mc(mc) = (eig_sorted(K) - eig_sorted(K+1)) / max(eig_sorted(1), eps);
        U_n_est = V_est(:, eig_idx(K+1:end));
        P_n_est = U_n_est * U_n_est';
        P_n_est = (P_n_est + P_n_est') / 2;
        dn_mc(mc) = norm(P_n_ideal - P_n_est, 'fro') / sqrt(2);
    end
    
    gamma_norm_sample_mean(ii) = mean(gamma_mc, 'omitnan');
    gamma_norm_sample_std(ii) = std(gamma_mc, 'omitnan');
    mean_dn(ii) = mean(dn_mc, 'omitnan');
    std_dn(ii) = std(dn_mc, 'omitnan');
    all_gamma_sample{ii} = gamma_mc;
    all_dn{ii} = dn_mc;
    
    waitbar(ii/n_rho, wb, sprintf('rho_s = %.4f (%.1f%%)', rho_s, ii/n_rho*100));
    drawnow;
end
close(wb);

%% 保存结果
save(fullfile(data_dir, 'B2_CoherenceGap_results.mat'), ...
    'rho_s_vec', 'gamma_norm_theory', ...
    'gamma_norm_sample_mean', 'gamma_norm_sample_std', ...
    'mean_dn', 'std_dn', 'all_gamma_sample', 'all_dn', ...
    'theta_true', 'SNR', 'T', 'N_mc');
fprintf('\n结果已保存至：%s\n', fullfile(data_dir, 'B2_CoherenceGap_results.mat'));

%% 局部函数
function tag = make_rho_tag(rho_val)
    tag = sprintf('rho_%.3f', rho_val);
    tag = strrep(tag, '.', 'p');
end
