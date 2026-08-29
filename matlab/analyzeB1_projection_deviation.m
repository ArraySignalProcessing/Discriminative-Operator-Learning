% analyzeB1_projection_deviation.m
% 分析 SNR 扫描、快拍数扫描、阵列误差扫描
% 使用 rootmusic 进行 DOA 估计（连续角度）
% 进度显示：waitbar 图形窗口

clear; close all; clc;

%% 路径设置
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(fileparts(script_dir), 'data', 'B1');
if ~exist(data_dir, 'dir')
    data_dir = fullfile(script_dir, 'data', 'B1');
end

fname = fullfile(data_dir, 'B1_ProjectionDeviation.h5');
param_file = fullfile(data_dir, 'B1_ProjectionDeviation_params.mat');

if ~exist(fname, 'file')
    error('找不到数据文件：%s，请先运行 genB1_projection_deviation.m', fname);
end
if ~exist(param_file, 'file')
    error('找不到参数文件：%s，请先运行 genB1_projection_deviation.m', param_file);
end
load(param_file);  % 加载 M, K, theta_true, d, SNR_vec, T_vec, rho_err_vec 等

eps0 = 1e-12;

%% 理想阵列下的噪声子空间投影矩阵
A_ideal = build_ideal_manifold(M, K, theta_true, d);
P_n_ideal = eye(M) - A_ideal * ((A_ideal' * A_ideal) \ A_ideal');
P_n_ideal = (P_n_ideal + P_n_ideal') / 2;

%% ================= 1. SNR 扫描分析 =================
fprintf('\n--- 分析 SNR 扫描 ---\n');
mean_dn_snr = zeros(size(SNR_vec));
std_dn_snr = zeros(size(SNR_vec));
mean_rmse_snr = zeros(size(SNR_vec));
std_rmse_snr = zeros(size(SNR_vec));
median_rmse_snr = zeros(size(SNR_vec));
all_dn_snr = cell(size(SNR_vec));
all_rmse_snr = cell(size(SNR_vec));

total = length(SNR_vec);
wb = waitbar(0, 'SNR 扫描进度', 'Name', 'B1 分析');
for ii = 1:total
    snr_val = SNR_vec(ii);
    snr_tag = make_snr_tag(snr_val);
    grp = ['/SNR/SNR_' snr_tag];
    [dn_mc, rmse_mc] = analyze_one_group(fname, grp, M, K, P_n_ideal, theta_true, d);
    mean_dn_snr(ii) = mean(dn_mc, 'omitnan');
    std_dn_snr(ii) = std(dn_mc, 'omitnan');
    mean_rmse_snr(ii) = mean(rmse_mc, 'omitnan');
    std_rmse_snr(ii) = std(rmse_mc, 'omitnan');
    median_rmse_snr(ii) = median(rmse_mc, 'omitnan');
    all_dn_snr{ii} = dn_mc;
    all_rmse_snr{ii} = rmse_mc;
    
    waitbar(ii/total, wb, sprintf('SNR 扫描: %d dB (%.1f%%)', snr_val, ii/total*100));
    drawnow;
end
close(wb);

%% ================= 2. 快拍数 T 扫描分析 =================
fprintf('\n--- 分析快拍数 T 扫描 ---\n');
mean_dn_T = zeros(size(T_vec));
std_dn_T = zeros(size(T_vec));
mean_rmse_T = zeros(size(T_vec));
std_rmse_T = zeros(size(T_vec));
median_rmse_T = zeros(size(T_vec));
all_dn_T = cell(size(T_vec));
all_rmse_T = cell(size(T_vec));

total = length(T_vec);
wb = waitbar(0, '快拍数扫描进度', 'Name', 'B1 分析');
for ii = 1:total
    T_val = T_vec(ii);
    grp = sprintf('/T/T_%d', T_val);
    [dn_mc, rmse_mc] = analyze_one_group(fname, grp, M, K, P_n_ideal, theta_true, d);
    mean_dn_T(ii) = mean(dn_mc, 'omitnan');
    std_dn_T(ii) = std(dn_mc, 'omitnan');
    mean_rmse_T(ii) = mean(rmse_mc, 'omitnan');
    std_rmse_T(ii) = std(rmse_mc, 'omitnan');
    median_rmse_T(ii) = median(rmse_mc, 'omitnan');
    all_dn_T{ii} = dn_mc;
    all_rmse_T{ii} = rmse_mc;
    
    waitbar(ii/total, wb, sprintf('快拍数扫描: T=%d (%.1f%%)', T_val, ii/total*100));
    drawnow;
end
close(wb);

%% ================= 3. 阵列误差扫描分析 =================
fprintf('\n--- 分析阵列误差扫描 ---\n');
mean_dn_err = zeros(size(rho_err_vec));
std_dn_err = zeros(size(rho_err_vec));
mean_rmse_err = zeros(size(rho_err_vec));
std_rmse_err = zeros(size(rho_err_vec));
median_rmse_err = zeros(size(rho_err_vec));
all_dn_err = cell(size(rho_err_vec));
all_rmse_err = cell(size(rho_err_vec));

total = length(rho_err_vec);
wb = waitbar(0, '阵列误差扫描进度', 'Name', 'B1 分析');
for ii = 1:total
    rho_val = rho_err_vec(ii);
    rho_tag = make_rho_tag(rho_val);
    grp = ['/ERR/' rho_tag];
    [dn_mc, rmse_mc] = analyze_one_group(fname, grp, M, K, P_n_ideal, theta_true, d);
    mean_dn_err(ii) = mean(dn_mc, 'omitnan');
    std_dn_err(ii) = std(dn_mc, 'omitnan');
    mean_rmse_err(ii) = mean(rmse_mc, 'omitnan');
    std_rmse_err(ii) = std(rmse_mc, 'omitnan');
    median_rmse_err(ii) = median(rmse_mc, 'omitnan');
    all_dn_err{ii} = dn_mc;
    all_rmse_err{ii} = rmse_mc;
    
    waitbar(ii/total, wb, sprintf('阵列误差扫描: ρ=%.3f (%.1f%%)', rho_val, ii/total*100));
    drawnow;
end
close(wb);

%% 保存结果
save(fullfile(data_dir, 'B1_ProjectionDeviation_results.mat'), ...
    'SNR_vec', 'T_vec', 'rho_err_vec', ...
    'mean_dn_snr', 'std_dn_snr', 'mean_rmse_snr', 'std_rmse_snr', 'median_rmse_snr', ...
    'mean_dn_T', 'std_dn_T', 'mean_rmse_T', 'std_rmse_T', 'median_rmse_T', ...
    'mean_dn_err', 'std_dn_err', 'mean_rmse_err', 'std_rmse_err', 'median_rmse_err', ...
    'all_dn_snr', 'all_rmse_snr', ...
    'all_dn_T', 'all_rmse_T', ...
    'all_dn_err', 'all_rmse_err');

fprintf('\n结果已保存至：%s\n', fullfile(data_dir, 'B1_ProjectionDeviation_results.mat'));

%% 打印表格
fprintf('\n========== 统计结果汇总 ==========\n');
fprintf('\n--- 表1: SNR 扫描 ---\n');
fprintf('SNR (dB)\tMean dn\t\tStd dn\t\tMean RMSE\tStd RMSE\tMed RMSE\n');
for i = 1:length(SNR_vec)
    fprintf('%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        SNR_vec(i), mean_dn_snr(i), std_dn_snr(i), ...
        mean_rmse_snr(i), std_rmse_snr(i), median_rmse_snr(i));
end

fprintf('\n--- 表2: 快拍数 T 扫描 ---\n');
fprintf('T\t\tMean dn\t\tStd dn\t\tMean RMSE\tStd RMSE\tMed RMSE\n');
for i = 1:length(T_vec)
    fprintf('%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        T_vec(i), mean_dn_T(i), std_dn_T(i), ...
        mean_rmse_T(i), std_rmse_T(i), median_rmse_T(i));
end

fprintf('\n--- 表3: 阵列误差强度扫描 ---\n');
fprintf('rho\t\tMean dn\t\tStd dn\t\tMean RMSE\tStd RMSE\tMed RMSE\n');
for i = 1:length(rho_err_vec)
    fprintf('%.3f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        rho_err_vec(i), mean_dn_err(i), std_dn_err(i), ...
        mean_rmse_err(i), std_rmse_err(i), median_rmse_err(i));
end

fprintf('\nB1 分析完成。\n');

%% ======================= 局部函数 =======================
function [dn_mc, rmse_mc] = analyze_one_group(fname, grp, M, K, P_n_ideal, theta_true, d)
    r_sam = h5read(fname, [grp '/sam']);
    true_angles = h5read(fname, [grp '/angles']);
    R_sam_all = squeeze(r_sam(:, :, 1, :) + 1j * r_sam(:, :, 2, :));
    if ndims(R_sam_all) == 2
        R_sam_all = reshape(R_sam_all, M, M, 1);
    end
    N_mc = size(R_sam_all, 3);
    dn_mc = zeros(N_mc, 1);
    rmse_mc = zeros(N_mc, 1);
    
    for mc = 1:N_mc
        R_hat = R_sam_all(:, :, mc);
        R_hat = (R_hat + R_hat') / 2;
        [V_est, D_est] = eig(R_hat);
        [~, eig_idx] = sort(real(diag(D_est)), 'descend');
        U_n_est = V_est(:, eig_idx(K+1:end));
        P_n_est = U_n_est * U_n_est';
        P_n_est = (P_n_est + P_n_est') / 2;
        dn_mc(mc) = norm(P_n_ideal - P_n_est, 'fro') / sqrt(2);
        
        try
            doa_est = rootmusic(R_hat, K);
            doa_est = sort(doa_est(:));
            doa_est = doa_est(doa_est >= -90 & doa_est <= 90);
            if length(doa_est) < K
                doa_est = NaN(K, 1);
            end
        catch
            doa_est = NaN(K, 1);
        end
        
        gt = true_angles(:, mc);
        rmse_mc(mc) = calc_rmse(doa_est, gt, K);
    end
end

function rmse_val = calc_rmse(doa_est, gt, K)
    if any(isnan(doa_est)) || length(doa_est) ~= K
        rmse_val = NaN; return;
    end
    doa_est = sort(doa_est(:));
    gt = sort(gt(:));
    if K == 2
        err1 = sum((doa_est - gt).^2);
        err2 = sum((flipud(doa_est) - gt).^2);
        rmse_val = sqrt(min(err1, err2) / K);
    else
        rmse_val = sqrt(sum((doa_est - gt).^2) / K);
    end
end

function A = build_ideal_manifold(M, K, theta_true, d)
    pos = (0:M-1).' * d;
    A = zeros(M, K);
    for k = 1:K
        A(:, k) = exp(1j * 2*pi * pos * sin(deg2rad(theta_true(k))));
    end
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
