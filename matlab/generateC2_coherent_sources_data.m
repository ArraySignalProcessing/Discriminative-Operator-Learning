% generateC2_coherent_sources_data.m
% C2 coherent-source data generator.
% Scenario 1D: coherent-source robustness with a moderately close off-grid pair.
% Fixed setting: DOA=[-5.2, 13.8] deg, SNR=-5 dB, T=100, M=10.
% 论文实验 Topic 1D —— 相干源（相关系数 ρ 扫描），固定 SNR
% Fixed two-source DOA with coherence-rho scan.
% 生成数据：2通道样本/理论协方差（实部+虚部）。
% 可选择同时运行 l1-SVD 和 UnESPRIT 算法。
% 所有可调参数集中于脚本开头。
% 作者: D
% 日期: 2026-05-08 (revised 2026-05-09)

clear all; close all;
tic;
rng(42);

% ================= 路径设置 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C2');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

% ================= 可调参数区 =================
% --- 阵列与信号 ---
M            = 10;                  % 阵元数
K            = 2;                   % 信源数
theta_true   = [-5.2, 13.8];        % off-grid DOA with moderate angular separation
T            = 200;                % snapshot count
signal_power = 1;                   % 信号功率

% --- 固定 SNR ---
SNR_dB       = 0;                % fixed SNR (dB)

% --- 相干源相关系数扫描（0 = 独立, 1 = 完全相干）---
rho_vec      =  [ 0.3, 0.6, 0.9, 0.95:0.01:1.0];  % source coherence scan

% --- 蒙特卡洛 ---
N_mc         = 1000;                % 每个 ρ 下的独立试验次数


% --- 搜索网格 ---
angle_scan   = -60:0.5:60;          % MUSIC / l1-SVD 搜索网格

% --- 算法开关 ---
compute_l1SVD_UnESPRIT = false;      % 是否运行 l1-SVD 与 UnESPRIT

% --- L1-SVD 自适应阈值参数 ---
% 阈值公式: threshold = sigma_n * sqrt(2*M*T) * th_factor
th_factor_l1svd = 3.3;              % 阈值安全因子，可调

% --- UnESPRIT 参数 ---
ds = 1;                             % 子阵列抽取因子
ms = 8;                             % 加权子阵中心阶数
w  = min(ms, M - ds - ms + 1);      % 子阵列个数

% ================= 导向矢量 =================
steer_vec = @(theta, N) exp(1j*pi*sin(deg2rad(theta))*(0:N-1).');

% 噪声功率与自适应阈值（固定 SNR）
noise_power = signal_power * 10^(-SNR_dB/10);
sigma_n = sqrt(noise_power);
threshold_l1SVD = sigma_n * sqrt(2 * M * T) * th_factor_l1svd;

% 阵列流形
A = zeros(M, K);
for k = 1:K
    A(:,k) = steer_vec(theta_true(k), M);
end

% ================= 主循环 =================
fname_main = fullfile(data_dir, 'C2_CoherentSources.h5');
if exist(fname_main, 'file')
    delete(fname_main);
end
tau_max = 4;

for rho_idx = 1:length(rho_vec)
    rho_coherence = rho_vec(rho_idx);
    if rho_coherence < 1
        rho_tag = strrep(sprintf('%.2f', rho_coherence), '.', 'p');
    else
        rho_tag = '1p00';
    end

    fprintf('处理 ρ = %.2f, SNR = %d dB ...\n', rho_coherence, SNR_dB);

    % 相关源矩阵
    R_signal = [1, rho_coherence; rho_coherence, 1];

    % 理论协方差

    % 预分配（2 通道：实部+虚部）
    r_sam = zeros(M, M, 2, N_mc);
    tau_corr = zeros(N_mc, tau_max + 1, 2 * M, M, 'single');
    true_angles = repmat(theta_true(:), 1, N_mc);

    if compute_l1SVD_UnESPRIT
        l1SVD_est    = zeros(K, N_mc);
        UnESPRIT_est = zeros(K, N_mc);
    end

    % 根据 ρ 选择信号生成方式
    if rho_coherence == 1
        % 完全相干：两个信号完全相同
        for mc = 1:N_mc
            s1 = (randn(1,T) + 1j*randn(1,T)) / sqrt(2);
            S  = [s1; s1];

            X = A * S;
            Noise = sigma_n * (randn(M, T) + 1j*randn(M, T)) / sqrt(2);
            Y = X + Noise;

            Ry_sam = (Y * Y') / T;
            tau_corr(mc, :, :, :) = build_tau_corr(Y, tau_max);

            r_sam(:,:,1,mc) = real(Ry_sam);
            r_sam(:,:,2,mc) = imag(Ry_sam);

            if compute_l1SVD_UnESPRIT
                [ang_l1, ~] = l1_SVD_DoA_est(Y, M, threshold_l1SVD, K, angle_scan);
                l1SVD_est(:, mc) = sort(ang_l1)';
                doa_uesp = unit_ESPRIT(Y, T, ds, K, w);
                UnESPRIT_est(:, mc) = sort(doa_uesp);
            end
        end
    else
        % 部分相关：Cholesky 分解 R_signal = L*L'
        L = chol(R_signal, 'lower');
        for mc = 1:N_mc
            s_uncorr = (randn(K, T) + 1j*randn(K, T)) / sqrt(2);
            S = L * s_uncorr;

            X = A * S;
            Noise = sigma_n * (randn(M, T) + 1j*randn(M, T)) / sqrt(2);
            Y = X + Noise;

            Ry_sam = (Y * Y') / T;
            tau_corr(mc, :, :, :) = build_tau_corr(Y, tau_max);

            r_sam(:,:,1,mc) = real(Ry_sam);
            r_sam(:,:,2,mc) = imag(Ry_sam);

            if compute_l1SVD_UnESPRIT
                [ang_l1, ~] = l1_SVD_DoA_est(Y, M, threshold_l1SVD, K, angle_scan);
                l1SVD_est(:, mc) = sort(ang_l1)';
                doa_uesp = unit_ESPRIT(Y, T, ds, K, w);
                UnESPRIT_est(:, mc) = sort(doa_uesp);
            end
        end
    end

    % 保存到 HDF5
    grp = sprintf('/Rho_%s', rho_tag);
    h5create(fname_main, [grp '/sam'], size(r_sam));
    h5write(fname_main, [grp '/sam'], r_sam);
    h5create(fname_main, [grp '/tau_corr'], size(tau_corr));
    h5write(fname_main, [grp '/tau_corr'], tau_corr);
    h5create(fname_main, [grp '/angles'], size(true_angles));
    h5write(fname_main, [grp '/angles'], true_angles);

    % 保存 l1-SVD / UnESPRIT 结果
    if compute_l1SVD_UnESPRIT
        fname_l1 = fullfile(data_dir, sprintf('C2_l1SVD_M%d_K%d_Rho%s_T%d.h5', M, K, rho_tag, T));
        if exist(fname_l1, 'file'), delete(fname_l1); end
        h5create(fname_l1, '/l1_SVD_ang', size(l1SVD_est));
        h5write(fname_l1, '/l1_SVD_ang', l1SVD_est);

        fname_ues = fullfile(data_dir, sprintf('C2_UnESPRIT_M%d_K%d_Rho%s_T%d.h5', M, K, rho_tag, T));
        if exist(fname_ues, 'file'), delete(fname_ues); end
        h5create(fname_ues, '/UnESPRIT_ang', size(UnESPRIT_est));
        h5write(fname_ues, '/UnESPRIT_ang', UnESPRIT_est);
    end

    fprintf('ρ = %.2f 完成 (%.2f min)\n', rho_coherence, toc/60);
end

time_total = toc / 60;
fprintf('相干源数据生成完毕 (SNR=%d dB)，总耗时 %.2f 分钟。\n', SNR_dB, time_total);
if compute_l1SVD_UnESPRIT
    fprintf('l1-SVD 和 UnESPRIT 结果已保存至独立 HDF5 文件。\n');
else
    fprintf('l1-SVD 和 UnESPRIT 计算已跳过。\n');
end
