% generateC4_snr_scan_data.m
% C4 SNR-scan data generator.
% Scenario 1A: nominal independent-source SNR scan for basic noise robustness.
% Fixed setting: DOA=[-9.2, 13.8] deg, T=200, M=10, grid=0.5 deg.
% 论文实验 Topic 1A —— SNR 扫描（RMSE vs SNR）
% 生成数据：2通道样本/理论协方差（实部+虚部）。
% 可选择同时运行 l1-SVD 和 UnESPRIT 算法。
% 所有可调参数集中于脚本开头。
% 作者: D
% 日期: 2026-05-08 (revised 2026-05-09)

clear all; close all;
tic;
rng(33);

% ================= 路径设置 =================

data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C4');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

% ================= 可调参数区 =================
% --- 阵列与信号 ---
M            = 10;                  % 阵元数
K            = 2;                   % 信源数
theta_true   = [-2.2, 3.3];       % off-grid DOA with moderate separation
T            = 100;                % snapshot count for basic SNR scan
signal_power = 1;                   % 信号功率（假设归一化）

% --- SNR 扫描 ---
SNR_vec      = -20:2:10;           % SNR scan within/near the training range (dB)

% --- 蒙特卡洛 ---
N_mc         = 2000;                % 每个 SNR 下的独立试验次数


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
w  = min(ms, M - ds - ms + 1);      % 子阵列个数（根据公式计算，一般不需改动）

% ================= 导向矢量 =================
steer_vec = @(theta, N) exp(1j*pi*sin(deg2rad(theta))*(0:N-1).');

% ================= 主循环 =================
fname_main = fullfile(data_dir, 'C4_SnrScan.h5');
if exist(fname_main, 'file')
    delete(fname_main);
end
tau_max = 4;

for snr_idx = 1:length(SNR_vec)
    snr_val = SNR_vec(snr_idx);
    % 构建 SNR 标签（无 iif）
    if snr_val < 0
        snr_tag = sprintf('min%ddB', abs(snr_val));
    else
        snr_tag = sprintf('%ddB', snr_val);
    end
    fprintf('处理 SNR = %d dB ...\n', snr_val);
    
    % 噪声功率与自适应阈值
    noise_power = signal_power * 10^(-snr_val/10);
    sigma_n = sqrt(noise_power);
    threshold_l1SVD = sigma_n * sqrt(2 * M * T) * th_factor_l1svd;   
    
    % 预分配数组（2 通道：实部 + 虚部）
    r_sam = zeros(M, M, 2, N_mc);
    tau_corr = zeros(N_mc, tau_max + 1, 2 * M, M, 'single');
    true_angles = repmat(theta_true(:), 1, N_mc);
    
    if compute_l1SVD_UnESPRIT
        l1SVD_est  = zeros(K, N_mc);
        UnESPRIT_est = zeros(K, N_mc);
    end
    
    % 阵列流形矩阵
    A = zeros(M, K);
    for k = 1:K
        A(:,k) = steer_vec(theta_true(k), M);
    end
    
    % 理论协方差矩阵
    
    % 蒙特卡洛循环
    for mc = 1:N_mc
        S = (randn(K, T) + 1j*randn(K, T)) / sqrt(2);
        X = A * S;
        Noise = sigma_n * (randn(M, T) + 1j*randn(M, T)) / sqrt(2);
        Y = X + Noise;
        
        Ry_sam = (Y * Y') / T;
        tau_corr(mc, :, :, :) = build_tau_corr(Y, tau_max);
        
        % 保存 2 通道（实部 + 虚部）
        r_sam(:,:,1,mc) = real(Ry_sam);
        r_sam(:,:,2,mc) = imag(Ry_sam);
        
        if compute_l1SVD_UnESPRIT
            [ang_l1, ~] = l1_SVD_DoA_est(Y, M, threshold_l1SVD, K, angle_scan);
            l1SVD_est(:, mc) = sort(ang_l1)';
            
            doa_uesp = unit_ESPRIT(Y, T, ds, K, w);
            UnESPRIT_est(:, mc) = sort(doa_uesp);
        end
    end
    
    % 保存到 HDF5
    grp = sprintf('/SNR_%s', snr_tag);
    h5create(fname_main, [grp '/sam'], size(r_sam));
    h5write(fname_main, [grp '/sam'], r_sam);
    h5create(fname_main, [grp '/tau_corr'], size(tau_corr));
    h5write(fname_main, [grp '/tau_corr'], tau_corr);
    h5create(fname_main, [grp '/angles'], size(true_angles));
    h5write(fname_main, [grp '/angles'], true_angles);
    
    % 保存 l1-SVD / UnESPRIT 结果（独立 HDF5）
    if compute_l1SVD_UnESPRIT
        fname_l1 = fullfile(data_dir, sprintf('C4_l1SVD_M%d_K%d_%s_T%d.h5', M, K, snr_tag, T));
        if exist(fname_l1, 'file'), delete(fname_l1); end
        h5create(fname_l1, '/l1_SVD_ang', size(l1SVD_est));
        h5write(fname_l1, '/l1_SVD_ang', l1SVD_est);
        
        fname_ues = fullfile(data_dir, sprintf('C4_UnESPRIT_M%d_K%d_%s_T%d.h5', M, K, snr_tag, T));
        if exist(fname_ues, 'file'), delete(fname_ues); end
        h5create(fname_ues, '/UnESPRIT_ang', size(UnESPRIT_est));
        h5write(fname_ues, '/UnESPRIT_ang', UnESPRIT_est);
    end
    
    fprintf('SNR = %d dB 完成 (%.2f min)\n', snr_val, toc/60);
end

time_total = toc / 60;
fprintf('全部数据生成完毕，总耗时 %.2f 分钟。\n', time_total);
if compute_l1SVD_UnESPRIT
    fprintf('l1-SVD 和 UnESPRIT 结果已保存至独立 HDF5 文件。\n');
else
    fprintf('l1-SVD 和 UnESPRIT 计算已跳过。\n');
end
