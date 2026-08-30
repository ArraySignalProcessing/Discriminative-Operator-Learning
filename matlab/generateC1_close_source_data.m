% generateC1_close_source_data.m
% C1 close-source data generator.
% 论文实验 C1 —— 角度间隔扫描（RMSE vs Δθ）
% 生成数据：2通道样本/理论协方差（实部+虚部）
% 可选择同时运行 l1-SVD 和 UnESPRIT
% 所有可调参数集中于脚本开头。
% 作者: D
% 日期: 2026-05-08 (revised 2026-05-09)

clear all; close all;
tic;
rng(42);

% ================= 路径设置 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C1');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

% ================= 可调参数区 =================
% --- 阵列与信号 ---
M            = 10;                  % 阵元数
K            = 2;                   % 信源数
T            = 300;                % snapshot count for close-source resolution
SNR_fixed    = -5;               % fixed SNR (dB)
signal_power = 1;                   % 信号功率

% --- 角度间隔扫描 ---
delta_theta_vec =[2:1:5,6:3:15];  % angular separations (deg)

% --- 蒙特卡洛 ---
N_mc         = 2000;                % 每个间隔下的独立试验次数


% --- 搜索网格 ---
angle_scan   = -60:0.5:60;         % MUSIC / l1-SVD 搜索网格

% --- 算法开关 ---
compute_l1SVD_UnESPRIT = false;     % 是否运行 l1-SVD 与 UnESPRIT

% --- L1-SVD 自适应阈值参数 ---
% 阈值公式: threshold = sigma_n * sqrt(2*M*T) * th_factor
th_factor_l1svd = 3.3;             % 阈值安全因子，可调

% --- UnESPRIT 参数 ---
ds = 1;                             % 子阵列抽取因子
ms = 8;                             % 加权子阵中心阶数
w  = min(ms, M - ds - ms + 1);      % 子阵列个数

% ================= 导向矢量 =================
steer_vec = @(theta, N) exp(1j*pi*sin(deg2rad(theta))*(0:N-1).');

% 噪声功率（固定 SNR）
noise_power = signal_power * 10^(-SNR_fixed/10);
sigma_n = sqrt(noise_power);

% 自适应 L1-SVD 阈值（使用 T 代替 K）
threshold_l1SVD = sigma_n * sqrt(2 * M * T) * th_factor_l1svd;
fprintf('L1-SVD 阈值 = %.1f\n', threshold_l1SVD);

% ================= 主循环 =================
fname_main = fullfile(data_dir, 'C1_CloseSource.h5');
if exist(fname_main, 'file')
    delete(fname_main);
end
tau_max = 4;

theta_center = 1.7;             % off-grid center angle

for sep_idx = 1:length(delta_theta_vec)
    Delta = delta_theta_vec(sep_idx);
    theta_true = [theta_center - Delta / 2, theta_center + Delta / 2];
    sep_tag = sprintf('%ddeg', Delta);
    fprintf('处理 θ1=%.1f°, θ2=%.1f° ...\n', theta_true(1), theta_true(2));

    % 阵列流形
    A = zeros(M, K);
    for k = 1:K
        A(:,k) = steer_vec(theta_true(k), M);
    end

    % 预分配（2 通道：实部 + 虚部）
    r_sam = zeros(M, M, 2, N_mc);
    tau_corr = zeros(N_mc, tau_max + 1, 2 * M, M, 'single');
    true_angles = repmat(theta_true(:), 1, N_mc);

    if compute_l1SVD_UnESPRIT
        l1SVD_est  = zeros(K, N_mc);
        UnESPRIT_est = zeros(K, N_mc);
    end

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
    grp = sprintf('/Sep_%s', sep_tag);
    h5create(fname_main, [grp '/sam'], size(r_sam));
    h5write(fname_main, [grp '/sam'], r_sam);
    h5create(fname_main, [grp '/tau_corr'], size(tau_corr));
    h5write(fname_main, [grp '/tau_corr'], tau_corr);
    h5create(fname_main, [grp '/angles'], size(true_angles));
    h5write(fname_main, [grp '/angles'], true_angles);

    % 保存 l1-SVD / UnESPRIT 结果（独立 HDF5）
    if compute_l1SVD_UnESPRIT
        fname_l1 = fullfile(data_dir, sprintf('C1_l1SVD_M%d_K%d_Sep%s_T%d.h5', M, K, sep_tag, T));
        if exist(fname_l1, 'file'), delete(fname_l1); end
        h5create(fname_l1, '/l1_SVD_ang', size(l1SVD_est));
        h5write(fname_l1, '/l1_SVD_ang', l1SVD_est);

        fname_ues = fullfile(data_dir, sprintf('C1_UnESPRIT_M%d_K%d_Sep%s_T%d.h5', M, K, sep_tag, T));
        if exist(fname_ues, 'file'), delete(fname_ues); end
        h5create(fname_ues, '/UnESPRIT_ang', size(UnESPRIT_est));
        h5write(fname_ues, '/UnESPRIT_ang', UnESPRIT_est);
    end

    fprintf('θ1=%.1f°, θ2=%.1f° 完成 (%.2f min)\n', theta_true(1), theta_true(2), toc/60);
end

time_total = toc / 60;
fprintf('全部数据生成完毕，总耗时 %.2f 分钟。\n', time_total);
if compute_l1SVD_UnESPRIT
    fprintf('l1-SVD 和 UnESPRIT 结果已保存。\n');
else
    fprintf('l1-SVD 和 UnESPRIT 计算已跳过。\n');
end
