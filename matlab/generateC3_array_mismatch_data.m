% generateC3_array_mismatch_data.m
% C3 array-mismatch data generator.
% Topic 1E：联合阵列误差测试数据生成。
%
% 本脚本扫描统一的阵列误差强度因子 rho。rho = 0 表示理想阵列；
% rho = 1 表示标称联合阵列误差强度：
%   - 阵元位置扰动标准差：0.05 lambda
%   - 增益误差标准差：10%
%   - 相位误差标准差：10 度
%   - 一阶互耦：|c1| = 0.2，相位 = 45 度
%   - 二阶互耦：|c2| = 0.1，相位 = 45 度
%
% 固定 SNR、DOA、快拍数、阵元数、信源数和蒙特卡洛次数在下方参数区设置。
% 输出 HDF5 文件保存样本协方差、多滞后相关特征和真实阵列误差参数，
% 供 MATLAB 基线、CNN、SubspaceNet、本文方法和 Oracle-MUSIC 上界统一评估使用。

clear all; close all;
tic;
rng(42);

% ================= 路径设置 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C3');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

% ================= 参数设置 =================
% --- 阵列与信号 ---
M            = 10;                  % array elements
K            = 2;                   % sources
T            = 2000;                 % snapshots
SNR_fixed    = -10;                   % fixed SNR (dB)
theta_true   = [-5.2, 8.3];         % off-grid DOA with moderate separation
signal_power = 1;                   % 信号功率

% --- 蒙特卡洛 ---
N_mc         = 1000;                % 每个 rho 下的独立试验次数

% --- 搜索网格 ---
angle_scan   = -60:0.5:60;          % MUSIC / l1-SVD 搜索网格

% --- 阵列误差强度因子 rho 扫描 ---
rho_all = 0:0.1:1;

% --- 标称误差强度（rho = 1）---
sigma_p_nom   = 0.05;            % 位置扰动标准差 (lambda)
sigma_g_nom   = 0.10;            % 增益误差标准差 (10%)
sigma_phi_nom = 10 * pi/180;     % 相位误差标准差 (10 度)
cpl_mag_nom   = 0.2;             % 一阶邻接互耦幅度
cpl_phase     = 45 * pi/180;     % 互耦相位 (45 度)
cpl_neighbors = 2;               % 互耦阶数：一阶和二阶邻接阵元

% --- 算法开关 ---
compute_l1SVD_UnESPRIT = false;  % 是否同时运行 l1-SVD 和 UnESPRIT

% --- l1-SVD 自适应阈值 ---
% threshold = sigma_n * sqrt(2*M*T) * th_factor_l1svd
th_factor_l1svd = 3.3;

% --- UnESPRIT 参数 ---
ds = 1;
ms = 8;
w  = min(ms, M - ds - ms + 1);

% ================= 噪声功率与阈值 =================
noise_power = signal_power * 10^(-SNR_fixed/10);
sigma_n = sqrt(noise_power);
threshold_l1SVD = sigma_n * sqrt(2 * M * T) * th_factor_l1svd;

% ================= 导向矢量与理想阵列 =================
d_ideal = 0.5;                                  % 理想半波长间距
steer_vec = @(theta, pos) exp(1j*2*pi*pos*sin(deg2rad(theta)));
pos_ideal = (0:M-1)' * d_ideal;
A_ideal = zeros(M, K);
for k = 1:K
    A_ideal(:,k) = steer_vec(theta_true(k), pos_ideal);
end

% ================= 主 HDF5 文件 =================
fname_main = fullfile(data_dir, 'C3_ArrayMismatch.h5');
if exist(fname_main, 'file')
    delete(fname_main);
end
tau_max = 4;

% ================= 主循环：扫描 rho =================
for r_idx = 1:length(rho_all)
    rho = rho_all(r_idx);
    fprintf('处理 rho = %.2f（联合阵列误差）...\n', rho);
    grp = sprintf('/Rho_%.2f', rho);

    % 根据 rho 缩放标称阵列误差强度。
    sigma_p   = rho * sigma_p_nom;
    sigma_g   = rho * sigma_g_nom;
    sigma_phi = rho * sigma_phi_nom;
    cpl_mag   = rho * cpl_mag_nom;

    % 生成随邻接距离衰减的互耦系数。
    cpl_vec = cpl_mag * exp(1j*cpl_phase) * (0.5).^(0:cpl_neighbors-1);

    % 预分配实部/虚部协方差通道和多滞后相关特征。
    r_sam = zeros(M, M, 2, N_mc);
    tau_corr = zeros(N_mc, tau_max + 1, 2 * M, M, 'single');
    true_angles = repmat(theta_true(:), 1, N_mc);
    oracle_pos = zeros(M, N_mc, 'single');
    oracle_resp_real = zeros(M, N_mc, 'single');
    oracle_resp_imag = zeros(M, N_mc, 'single');

    if compute_l1SVD_UnESPRIT
        l1SVD_est    = zeros(K, N_mc);
        UnESPRIT_est = zeros(K, N_mc);
    end

    for mc = 1:N_mc
        % 1. 阵元位置扰动。
        pert = sigma_p * randn(M, 1);
        pos_actual = pos_ideal + pert;
        A_pert = zeros(M, K);
        for k = 1:K
            A_pert(:,k) = steer_vec(theta_true(k), pos_actual);
        end

        % 2. 增益和相位误差。
        gain_err   = 1 + sigma_g * randn(M,1);
        phase_err  = exp(1j * sigma_phi * randn(M,1));
        sensor_resp = gain_err .* phase_err;
        Gamma      = diag(sensor_resp);

        % 3. 互耦矩阵。
        C = mutual_coupling_matrix(M, cpl_vec);

        % 实际阵列流形。
        A_actual = C * Gamma * A_pert;
        for k = 1:K
            A_actual(:, k) = sqrt(M) * A_actual(:, k) / norm(A_actual(:, k));
        end

        % 快拍信号生成。
        S = (randn(K, T) + 1j*randn(K, T)) / sqrt(2);
        X = A_actual * S;
        Noise = sigma_n * (randn(M, T) + 1j*randn(M, T)) / sqrt(2);
        Y = X + Noise;

        Ry_sam = (Y * Y') / T;
        tau_corr(mc, :, :, :) = build_tau_corr(Y, tau_max);

        r_sam(:,:,1,mc) = real(Ry_sam);
        r_sam(:,:,2,mc) = imag(Ry_sam);
        oracle_pos(:, mc) = single(pos_actual);
        oracle_resp_real(:, mc) = single(real(sensor_resp));
        oracle_resp_imag(:, mc) = single(imag(sensor_resp));

        if compute_l1SVD_UnESPRIT
            [ang_l1, ~] = l1_SVD_DoA_est(Y, M, threshold_l1SVD, K, angle_scan);
            l1SVD_est(:, mc) = sort(ang_l1)';

            doa_uesp = unit_ESPRIT(Y, T, ds, K, w);
            UnESPRIT_est(:, mc) = sort(doa_uesp);
        end
    end

    % 将当前 rho 组写入 HDF5。
    h5create(fname_main, [grp '/sam'], size(r_sam));
    h5write(fname_main, [grp '/sam'], r_sam);
    h5create(fname_main, [grp '/tau_corr'], size(tau_corr));
    h5write(fname_main, [grp '/tau_corr'], tau_corr);
    h5create(fname_main, [grp '/angles'], size(true_angles));
    h5write(fname_main, [grp '/angles'], true_angles);
    h5create(fname_main, [grp '/oracle_pos'], size(oracle_pos), 'Datatype', 'single');
    h5write(fname_main, [grp '/oracle_pos'], oracle_pos);
    h5create(fname_main, [grp '/oracle_resp_real'], size(oracle_resp_real), 'Datatype', 'single');
    h5write(fname_main, [grp '/oracle_resp_real'], oracle_resp_real);
    h5create(fname_main, [grp '/oracle_resp_imag'], size(oracle_resp_imag), 'Datatype', 'single');
    h5write(fname_main, [grp '/oracle_resp_imag'], oracle_resp_imag);
    h5create(fname_main, [grp '/oracle_cpl_real'], size(real(cpl_vec)), 'Datatype', 'single');
    h5write(fname_main, [grp '/oracle_cpl_real'], single(real(cpl_vec)));
    h5create(fname_main, [grp '/oracle_cpl_imag'], size(imag(cpl_vec)), 'Datatype', 'single');
    h5write(fname_main, [grp '/oracle_cpl_imag'], single(imag(cpl_vec)));

    % 可选：保存 l1-SVD / UnESPRIT 结果文件。
    if compute_l1SVD_UnESPRIT
        fname_l1 = fullfile(data_dir, sprintf('C3_l1SVD_M%d_K%d_Rho%.1f_T%d.h5', M, K, rho, T));
        if exist(fname_l1, 'file'), delete(fname_l1); end
        h5create(fname_l1, '/l1_SVD_ang', size(l1SVD_est));
        h5write(fname_l1, '/l1_SVD_ang', l1SVD_est);

        fname_ues = fullfile(data_dir, sprintf('C3_UnESPRIT_M%d_K%d_Rho%.1f_T%d.h5', M, K, rho, T));
        if exist(fname_ues, 'file'), delete(fname_ues); end
        h5create(fname_ues, '/UnESPRIT_ang', size(UnESPRIT_est));
        h5write(fname_ues, '/UnESPRIT_ang', UnESPRIT_est);
    end

    fprintf('rho = %.2f 完成（%.2f 分钟）\n', rho, toc/60);
end

time_total = toc / 60;
fprintf('联合阵列误差数据生成完成（N_mc=%d），总耗时 %.2f 分钟。\n', N_mc, time_total);
if compute_l1SVD_UnESPRIT
    fprintf('l1-SVD 和 UnESPRIT 结果已保存。\n');
else
    fprintf('已跳过 l1-SVD 和 UnESPRIT 计算。\n');
end

% ================= 辅助函数：互耦矩阵 =================
function C = mutual_coupling_matrix(M, c_vec)
    C = eye(M);
    for d = 1:length(c_vec)
        c = c_vec(d);
        for i = 1:M-d
            C(i, i+d) = c;
            C(i+d, i) = c;
        end
    end
end
