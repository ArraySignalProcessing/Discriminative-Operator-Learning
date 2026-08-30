% baselineC2_coherent_sources.m
% C2：相干源 rho 扫描下的传统基线 RMSE 与 PoR 计算。
% 包含 MUSIC、Root-MUSIC、标准 ESPRIT、TR-MUSIC、SS-MUSIC、SS-ESPRIT，以及可选的 l1-SVD、UnESPRIT。

clear all; close all;
tic;
addpath(fullfile(fileparts(mfilename('fullpath')), 'function'));

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C2');
if ~exist(data_dir, 'dir')
    error('数据目录不存在: %s', data_dir);
end

% ================= 参数，需要与 generateC2_coherent_sources_data.m 对齐 =================
rho_vec    = [ 0.3, 0.6, 0.9, 0.95:0.01:1.0];
SNR_dB     = -10;
K          = 2;
M          = 10;
T          = 200;
N_mc       = 1000;
angle_scan = -60:0.5:60;

% 标准 ESPRIT 参数
ds = 1;

% 空间平滑参数
L_ss = 7;

% PoR 判定阈值，单位：度
resolution_threshold = 1.0;

% ================= 主 HDF5 文件 =================
fname_main = fullfile(data_dir, 'C2_CoherentSources.h5');
if ~exist(fname_main, 'file')
    error('找不到数据文件: %s', fname_main);
end

nRho = length(rho_vec);

% ================= 预分配 =================
RMSE.MUSIC      = zeros(1, nRho);
RMSE.RootMUSIC  = zeros(1, nRho);
RMSE.ESPRIT     = zeros(1, nRho);
RMSE.TRMUSIC    = zeros(1, nRho);
RMSE.SSMUSIC    = zeros(1, nRho);
RMSE.SSESPRIT   = zeros(1, nRho);
RMSE.l1SVD      = nan(1, nRho);
RMSE.UnESPRIT   = nan(1, nRho);

PoR.MUSIC       = zeros(1, nRho);
PoR.RootMUSIC   = zeros(1, nRho);
PoR.ESPRIT      = zeros(1, nRho);
PoR.TRMUSIC     = zeros(1, nRho);
PoR.SSMUSIC     = zeros(1, nRho);
PoR.SSESPRIT    = zeros(1, nRho);
PoR.l1SVD       = nan(1, nRho);
PoR.UnESPRIT    = nan(1, nRho);

for idx = 1:nRho
    rho_val = rho_vec(idx);
    if rho_val < 1
        rho_tag = strrep(sprintf('%.2f', rho_val), '.', 'p');
    else
        rho_tag = '1p00';
    end
    grp = sprintf('/Rho_%s', rho_tag);

    % 读取样本协方差与真实角度
    r_sam = h5read(fname_main, [grp '/sam']);       % [M, M, 2, N_mc]
    true_angles_raw = h5read(fname_main, [grp '/angles']);
    R_sam = squeeze(r_sam(:, :, 1, :) + 1j * r_sam(:, :, 2, :));
    true_angles = sort(true_angles_raw, 1);

    if size(R_sam, 3) ~= N_mc
        actual_N_mc = size(R_sam, 3);
    else
        actual_N_mc = N_mc;
    end

    sq_music = 0; sq_rmusic = 0; sq_esprit = 0;
    sq_trmusic = 0; sq_ssmusic = 0; sq_ssesprit = 0;
    cnt_music = 0; cnt_rmusic = 0; cnt_esprit = 0;
    cnt_trmusic = 0; cnt_ssmusic = 0; cnt_ssesprit = 0;

    for mc = 1:actual_N_mc
        Rx = R_sam(:, :, mc);
        gt = true_angles(:, mc);

        % 普通子空间算法
        doa_m = sort(musicdoa(Rx, K, 'ScanAngles', angle_scan))';
        doa_rm = sort(rootmusicdoa(Rx, K))';
        doa_esp = sort(ESPRIT_doa(Rx, ds, K))';

        [err_music, maxerr_music] = match_doa_error(doa_m, gt);
        [err_rmusic, maxerr_rmusic] = match_doa_error(doa_rm, gt);
        [err_esprit, maxerr_esprit] = match_doa_error(doa_esp, gt);
        sq_music = sq_music + err_music;
        sq_rmusic = sq_rmusic + err_rmusic;
        sq_esprit = sq_esprit + err_esprit;

        if maxerr_music <= resolution_threshold
            cnt_music = cnt_music + 1;
        end
        if maxerr_rmusic <= resolution_threshold
            cnt_rmusic = cnt_rmusic + 1;
        end
        if maxerr_esprit <= resolution_threshold
            cnt_esprit = cnt_esprit + 1;
        end

        % Toeplitz reconstruction MUSIC for coherent-source decorrelation
        R_tr = toeplitz_covariance_reconstruct(Rx);
        doa_tr = sort(musicdoa(R_tr, K, 'ScanAngles', angle_scan))';
        [err_trmusic, maxerr_trmusic] = match_doa_error(doa_tr, gt);
        sq_trmusic = sq_trmusic + err_trmusic;
        if maxerr_trmusic <= resolution_threshold
            cnt_trmusic = cnt_trmusic + 1;
        end

        % 空间平滑 MUSIC
        R_ss = forward_spatial_smoothing(Rx, L_ss);
        doa_ssm = sort(musicdoa(R_ss, K, 'ScanAngles', angle_scan))';
        [err_ssmusic, maxerr_ssmusic] = match_doa_error(doa_ssm, gt);
        sq_ssmusic = sq_ssmusic + err_ssmusic;
        if maxerr_ssmusic <= resolution_threshold
            cnt_ssmusic = cnt_ssmusic + 1;
        end

        % 空间平滑 ESPRIT
        doa_sse = sort(ss_esprit_robust(Rx, L_ss, K))';
        [err_ssesprit, maxerr_ssesprit] = match_doa_error(doa_sse, gt);
        sq_ssesprit = sq_ssesprit + err_ssesprit;
        if maxerr_ssesprit <= resolution_threshold
            cnt_ssesprit = cnt_ssesprit + 1;
        end
    end

    RMSE.MUSIC(idx)     = sqrt(sq_music    / (K * actual_N_mc));
    RMSE.RootMUSIC(idx) = sqrt(sq_rmusic   / (K * actual_N_mc));
    RMSE.ESPRIT(idx)    = sqrt(sq_esprit   / (K * actual_N_mc));
    RMSE.TRMUSIC(idx)   = sqrt(sq_trmusic  / (K * actual_N_mc));
    RMSE.SSMUSIC(idx)   = sqrt(sq_ssmusic  / (K * actual_N_mc));
    RMSE.SSESPRIT(idx)  = sqrt(sq_ssesprit / (K * actual_N_mc));

    PoR.MUSIC(idx)      = cnt_music    / actual_N_mc;
    PoR.RootMUSIC(idx)  = cnt_rmusic   / actual_N_mc;
    PoR.ESPRIT(idx)     = cnt_esprit   / actual_N_mc;
    PoR.TRMUSIC(idx)    = cnt_trmusic  / actual_N_mc;
    PoR.SSMUSIC(idx)    = cnt_ssmusic  / actual_N_mc;
    PoR.SSESPRIT(idx)   = cnt_ssesprit / actual_N_mc;

    % l1-SVD 结果，如果已经预先生成则一并统计
    fname_l1 = fullfile(data_dir, sprintf('C2_l1SVD_M%d_K%d_Rho%s_T%d.h5', M, K, rho_tag, T));
    if exist(fname_l1, 'file')
        l1_ang = h5read(fname_l1, '/l1_SVD_ang');
        sq_l1 = 0; cnt_l1 = 0;
        for mc = 1:actual_N_mc
            [err_l1, maxerr_l1] = match_doa_error(l1_ang(:, mc), true_angles(:, mc));
            sq_l1 = sq_l1 + err_l1;
            if maxerr_l1 <= resolution_threshold
                cnt_l1 = cnt_l1 + 1;
            end
        end
        RMSE.l1SVD(idx) = sqrt(sq_l1 / (K * actual_N_mc));
        PoR.l1SVD(idx)  = cnt_l1 / actual_N_mc;
    end

    % UnESPRIT 结果，如果已经预先生成则一并统计
    fname_ues = fullfile(data_dir, sprintf('C2_UnESPRIT_M%d_K%d_Rho%s_T%d.h5', M, K, rho_tag, T));
    if exist(fname_ues, 'file')
        ues_ang = h5read(fname_ues, '/UnESPRIT_ang');
        sq_ues = 0; cnt_ues = 0;
        for mc = 1:actual_N_mc
            [err_ues, maxerr_ues] = match_doa_error(ues_ang(:, mc), true_angles(:, mc));
            sq_ues = sq_ues + err_ues;
            if maxerr_ues <= resolution_threshold
                cnt_ues = cnt_ues + 1;
            end
        end
        RMSE.UnESPRIT(idx) = sqrt(sq_ues / (K * actual_N_mc));
        PoR.UnESPRIT(idx)  = cnt_ues / actual_N_mc;
    end

    fprintf('rho = %.2f 完成 (%.1f min)\n', rho_val, toc / 60);
end

save(fullfile(data_dir, 'C2_Baselines.mat'), 'rho_vec', 'SNR_dB', 'RMSE', 'PoR');

fprintf('\n=== C2 RMSE 表格 (SNR=%d dB) ===\n', SNR_dB);
Tab_RMSE = table(rho_vec(:), ...
    RMSE.MUSIC(:), RMSE.RootMUSIC(:), RMSE.ESPRIT(:), ...
    RMSE.TRMUSIC(:), RMSE.SSMUSIC(:), RMSE.SSESPRIT(:), RMSE.l1SVD(:), RMSE.UnESPRIT(:), ...
    'VariableNames', {'rho', 'MUSIC', 'Root-MUSIC', 'ESPRIT', 'TR-MUSIC', 'SS-MUSIC', 'SS-ESPRIT', 'l1-SVD', 'UnESPRIT'});
disp(Tab_RMSE);

fprintf('\n=== C2 PoR 表格 (SNR=%d dB) ===\n', SNR_dB);
Tab_PoR = table(rho_vec(:), ...
    PoR.MUSIC(:) * 100, PoR.RootMUSIC(:) * 100, PoR.ESPRIT(:) * 100, ...
    PoR.TRMUSIC(:) * 100, PoR.SSMUSIC(:) * 100, PoR.SSESPRIT(:) * 100, PoR.l1SVD(:) * 100, PoR.UnESPRIT(:) * 100, ...
    'VariableNames', {'rho', 'MUSIC_%', 'Root-MUSIC_%', 'ESPRIT_%', 'TR-MUSIC_%', 'SS-MUSIC_%', 'SS-ESPRIT_%', 'l1-SVD_%', 'UnESPRIT_%'});
disp(Tab_PoR);

fprintf('C2 所有基线 RMSE 和 PoR 已保存。\n');
toc;

% ================= 辅助函数 =================
function R_ss = forward_spatial_smoothing(R, L)
    M = size(R, 1);
    P = M - L + 1;
    R_ss = zeros(L, L);
    for p = 1:P
        idx = p:p+L-1;
        R_ss = R_ss + R(idx, idx);
    end
    R_ss = R_ss / P;
    R_ss = (R_ss + R_ss') / 2;
end

function doa = ss_esprit_robust(Rx, L_ss, K)
    R_ss = forward_spatial_smoothing(Rx, L_ss);
    [E, D] = eig(R_ss);
    [~, ind] = sort(real(diag(D)), 'descend');
    Es = E(:, ind(1:K));
    Es1 = Es(1:end-1, :);
    Es2 = Es(2:end, :);
    Psi = pinv(Es1) * Es2;
    angles_rad = angle(eig(Psi));
    u = min(max(real(angles_rad) / pi, -1), 1);
    doa = asin(u) * 180 / pi;
    doa = sort(real(doa(:)))';
end
