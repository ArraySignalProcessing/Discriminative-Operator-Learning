% baselineC3_array_mismatch.m
% 论文实验 Topic 1E —— 联合阵列误差基线 RMSE + PoR
% 读取 C3_ArrayMismatch.h5，使用理想 ULA 模型估计 DOA
% 包含 MUSIC, Root‑MUSIC, ESPRIT, TP-MUSIC, Oracle-MUSIC + l1‑SVD, UnESPRIT（若存在）
% 所有可调参数集中于开头，输出 RMSE/PoR 表格。
% 作者: D
% 日期: 2026-05-09 (revised 2026-05-09)

clear all; close all;
tic;
addpath(fullfile(fileparts(mfilename('fullpath')), 'function'));

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C3');
if ~exist(data_dir, 'dir')
    error('数据目录不存在: %s', data_dir);
end

% ================= 可调参数（需与 GENER 一致） =================
M          = 10;
K          = 2;
T          = 500;                 % Must match generateC3_array_mismatch_data.m
N_mc       = 1000;               % 1000（与 GENER 一致）
angle_scan = -60:0.5:60;
ds = 1;

rho_all = 0:0.1:1;
nRho = length(rho_all);

% ----- 新增：PoR 分辨门限（度）-----
resolution_threshold = 1.0;        % PoR threshold, consistent with learning baselines

% ================= 主 HDF5 文件 =================
fname_main = fullfile(data_dir, 'C3_ArrayMismatch.h5');
if ~exist(fname_main, 'file')
    error('找不到数据文件: %s', fname_main);
end

% 预分配 RMSE 和 PoR 结构体
RMSE.MUSIC      = zeros(1, nRho);
RMSE.RootMUSIC  = zeros(1, nRho);
RMSE.ESPRIT     = zeros(1, nRho);
RMSE.TPMUSIC    = zeros(1, nRho);
RMSE.OracleMUSIC = nan(1, nRho);
RMSE.l1SVD      = nan(1, nRho);
RMSE.UnESPRIT   = nan(1, nRho);

PoR.MUSIC       = zeros(1, nRho);
PoR.RootMUSIC   = zeros(1, nRho);
PoR.ESPRIT      = zeros(1, nRho);
PoR.TPMUSIC     = zeros(1, nRho);
PoR.OracleMUSIC = nan(1, nRho);
PoR.l1SVD       = nan(1, nRho);
PoR.UnESPRIT    = nan(1, nRho);

for idx = 1:nRho
    rho = rho_all(idx);
    grp = sprintf('/Rho_%.2f', rho);

    % 读取 2 通道协方差
    r_sam       = h5read(fname_main, [grp '/sam']);       % [M, M, 2, N_mc]
    true_angles = h5read(fname_main, [grp '/angles']);    % [K, N_mc]
    R_sam = squeeze(r_sam(:,:,1,:) + 1j*r_sam(:,:,2,:));
    true_angles = sort(true_angles, 1);                   % 每列排序
    has_oracle = h5_dataset_exists(fname_main, [grp '/oracle_pos']) && ...
        h5_dataset_exists(fname_main, [grp '/oracle_resp_real']) && ...
        h5_dataset_exists(fname_main, [grp '/oracle_resp_imag']) && ...
        h5_dataset_exists(fname_main, [grp '/oracle_cpl_real']) && ...
        h5_dataset_exists(fname_main, [grp '/oracle_cpl_imag']);
    if has_oracle
        oracle_pos = h5read(fname_main, [grp '/oracle_pos']);
        oracle_resp = h5read(fname_main, [grp '/oracle_resp_real']) + ...
            1j * h5read(fname_main, [grp '/oracle_resp_imag']);
        oracle_cpl_vec = h5read(fname_main, [grp '/oracle_cpl_real']) + ...
            1j * h5read(fname_main, [grp '/oracle_cpl_imag']);
    else
        warning('Oracle array manifold fields are missing for %s. Oracle-MUSIC is set to NaN.', grp);
    end

    if size(R_sam,3) ~= N_mc
        actual_N_mc = size(R_sam,3);
    else
        actual_N_mc = N_mc;
    end

    % MUSIC / Root-MUSIC / ESPRIT + PoR
    sq_music=0; sq_rmusic=0; sq_esprit=0; sq_tpmusic=0; sq_oracle=0;
    cnt_music=0; cnt_rmusic=0; cnt_esprit=0; cnt_tpmusic=0; cnt_oracle=0;
    for mc = 1:actual_N_mc
        Rx = R_sam(:,:,mc);
        gt = true_angles(:, mc);

        doa_m  = sort(musicdoa(Rx, K, 'ScanAngles', angle_scan))';
        doa_rm = sort(rootmusicdoa(Rx, K))';
        doa_esp = sort(ESPRIT_doa(Rx, ds, K))';
        R_tp = toeplitz_covariance_reconstruct(Rx);
        doa_tp = sort(musicdoa(R_tp, K, 'ScanAngles', angle_scan))';
        if has_oracle
            A_oracle_grid = build_oracle_steering_grid(angle_scan, oracle_pos(:, mc), ...
                oracle_resp(:, mc), oracle_cpl_vec, M);
            doa_oracle = oracle_music_doa(Rx, K, angle_scan, A_oracle_grid);
        end

        [err_music, maxerr_music] = match_doa_error(doa_m, gt);
        [err_rmusic, maxerr_rmusic] = match_doa_error(doa_rm, gt);
        [err_esprit, maxerr_esprit] = match_doa_error(doa_esp, gt);
        [err_tpmusic, maxerr_tpmusic] = match_doa_error(doa_tp, gt);
        sq_music  = sq_music  + err_music;
        sq_rmusic = sq_rmusic + err_rmusic;
        sq_esprit = sq_esprit + err_esprit;
        sq_tpmusic = sq_tpmusic + err_tpmusic;
        if has_oracle
            [err_oracle, maxerr_oracle] = match_doa_error(doa_oracle, gt);
            sq_oracle = sq_oracle + err_oracle;
        end

        if maxerr_music <= resolution_threshold
            cnt_music = cnt_music + 1;
        end
        if maxerr_rmusic <= resolution_threshold
            cnt_rmusic = cnt_rmusic + 1;
        end
        if maxerr_esprit <= resolution_threshold
            cnt_esprit = cnt_esprit + 1;
        end
        if maxerr_tpmusic <= resolution_threshold
            cnt_tpmusic = cnt_tpmusic + 1;
        end
        if has_oracle && maxerr_oracle <= resolution_threshold
            cnt_oracle = cnt_oracle + 1;
        end
    end
    RMSE.MUSIC(idx)     = sqrt(sq_music  / (K * actual_N_mc));
    RMSE.RootMUSIC(idx) = sqrt(sq_rmusic / (K * actual_N_mc));
    RMSE.ESPRIT(idx)    = sqrt(sq_esprit / (K * actual_N_mc));
    RMSE.TPMUSIC(idx)   = sqrt(sq_tpmusic / (K * actual_N_mc));
    if has_oracle
        RMSE.OracleMUSIC(idx) = sqrt(sq_oracle / (K * actual_N_mc));
    end
    PoR.MUSIC(idx)      = cnt_music  / actual_N_mc;
    PoR.RootMUSIC(idx)  = cnt_rmusic / actual_N_mc;
    PoR.ESPRIT(idx)     = cnt_esprit / actual_N_mc;
    PoR.TPMUSIC(idx)    = cnt_tpmusic / actual_N_mc;
    if has_oracle
        PoR.OracleMUSIC(idx) = cnt_oracle / actual_N_mc;
    end

    % l1-SVD
    fname_l1 = fullfile(data_dir, sprintf('C3_l1SVD_M%d_K%d_Rho%.1f_T%d.h5', M, K, rho, T));
    if exist(fname_l1, 'file')
        l1_ang = h5read(fname_l1, '/l1_SVD_ang');
        sq_l1 = 0; cnt_l1 = 0;
        for mc = 1:actual_N_mc
            [err_l1, maxerr_l1] = match_doa_error(l1_ang(:,mc), true_angles(:,mc));
            sq_l1 = sq_l1 + err_l1;
            if maxerr_l1 <= resolution_threshold
                cnt_l1 = cnt_l1 + 1;
            end
        end
        RMSE.l1SVD(idx) = sqrt(sq_l1 / (K * actual_N_mc));
        PoR.l1SVD(idx)  = cnt_l1 / actual_N_mc;
    end

    % UnESPRIT
    fname_ues = fullfile(data_dir, sprintf('C3_UnESPRIT_M%d_K%d_Rho%.2f_T%d.h5', M, K, rho, T));
    if exist(fname_ues, 'file')
        ues_ang = h5read(fname_ues, '/UnESPRIT_ang');
        sq_ues = 0; cnt_ues = 0;
        for mc = 1:actual_N_mc
            [err_ues, maxerr_ues] = match_doa_error(ues_ang(:,mc), true_angles(:,mc));
            sq_ues = sq_ues + err_ues;
            if maxerr_ues <= resolution_threshold
                cnt_ues = cnt_ues + 1;
            end
        end
        RMSE.UnESPRIT(idx) = sqrt(sq_ues / (K * actual_N_mc));
        PoR.UnESPRIT(idx)  = cnt_ues / actual_N_mc;
    end

    fprintf('ρ = %.2f 基线完成 (%.2f min)\n', rho, toc/60);
end

% 保存结果（包含 PoR）
save(fullfile(data_dir, 'C3_Baselines.mat'), 'rho_all', 'RMSE', 'PoR', 'resolution_threshold');

% 输出 RMSE 表格
fprintf('\n=== Topic 1E RMSE 表格 (N_mc=1000) ===\n');
Tab_RMSE = table(rho_all(:), ...
    RMSE.MUSIC(:), RMSE.RootMUSIC(:), RMSE.ESPRIT(:), ...
    RMSE.TPMUSIC(:), RMSE.OracleMUSIC(:), RMSE.l1SVD(:), RMSE.UnESPRIT(:), ...
    'VariableNames', {'rho', 'MUSIC', 'Root-MUSIC', 'ESPRIT', 'TP-MUSIC', 'Oracle-MUSIC', 'l1-SVD', 'UnESPRIT'});
disp(Tab_RMSE);

% 输出 PoR 表格
fprintf('\n=== Topic 1E PoR 表格 (threshold=%.1f deg, N_mc=1000) ===\n', resolution_threshold);
Tab_PoR = table(rho_all(:), ...
    PoR.MUSIC(:)*100, PoR.RootMUSIC(:)*100, PoR.ESPRIT(:)*100, ...
    PoR.TPMUSIC(:)*100, PoR.OracleMUSIC(:)*100, PoR.l1SVD(:)*100, PoR.UnESPRIT(:)*100, ...
    'VariableNames', {'rho', 'MUSIC_%', 'Root-MUSIC_%', 'ESPRIT_%', 'TP-MUSIC_%', 'Oracle-MUSIC_%', 'l1-SVD_%', 'UnESPRIT_%'});
disp(Tab_PoR);

fprintf('所有基线 RMSE 和 PoR 已保存。\n');
toc;

function exists_flag = h5_dataset_exists(fname, dataset_path)
exists_flag = false;
try
    h5info(fname, dataset_path);
    exists_flag = true;
catch
end
end

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

function A_grid = build_oracle_steering_grid(angle_scan, pos_actual, sensor_resp, cpl_vec, M)
C = mutual_coupling_matrix(M, cpl_vec);
A_grid = zeros(M, numel(angle_scan));
for ii = 1:numel(angle_scan)
    a = exp(1j * 2 * pi * pos_actual(:) * sin(deg2rad(angle_scan(ii))));
    a = C * (sensor_resp(:) .* a);
    A_grid(:, ii) = sqrt(M) * a / max(norm(a), eps);
end
end

function doa = oracle_music_doa(Rx, K, angle_scan, A_grid)
[U, D] = eig(Rx);
[~, order] = sort(real(diag(D)), 'descend');
Un = U(:, order(K+1:end));
Pn = Un * Un';
den = real(sum(conj(A_grid) .* (Pn * A_grid), 1));
spectrum = 1 ./ max(den, eps);
idx = pick_spectrum_peaks(spectrum, K);
doa = sort(angle_scan(idx)).';
end

function idx = pick_spectrum_peaks(spectrum, K)
values = real(spectrum(:)).';
values(~isfinite(values)) = -inf;
is_peak = false(size(values));
if numel(values) == 1
    is_peak(1) = true;
else
    is_peak(1) = values(1) > values(2);
    is_peak(end) = values(end) > values(end-1);
    for ii = 2:numel(values)-1
        is_peak(ii) = values(ii) >= values(ii-1) && values(ii) >= values(ii+1) && ...
            (values(ii) > values(ii-1) || values(ii) > values(ii+1));
    end
end
candidates = find(is_peak);
[~, order] = sort(values(candidates), 'descend');
idx = candidates(order);
if numel(idx) < K
    [~, all_order] = sort(values, 'descend');
    idx = unique([idx, all_order], 'stable');
end
idx = idx(1:K);
end
