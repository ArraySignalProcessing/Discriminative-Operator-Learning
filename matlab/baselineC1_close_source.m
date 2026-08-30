% baselineC1_close_source.m
% 论文实验 C1 —— 传统基线 RMSE + PoR 计算（角度间隔扫描）
% 包括 MUSIC, Root-MUSIC, ESPRIT + l1-SVD, UnESPRIT
% 所有可调参数集中于脚本开头，自动输出 RMSE/PoR 表格。
% 作者: D
% 日期: 2026-05-09 (revised 2026-05-09)

clear all; close all;
tic;
addpath(fullfile(fileparts(mfilename('fullpath')), 'function'));

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C1');
if ~exist(data_dir, 'dir')
    error('数据目录不存在: %s', data_dir);
end

% ================= 可调参数（需与 GENER 脚本一致） =================
delta_theta_vec = [2:1:5,6:3:15];              % C1 angular separations
K          = 2;                        % 信源数
M          = 10;                       % 阵元数
T          = 200;                       % Must match generateC1_close_source_data.m
N_mc       = 2000;                      % 蒙特卡洛次数（用于检查）
angle_scan = -60:0.5:60;               % MUSIC 搜索网格

% ESPRIT 参数
ds = 1;                                 % 子阵列抽取因子

% ----- 新增：PoR 分辨门限（度）-----
resolution_threshold = 1.0;              % 认为成功分辨的角度误差上限

% ================= 主 HDF5 文件 =================
fname_main = fullfile(data_dir, 'C1_CloseSource.h5');
if ~exist(fname_main, 'file')
    error('找不到数据文件: %s', fname_main);
end

nSep = length(delta_theta_vec);

% 预分配 RMSE 和 PoR
RMSE.MUSIC      = zeros(1, nSep);
RMSE.RootMUSIC  = zeros(1, nSep);
RMSE.ESPRIT     = zeros(1, nSep);
RMSE.l1SVD      = nan(1, nSep);
RMSE.UnESPRIT   = nan(1, nSep);

PoR.MUSIC       = zeros(1, nSep);
PoR.RootMUSIC   = zeros(1, nSep);
PoR.ESPRIT      = zeros(1, nSep);
PoR.l1SVD       = nan(1, nSep);
PoR.UnESPRIT    = nan(1, nSep);

for idx = 1:nSep
    Delta = delta_theta_vec(idx);
    sep_tag = sprintf('%ddeg', Delta);
    grp = sprintf('/Sep_%s', sep_tag);

    % 读取协方差数据（2 通道：实部+虚部）
    r_sam       = h5read(fname_main, [grp '/sam']);       % [M, M, 2, N_mc]
    true_angles_raw = h5read(fname_main, [grp '/angles']);% [K, N_mc]
    R_sam = squeeze(r_sam(:,:,1,:) + 1j*r_sam(:,:,2,:));  % 复协方差矩阵
    true_angles = sort(true_angles_raw, 1);               % 按行排序（K,N_mc）

    % 检查蒙特卡洛次数
    if size(R_sam,3) ~= N_mc
        actual_N_mc = size(R_sam,3);
    else
        actual_N_mc = N_mc;
    end

    % 1. MUSIC / Root-MUSIC / ESPRIT
    sq_music=0; sq_rmusic=0; sq_esprit=0;
    cnt_music=0; cnt_rmusic=0; cnt_esprit=0;
    for mc = 1:actual_N_mc
        Rx = R_sam(:,:,mc);
        gt = true_angles(:, mc);

        doa_m  = sort(musicdoa(Rx, K, 'ScanAngles', angle_scan))';
        doa_rm = sort(rootmusicdoa(Rx, K))';
        doa_esp = sort(ESPRIT_doa(Rx, ds, K))';

        [err_music, maxerr_music] = match_doa_error(doa_m, gt);
        [err_rmusic, maxerr_rmusic] = match_doa_error(doa_rm, gt);
        [err_esprit, maxerr_esprit] = match_doa_error(doa_esp, gt);
        sq_music  = sq_music  + err_music;
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
    end
    RMSE.MUSIC(idx)     = sqrt(sq_music  / (K * actual_N_mc));
    RMSE.RootMUSIC(idx) = sqrt(sq_rmusic / (K * actual_N_mc));
    RMSE.ESPRIT(idx)    = sqrt(sq_esprit / (K * actual_N_mc));
    PoR.MUSIC(idx)      = cnt_music  / actual_N_mc;
    PoR.RootMUSIC(idx)  = cnt_rmusic / actual_N_mc;
    PoR.ESPRIT(idx)     = cnt_esprit / actual_N_mc;

    % 2. l1-SVD
    fname_l1 = fullfile(data_dir, sprintf('C1_l1SVD_M%d_K%d_Sep%s_T%d.h5', M, K, sep_tag, T));
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

    % 3. UnESPRIT
    fname_ues = fullfile(data_dir, sprintf('C1_UnESPRIT_M%d_K%d_Sep%s_T%d.h5', M, K, sep_tag, T));
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

    fprintf('Δθ = %d° 完成 (%.1f min)\n', Delta, toc/60);
end

% 保存结果（包含 PoR）
save(fullfile(data_dir, 'C1_Baselines.mat'), 'delta_theta_vec', 'RMSE', 'PoR');

% 输出 RMSE 表格
fprintf('\n=== C1 RMSE 表格 ===\n');
Tab_RMSE = table(delta_theta_vec(:), ...
    RMSE.MUSIC(:), RMSE.RootMUSIC(:), RMSE.ESPRIT(:), ...
    RMSE.l1SVD(:), RMSE.UnESPRIT(:), ...
    'VariableNames', {'DeltaTheta', 'MUSIC', 'Root-MUSIC', 'ESPRIT', 'l1-SVD', 'UnESPRIT'});
disp(Tab_RMSE);

% 输出 PoR 表格
fprintf('\n=== C1 PoR 表格 ===\n');
Tab_PoR = table(delta_theta_vec(:), ...
    PoR.MUSIC(:)*100, PoR.RootMUSIC(:)*100, PoR.ESPRIT(:)*100, ...
    PoR.l1SVD(:)*100, PoR.UnESPRIT(:)*100, ...
    'VariableNames', {'DeltaTheta', 'MUSIC_%', 'Root-MUSIC_%', 'ESPRIT_%', 'l1-SVD_%', 'UnESPRIT_%'});
disp(Tab_PoR);

fprintf('所有基线 RMSE 和 PoR 已保存。\n');
toc;
