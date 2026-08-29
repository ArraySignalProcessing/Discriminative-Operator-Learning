% baselineC5_snapshot_scan.m
% 论文实验 Topic 1B —— 传统基线 RMSE + PoR 计算（快拍数扫描）
% 包括 MUSIC, Root-MUSIC, ESPRIT, Shrinkage-MUSIC + l1-SVD, UnESPRIT
% 所有可调参数集中于脚本开头，自动输出统一 RMSE/PoR 表格。
% 作者: D
% 日期: 2026-05-09 (revised 2026-05-09)

clear all; close all;
tic;
addpath(fullfile(fileparts(mfilename('fullpath')), 'function'));

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C5');
if ~exist(data_dir, 'dir')
    error('数据目录不存在: %s', data_dir);
end

% ================= 可调参数（需与 GENER 脚本一致） =================
% Unified benchmark T scan.
T_vec      = [10:10:50, 100:100:500, 1000, 2000];  % Scenario 1B snapshot counts
K          = 2;                        % 信源数
M          = 10;                       % 阵元数
N_mc       = 1000;                     % 蒙特卡洛次数（用于检查）
angle_scan = -60:0.5:60;               % MUSIC 搜索网格

% ESPRIT 参数
ds = 1;                                 % 子阵列抽取因子

% Shrinkage-MUSIC 参数
shrinkage_alpha = 0.80;                  % shrinkage toward Toeplitz target
shrinkage_beta  = 0.20;                  % shrinkage toward trace(R)/M * I

% ----- 新增：PoR 分辨门限（度）-----
resolution_threshold = 2.0;              % 认为成功分辨的角度误差上限

% ================= 主 HDF5 文件 =================
fname_main = fullfile(data_dir, 'C5_SnapshotScan.h5');
if ~exist(fname_main, 'file')
    error('找不到数据文件: %s', fname_main);
end

nT = length(T_vec);

% 预分配 RMSE 和 PoR
RMSE.MUSIC      = zeros(1, nT);
RMSE.RootMUSIC  = zeros(1, nT);
RMSE.ESPRIT     = zeros(1, nT);
RMSE.ShrinkageMUSIC = zeros(1, nT);
RMSE.l1SVD      = nan(1, nT);
RMSE.UnESPRIT   = nan(1, nT);

PoR.MUSIC       = zeros(1, nT);
PoR.RootMUSIC   = zeros(1, nT);
PoR.ESPRIT      = zeros(1, nT);
PoR.ShrinkageMUSIC = zeros(1, nT);
PoR.l1SVD       = nan(1, nT);
PoR.UnESPRIT    = nan(1, nT);

for idx = 1:nT
    T = T_vec(idx);
    grp = sprintf('/T_%d', T);
    
    % 读取协方差数据（2 通道：实部+虚部）
    r_sam       = h5read(fname_main, [grp '/sam']);       % [M, M, 2, N_mc]
    true_angles = h5read(fname_main, [grp '/angles']);    % [K, N_mc]
    R_sam = squeeze(r_sam(:,:,1,:) + 1j*r_sam(:,:,2,:));  % 复协方差矩阵
    
    % 对真实角度排序
    true_angles = sort(true_angles, 1);
    
    % 检查蒙特卡洛次数
    if size(R_sam,3) ~= N_mc
        actual_N_mc = size(R_sam,3);
    else
        actual_N_mc = N_mc;
    end
    
    % 1. MUSIC / Root-MUSIC / ESPRIT
    sq_music=0; sq_rmusic=0; sq_esprit=0; sq_shrink=0;
    cnt_music=0; cnt_rmusic=0; cnt_esprit=0; cnt_shrink=0;
    for mc = 1:actual_N_mc
        Rx = R_sam(:,:,mc);
        gt = true_angles(:, mc);
        
        doa_m  = sort(musicdoa(Rx, K, 'ScanAngles', angle_scan))';
        doa_rm = sort(rootmusicdoa(Rx, K))';
        doa_esp = sort(ESPRIT_doa(Rx, ds, K))';
        R_shrink = shrinkage_covariance(Rx, shrinkage_alpha, shrinkage_beta);
        doa_shrink = sort(musicdoa(R_shrink, K, 'ScanAngles', angle_scan))';
        
        [err_music, maxerr_music] = match_doa_error(doa_m, gt);
        [err_rmusic, maxerr_rmusic] = match_doa_error(doa_rm, gt);
        [err_esprit, maxerr_esprit] = match_doa_error(doa_esp, gt);
        [err_shrink, maxerr_shrink] = match_doa_error(doa_shrink, gt);
        sq_music  = sq_music  + err_music;
        sq_rmusic = sq_rmusic + err_rmusic;
        sq_esprit = sq_esprit + err_esprit;
        sq_shrink = sq_shrink + err_shrink;
        
        if maxerr_music <= resolution_threshold
            cnt_music = cnt_music + 1;
        end
        if maxerr_rmusic <= resolution_threshold
            cnt_rmusic = cnt_rmusic + 1;
        end
        if maxerr_esprit <= resolution_threshold
            cnt_esprit = cnt_esprit + 1;
        end
        if maxerr_shrink <= resolution_threshold
            cnt_shrink = cnt_shrink + 1;
        end
    end
    RMSE.MUSIC(idx)     = sqrt(sq_music  / (K * actual_N_mc));
    RMSE.RootMUSIC(idx) = sqrt(sq_rmusic / (K * actual_N_mc));
    RMSE.ESPRIT(idx)    = sqrt(sq_esprit / (K * actual_N_mc));
    RMSE.ShrinkageMUSIC(idx) = sqrt(sq_shrink / (K * actual_N_mc));
    PoR.MUSIC(idx)      = cnt_music  / actual_N_mc;
    PoR.RootMUSIC(idx)  = cnt_rmusic / actual_N_mc;
    PoR.ESPRIT(idx)     = cnt_esprit / actual_N_mc;
    PoR.ShrinkageMUSIC(idx) = cnt_shrink / actual_N_mc;
    
    % 2. l1-SVD
    fname_l1 = fullfile(data_dir, sprintf('C5_l1SVD_M%d_K%d_T%d.h5', M, K, T));
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
    fname_ues = fullfile(data_dir, sprintf('C5_UnESPRIT_M%d_K%d_T%d.h5', M, K, T));
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
    
    fprintf('T = %d 完成 (%.1f min)\n', T, toc/60);
end

% 保存结果（包含 PoR）
save(fullfile(data_dir, 'C5_Baselines.mat'), 'T_vec', 'RMSE', 'PoR', 'shrinkage_alpha', 'shrinkage_beta');

% 输出表格
fprintf('\n=== Topic 1B RMSE 表格 ===\n');
Tab_RMSE = table(T_vec(:), ...
    RMSE.MUSIC(:), RMSE.RootMUSIC(:), RMSE.ESPRIT(:), ...
    RMSE.ShrinkageMUSIC(:), RMSE.l1SVD(:), RMSE.UnESPRIT(:), ...
    'VariableNames', {'T', 'MUSIC', 'Root-MUSIC', 'ESPRIT', 'Shrinkage-MUSIC', 'l1-SVD', 'UnESPRIT'});
disp(Tab_RMSE);

% PoR is saved for backward compatibility but not displayed for Topic 1B.

fprintf('所有基线 RMSE 和 PoR 已保存。\n');
toc;
