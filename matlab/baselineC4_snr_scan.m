% baselineC4_snr_scan.m
% 论文实验 Topic 1A —— 传统基线 RMSE + PoR 计算
% 包括 MUSIC, Root-MUSIC, ESPRIT, Shrinkage-MUSIC + l1-SVD, UnESPRIT
% 所有可调参数集中于脚本开头。
% 作者: D
% 日期: 2026-05-09 (revised 2026-05-09)

clear all; close all;
tic;
addpath(fullfile(fileparts(mfilename('fullpath')), 'function'));

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C4');
if ~exist(data_dir, 'dir')
    error('数据目录不存在: %s', data_dir);
end

% ================= 可调参数（需与 GENER 脚本一致） =================
SNR_vec    = -20:2:10;          % Scenario 1A SNR scan (dB)
K          = 2;                 % 信源数
M          = 10;                % 阵元数
T          = 100;            % 快拍数 (与生成器对齐)
N_mc       = 1000;              % 蒙特卡洛次数（用于检查）
angle_scan = -60:0.5:60;        % MUSIC 搜索网格

% ESPRIT 参数
ds = 1;                           % 子阵列抽取因子

% Shrinkage-MUSIC 参数
shrinkage_alpha = 0.8;            % shrinkage toward Toeplitz target
shrinkage_beta  = 0.1;            % shrinkage toward trace(R)/M * I

% ----- 新增：PoR 分辨门限（度）-----
resolution_threshold = 2.0;        % 认为成功分辨的角度误差上限

% ================= 主 HDF5 文件 =================
fname_main = fullfile(data_dir, 'C4_SnrScan.h5');
if ~exist(fname_main, 'file')
    error('找不到数据文件: %s', fname_main);
end

nSNR = length(SNR_vec);

% 预分配 RMSE 和 PoR
RMSE.MUSIC      = zeros(1, nSNR);
RMSE.RootMUSIC  = zeros(1, nSNR);
RMSE.ESPRIT     = zeros(1, nSNR);
RMSE.ShrinkageMUSIC = zeros(1, nSNR);
RMSE.l1SVD      = nan(1, nSNR);
RMSE.UnESPRIT   = nan(1, nSNR);

PoR.MUSIC       = zeros(1, nSNR);
PoR.RootMUSIC   = zeros(1, nSNR);
PoR.ESPRIT      = zeros(1, nSNR);
PoR.ShrinkageMUSIC = zeros(1, nSNR);
PoR.l1SVD       = nan(1, nSNR);
PoR.UnESPRIT    = nan(1, nSNR);

for idx = 1:nSNR
    snr_val = SNR_vec(idx);
    if snr_val < 0
        snr_tag = sprintf('min%ddB', abs(snr_val));
    else
        snr_tag = sprintf('%ddB', snr_val);
    end
    grp = sprintf('/SNR_%s', snr_tag);
    
    % 读取协方差数据（2通道，实部+虚部）
    r_sam       = h5read(fname_main, [grp '/sam']);       % [M, M, 2, N_mc]
    true_angles = h5read(fname_main, [grp '/angles']);    % [K, N_mc]
    R_sam = squeeze(r_sam(:,:,1,:) + 1j*r_sam(:,:,2,:));  % 复协方差
    
    % 对真实角度排序，确保与估计值对齐
    true_angles = sort(true_angles, 1);
    
    % 检查蒙特卡洛次数
    if size(R_sam,3) ~= N_mc
        actual_N_mc = size(R_sam,3);
    else
        actual_N_mc = N_mc;
    end

    % 初始化平方误差和成功分辨计数
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
        
        % RMSE 累积
        [err_music, maxerr_music] = match_doa_error(doa_m, gt);
        [err_rmusic, maxerr_rmusic] = match_doa_error(doa_rm, gt);
        [err_esprit, maxerr_esprit] = match_doa_error(doa_esp, gt);
        [err_shrink, maxerr_shrink] = match_doa_error(doa_shrink, gt);
        sq_music  = sq_music  + err_music;
        sq_rmusic = sq_rmusic + err_rmusic;
        sq_esprit = sq_esprit + err_esprit;
        sq_shrink = sq_shrink + err_shrink;
        
        % PoR 判断：所有角度误差均 <= 门限
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
    fname_l1 = fullfile(data_dir, sprintf('C4_l1SVD_M%d_K%d_%s_T%d.h5', M, K, snr_tag, T));
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
    fname_ues = fullfile(data_dir, sprintf('C4_UnESPRIT_M%d_K%d_%s_T%d.h5', M, K, snr_tag, T));
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
    
    fprintf('SNR = %2d dB 完成 (%.1f min)\n', snr_val, toc/60);
end

% 保存结果（现在包含 PoR）
save(fullfile(data_dir, 'C4_Baselines.mat'), ...
    'SNR_vec', 'RMSE', 'PoR', 'shrinkage_alpha', 'shrinkage_beta');

% ================= 输出统一表格（包含 RMSE 和 PoR） =================
% RMSE 表格
fprintf('\n=== Topic 1A RMSE 表格 ===\n');
Tab_RMSE = table(SNR_vec(:), ...
    RMSE.MUSIC(:), RMSE.RootMUSIC(:), RMSE.ESPRIT(:), ...
    RMSE.ShrinkageMUSIC(:), RMSE.l1SVD(:), RMSE.UnESPRIT(:), ...
    'VariableNames', {'SNR_dB', 'MUSIC', 'Root-MUSIC', 'ESPRIT', 'Shrinkage-MUSIC', 'l1-SVD', 'UnESPRIT'});
disp(Tab_RMSE);

% PoR is saved for backward compatibility but not displayed for Topic 1A.

fprintf('所有基线 RMSE 和 PoR 已保存。\n');
toc;
