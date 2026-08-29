% plotC4_snr_scan.m
% 绘制 Topic 1A：RMSE vs SNR
% 自动检测所有可用结果文件。
% 作者: D
% 日期: 2026-05-09 (revised 2026-05-09)

clear all; close all;

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C4');

% ================= Display switches =================
show_MUSIC       = true;
show_RootMUSIC   = true;
show_ESPRIT      = false;
show_ShrinkageMUSIC = true;
show_l1SVD       = false;
show_UnESPRIT    = false;
show_CNN         = true;
show_SubspaceNet = true;
show_Ours        = true;
show_RootOurs    = false;

% ================= 加载传统基线 =================
baseline_file = fullfile(data_dir, 'C4_Baselines.mat');
if ~exist(baseline_file, 'file')
    error('找不到基线结果: %s', baseline_file);
end
load(baseline_file, 'SNR_vec', 'RMSE', 'PoR');

% 提取基线
MUSIC_rmse    = RMSE.MUSIC;
RMUSIC_rmse   = RMSE.RootMUSIC;
ESPRIT_rmse   = RMSE.ESPRIT;
ShrinkageMUSIC_rmse = get_struct_vector(RMSE, 'ShrinkageMUSIC', size(SNR_vec));
l1SVD_rmse    = RMSE.l1SVD;
UnESPRIT_rmse = RMSE.UnESPRIT;
MUSIC_por     = PoR.MUSIC * 100;
RMUSIC_por    = PoR.RootMUSIC * 100;
ESPRIT_por    = PoR.ESPRIT * 100;
ShrinkageMUSIC_por = get_struct_vector(PoR, 'ShrinkageMUSIC', size(SNR_vec)) * 100;
l1SVD_por     = PoR.l1SVD * 100;
UnESPRIT_por  = PoR.UnESPRIT * 100;

% ================= 尝试加载 CNN 结果（仅 RMSE） =================
cnn_file = fullfile(data_dir, 'C4_CNN.h5');
CNN_rmse = nan(size(SNR_vec));
CNN_por  = nan(size(SNR_vec));
if exist(cnn_file, 'file')
    try
        CNN_rmse = h5read(cnn_file, '/CNN_RMSE');
    catch
    end
    try
        CNN_por = h5read(cnn_file, '/CNN_PoR') * 100;
    catch
    end
end

% ================= 尝试加载 Ours 结果 =================
ours_file = fullfile(data_dir, 'C4_Ours.h5');
Ours_rmse = nan(size(SNR_vec));
Ours_por  = nan(size(SNR_vec));
if exist(ours_file, 'file')
    try
        Ours_rmse = h5read(ours_file, '/Ours_RMSE');
    catch
    end
    try
        Ours_por = h5read(ours_file, '/Ours_PoR') * 100;
    catch
    end
end

% ================= 尝试加载 SubspaceNet 结果 =================
root_ours_file = fullfile(data_dir, 'C4_RootOurs.h5');
RootOurs_rmse = nan(size(SNR_vec));
RootOurs_por  = nan(size(SNR_vec));
if exist(root_ours_file, 'file')
    try
        RootOurs_rmse = h5read(root_ours_file, '/RootOurs_RMSE');
    catch
    end
    try
        RootOurs_por = h5read(root_ours_file, '/RootOurs_PoR') * 100;
    catch
    end
end

subspace_file = fullfile(data_dir, 'C4_SubspaceNet.h5');
SubspaceNet_rmse = nan(size(SNR_vec));
SubspaceNet_por  = nan(size(SNR_vec));
if exist(subspace_file, 'file')
    try
        SubspaceNet_rmse = h5read(subspace_file, '/SubspaceNet_RMSE');
    catch
    end
    try
        SubspaceNet_por = h5read(subspace_file, '/SubspaceNet_PoR') * 100;
    catch
    end
end

% ================= 构建动态方法列表 =================
allMethods = {'MUSIC','Root-MUSIC','ESPRIT','Shrinkage-MUSIC','l1-SVD','UnESPRIT','MLC','SubspaceNet','Ours','Root-Ours'};
rmse_data = {MUSIC_rmse, RMUSIC_rmse, ESPRIT_rmse, ShrinkageMUSIC_rmse, l1SVD_rmse, UnESPRIT_rmse, CNN_rmse, SubspaceNet_rmse, Ours_rmse, RootOurs_rmse};
por_data  = {MUSIC_por, RMUSIC_por, ESPRIT_por, ShrinkageMUSIC_por, l1SVD_por, UnESPRIT_por, CNN_por, SubspaceNet_por, Ours_por, RootOurs_por};
show_data = [show_MUSIC, show_RootMUSIC, show_ESPRIT, show_ShrinkageMUSIC, show_l1SVD, show_UnESPRIT, ...
             show_CNN, show_SubspaceNet, show_Ours, show_RootOurs];

methodNames = {};
methodRMSE  = [];
methodPOR   = [];

for i = 1:length(allMethods)
    r = rmse_data{i}(:);
    p = por_data{i}(:);
    if show_data(i) && ~all(isnan(r))
        methodNames{end+1} = allMethods{i};
        methodRMSE(:,end+1) = r;
        if ~all(isnan(p))
            methodPOR(:,end+1) = p;
        else
            methodPOR(:,end+1) = nan(size(r));
        end
    end
end

% 输出表格
Tab_RMSE = array2table([SNR_vec(:), methodRMSE], ...
    'VariableNames', ['SNR_dB', methodNames]);
fprintf('=== Topic 1A RMSE 表格 ===\n');
disp(Tab_RMSE);

% PoR is intentionally not displayed for Topic 1A; C.1 uses RMSE only.

% ================= 绘图：RMSE only =================
fig = figure('Position', [100 100 900 360], 'Color', 'w');

baselineColor = [0.47 0.67 0.19];
oursColor = [0.90 0.10 0.10];
defaultColor = [0.50 0.50 0.50];

methodColor = cell(1, length(methodNames));
methodLineStyle = cell(1, length(methodNames));
methodMarker = cell(1, length(methodNames));
for i = 1:length(methodNames)
    [methodColor{i}, methodLineStyle{i}, methodMarker{i}] = get_method_style(methodNames{i}, ...
        baselineColor, oursColor, defaultColor);
end

% ---- RMSE ----
hold on; grid on;
for i = 1:length(methodNames)
    plot(SNR_vec, methodRMSE(:,i), ...
        'LineStyle', methodLineStyle{i}, ...
        'Marker', methodMarker{i}, ...
        'LineWidth', 1.2, ...
        'MarkerSize', 3.6, ...
        'Color', methodColor{i}, ...
        'DisplayName', methodNames{i});
end
ylabel('RMSE (deg)', 'Interpreter', 'latex');
xlabel('SNR (dB)', 'Interpreter', 'latex');
title('RMSE versus SNR', 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
lgd = legend('Location', 'northoutside', 'Orientation', 'horizontal', ...
    'NumColumns', 3, 'Interpreter', 'latex');
lgd.FontSize = 8;
format_ieee_axes(gca);
set(gca, 'YScale', 'log');
valid_rmse = methodRMSE(isfinite(methodRMSE) & methodRMSE > 0);
if ~isempty(valid_rmse)
    ymin = 10 ^ floor(log10(min(valid_rmse)));
    ymax = 10 ^ ceil(log10(max(valid_rmse)));
    ylim([ymin, ymax]);
end
hold off;

export_ieee_figure(fig, 'plotC4_snr_scan');

function [color, lineStyle, marker] = get_method_style(name, baselineColor, oursColor, defaultColor)
switch name
    case {'MUSIC'}
        color = [0.00 0.45 0.74]; lineStyle = '-'; marker = 'o';
    case {'Root-MUSIC'}
        color = [0.85 0.33 0.10]; lineStyle = '-'; marker = 's';
    case {'ESPRIT'}
        color = [0.49 0.18 0.56]; lineStyle = '-'; marker = '^';
    case {'SS-MUSIC'}
        color = [0.30 0.75 0.93]; lineStyle = '-'; marker = 'd';
    case {'Shrinkage-MUSIC'}
        color = [0.93 0.69 0.13]; lineStyle = '-'; marker = 'v';
    case {'MLC'}
        color = baselineColor; lineStyle = '--'; marker = 's';
    case {'SubspaceNet'}
        color = [0.00 0.60 0.50]; lineStyle = '--'; marker = 's';
    case {'Ours'}
        color = oursColor; lineStyle = '--'; marker = '^';
    case {'Root-Ours'}
        color = [0.64 0.08 0.18]; lineStyle = '--'; marker = 'v';
    otherwise
        color = defaultColor; lineStyle = ':'; marker = 'o';
end
end

function value = get_struct_vector(S, fieldName, targetSize)
if isfield(S, fieldName)
    value = S.(fieldName);
else
    value = nan(targetSize);
end
end
