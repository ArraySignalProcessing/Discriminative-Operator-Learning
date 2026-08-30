% plotC2_coherent_sources.m
% 绘制 C2：相干源 RMSE vs rho + PoR vs rho（双面板），SNR 固定
% 自动检测所有可用结果文件并绘制。
% 作者: D
% 日期: 2026-05-09 (revised 2026-05-09)

clear all; close all;

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C2');

% ================= 显示开关 =================
show_MUSIC       = true;
show_RootMUSIC   = false;
show_ESPRIT      = true;
show_TRMUSIC     = true;
show_SSMUSIC     = false;
show_SSESPRIT    = false;
show_l1SVD       = false;
show_UnESPRIT    = false;
show_CNN         = true;
show_SubspaceNet = true;
show_Ours        = true;
show_RootOurs    = false;

% ================= 加载传统基线 =================
baseline_candidates = {fullfile(data_dir, 'C2_Baselines.mat')};
baseline_file = '';
for i = 1:numel(baseline_candidates)
    if exist(baseline_candidates{i}, 'file')
        baseline_file = baseline_candidates{i};
        break;
    end
end
if isempty(baseline_file)
    error('找不到基线结果: %s', baseline_candidates{1});
end
load(baseline_file);

if ~exist('SNR_dB', 'var')
    error('基线结果中缺少 SNR_dB，无法核验 C2 绘图参数。');
end
fixed_snr_db = SNR_dB;

if exist('rho_vec', 'var')
    x_vec = rho_vec;
    x_name = 'rho';
elseif exist('SNR_vec', 'var')
    x_vec = SNR_vec;
    x_name = 'SNR';
else
    error('基线文件中既没有 rho_vec 也没有 SNR_vec。');
end

% 提取基线
MUSIC_rmse    = RMSE.MUSIC;
RMUSIC_rmse   = RMSE.RootMUSIC;
ESPRIT_rmse   = RMSE.ESPRIT;
TRMUSIC_rmse  = get_struct_vector(RMSE, 'TRMUSIC', size(x_vec));
SSMUSIC_rmse  = RMSE.SSMUSIC;
SSESPRIT_rmse = RMSE.SSESPRIT;
l1SVD_rmse    = RMSE.l1SVD;
UnESPRIT_rmse = RMSE.UnESPRIT;
MUSIC_por     = PoR.MUSIC * 100;
RMUSIC_por    = PoR.RootMUSIC * 100;
ESPRIT_por    = PoR.ESPRIT * 100;
TRMUSIC_por   = get_struct_vector(PoR, 'TRMUSIC', size(x_vec)) * 100;
SSMUSIC_por   = PoR.SSMUSIC * 100;
SSESPRIT_por  = PoR.SSESPRIT * 100;
l1SVD_por     = PoR.l1SVD * 100;
UnESPRIT_por  = PoR.UnESPRIT * 100;

% ================= 尝试加载 CNN 结果 =================
cnn_candidates = {fullfile(data_dir, 'C2_CNN.h5')};
cnn_file = '';
for i = 1:numel(cnn_candidates)
    if exist(cnn_candidates{i}, 'file')
        cnn_file = cnn_candidates{i};
        break;
    end
end
CNN_rmse = nan(size(x_vec));
CNN_por  = nan(size(x_vec));
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
ours_candidates = {fullfile(data_dir, 'C2_Ours.h5')};
ours_file = '';
for i = 1:numel(ours_candidates)
    if exist(ours_candidates{i}, 'file')
        ours_file = ours_candidates{i};
        break;
    end
end
Ours_rmse = nan(size(x_vec));
Ours_por  = nan(size(x_vec));
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
root_ours_candidates = {fullfile(data_dir, 'C2_RootOurs.h5')};
root_ours_file = '';
for i = 1:numel(root_ours_candidates)
    if exist(root_ours_candidates{i}, 'file')
        root_ours_file = root_ours_candidates{i};
        break;
    end
end
RootOurs_rmse = nan(size(x_vec));
RootOurs_por  = nan(size(x_vec));
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

subspace_candidates = {fullfile(data_dir, 'C2_SubspaceNet.h5')};
subspace_file = '';
for i = 1:numel(subspace_candidates)
    if exist(subspace_candidates{i}, 'file')
        subspace_file = subspace_candidates{i};
        break;
    end
end
SubspaceNet_rmse = nan(size(x_vec));
SubspaceNet_por  = nan(size(x_vec));
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
allMethods = {'MUSIC','Root-MUSIC','ESPRIT','TR-MUSIC','SS-MUSIC','SS-ESPRIT','l1-SVD','UnESPRIT','MLC','SubspaceNet','Ours','Root-Ours'};
rmse_data = {MUSIC_rmse, RMUSIC_rmse, ESPRIT_rmse, TRMUSIC_rmse, SSMUSIC_rmse, SSESPRIT_rmse, l1SVD_rmse, UnESPRIT_rmse, CNN_rmse, SubspaceNet_rmse, Ours_rmse, RootOurs_rmse};
por_data  = {MUSIC_por, RMUSIC_por, ESPRIT_por, TRMUSIC_por, SSMUSIC_por, SSESPRIT_por, l1SVD_por, UnESPRIT_por, CNN_por, SubspaceNet_por, Ours_por, RootOurs_por};
show_data = [show_MUSIC, show_RootMUSIC, show_ESPRIT, show_TRMUSIC, show_SSMUSIC, show_SSESPRIT, ...
             show_l1SVD, show_UnESPRIT, show_CNN, show_SubspaceNet, show_Ours, show_RootOurs];

methodNames = {};
methodRMSE  = [];
methodPOR   = [];

for i = 1:length(allMethods)
    r = rmse_data{i}(:);
    p = por_data{i}(:);
    if show_data(i) && ~all(isnan(r))
        methodNames{end+1} = allMethods{i};
        methodRMSE(:,end+1) = r;
        methodPOR(:,end+1) = p;
    end
end

% 输出表格
Tab_RMSE = array2table([x_vec(:), methodRMSE], ...
    'VariableNames', [x_name, methodNames]);
fprintf('=== C2 RMSE 表格 (SNR=%d dB) ===\n', fixed_snr_db);
disp(Tab_RMSE);
if ~isempty(methodPOR) && any(~all(isnan(methodPOR)))
    Tab_PoR = array2table([x_vec(:), methodPOR], ...
        'VariableNames', [x_name, methodNames]);
    fprintf('=== C2 PoR (%%) 表格 (SNR=%d dB) ===\n', fixed_snr_db);
    disp(Tab_PoR);
end

% ================= 绘图：双面板 =================
fig = figure('Position', [100 100 900 560], 'Color', 'w');
x_plot = 1:numel(x_vec);
x_labels = arrayfun(@(v) sprintf('%.2g', v), x_vec, 'UniformOutput', false);

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

% Top: RMSE
subplot(2,1,1);
hold on; grid on;
valid_rmse = methodRMSE(isfinite(methodRMSE) & methodRMSE > 0);
if ~isempty(valid_rmse)
    ymin = 10 ^ floor(log10(min(valid_rmse)));
    ymax = 10 ^ ceil(log10(max(valid_rmse)));
else
    ymin = 1e-1;
    ymax = 1e2;
end
patch([2.5 max(x_plot)+0.2 max(x_plot)+0.2 2.5], [ymin ymin ymax ymax], ...
    [0.95 0.95 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
    'HandleVisibility', 'off');
for i = 1:length(methodNames)
    plot(x_plot, methodRMSE(:,i), ...
        'LineStyle', methodLineStyle{i}, ...
        'Marker', methodMarker{i}, ...
        'LineWidth', 1.2, ...
        'MarkerSize', 3.6, ...
        'Color', methodColor{i}, ...
        'DisplayName', methodNames{i});
end
xlim([min(x_plot), max(x_plot)]);
set(gca, 'XTick', x_plot, 'XTickLabel', x_labels);
set(gca, 'YScale', 'log');
ylim([ymin, ymax]);
ylabel('RMSE (deg)', 'Interpreter', 'latex');
title('(a) RMSE', 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
lgd = legend('Location', 'northoutside', 'Orientation', 'horizontal', ...
    'NumColumns', 3, 'Interpreter', 'latex');
lgd.FontSize = 8;
format_ieee_axes(gca);

hold off;

% Bottom: PoR
ax_por = subplot(2,1,2);
hold on; grid on;
patch([2.5 max(x_plot)+0.2 max(x_plot)+0.2 2.5], [0 0 105 105], ...
    [0.95 0.95 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
    'HandleVisibility', 'off');
for i = 1:length(methodNames)
    if ~all(isnan(methodPOR(:,i)))
        plot(x_plot, methodPOR(:,i), ...
            'LineStyle', methodLineStyle{i}, ...
            'Marker', methodMarker{i}, ...
            'LineWidth', 1.2, ...
            'MarkerSize', 3.6, ...
            'Color', methodColor{i}, ...
            'DisplayName', methodNames{i});
    end
end
xlim([min(x_plot), max(x_plot)]);
set(gca, 'XTick', x_plot, 'XTickLabel', x_labels);
xlabel('$\rho$', 'Interpreter', 'latex');
ylabel('PoR (\%)', 'Interpreter', 'latex');
title('(b) PoR', 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
ylim([0 105]);
format_ieee_axes(gca);
hold off;

export_ieee_figure(fig, 'plotC2_coherent_sources');

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
    case {'TR-MUSIC'}
        color = [0.93 0.69 0.13]; lineStyle = '-'; marker = 'v';
    case {'MLC'}
        color = baselineColor; lineStyle = '--'; marker = 's';
    case {'SubspaceNet'}
        color = [0.00 0.60 0.50]; lineStyle = '--'; marker = 'd';
    case {'Ours'}
        color = oursColor; lineStyle = '--'; marker = '^';
    case {'Root-Ours'}
        color = [0.64 0.08 0.18]; lineStyle = '--'; marker = 'v';
    otherwise
        color = defaultColor; lineStyle = '-'; marker = 'o';
end
end

function value = get_struct_vector(S, fieldName, targetSize)
if isfield(S, fieldName)
    value = S.(fieldName);
else
    value = nan(targetSize);
end
end
