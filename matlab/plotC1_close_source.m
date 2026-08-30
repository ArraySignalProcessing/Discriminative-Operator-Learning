% plotC1_close_source.m
% 绘制 C1：RMSE vs Δθ + PoR vs Δθ（双面板）
% 自动检测所有可用结果文件并绘制，输出统一表格。
% 作者: D
% 日期: 2026-05-09 (revised 2026-05-09)

clear all; close all;

% ================= 路径 =================
data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C1');

% ================= Display switches =================
show_MUSIC       = true;
show_RootMUSIC   = true;
show_ESPRIT      = true;
show_l1SVD       = false;
show_UnESPRIT    = false;
show_CNN         = true;
show_SubspaceNet = true;
show_Ours        = true;
show_RootOurs    = false;

% ================= 加载传统基线 =================
baseline_file = fullfile(data_dir, 'C1_Baselines.mat');
if ~exist(baseline_file, 'file')
    error('找不到基线结果: %s', baseline_file);
end
load(baseline_file, 'delta_theta_vec', 'RMSE', 'PoR');

% 确保 delta_theta_vec 是列向量
delta_theta_vec = delta_theta_vec(:);

% 提取基线
MUSIC_rmse    = RMSE.MUSIC(:);
RMUSIC_rmse   = RMSE.RootMUSIC(:);
ESPRIT_rmse   = RMSE.ESPRIT(:);
l1SVD_rmse    = RMSE.l1SVD(:);
UnESPRIT_rmse = RMSE.UnESPRIT(:);
MUSIC_por     = PoR.MUSIC(:) * 100;
RMUSIC_por    = PoR.RootMUSIC(:) * 100;
ESPRIT_por    = PoR.ESPRIT(:) * 100;
l1SVD_por     = PoR.l1SVD(:) * 100;
UnESPRIT_por  = PoR.UnESPRIT(:) * 100;

% ================= 尝试加载 CNN 结果 =================
cnn_file = fullfile(data_dir, 'C1_CNN.h5');
CNN_rmse = nan(size(delta_theta_vec));
CNN_por  = nan(size(delta_theta_vec));
if exist(cnn_file, 'file')
    try
        CNN_rmse = h5read(cnn_file, '/CNN_RMSE');
        CNN_rmse = CNN_rmse(:);  % 确保是列向量
    catch
    end
    try
        CNN_por = h5read(cnn_file, '/CNN_PoR');
        CNN_por = CNN_por(:) * 100;  % 确保是列向量，转换为百分比
    catch
    end
end

% ================= 尝试加载 Ours 结果 =================
ours_file = fullfile(data_dir, 'C1_Ours.h5');
Ours_rmse = nan(size(delta_theta_vec));
Ours_por  = nan(size(delta_theta_vec));
if exist(ours_file, 'file')
    try
        Ours_rmse = h5read(ours_file, '/Ours_RMSE');
        Ours_rmse = Ours_rmse(:);  % 确保是列向量
    catch
    end
    try
        Ours_por = h5read(ours_file, '/Ours_PoR');
        Ours_por = Ours_por(:) * 100;  % 确保是列向量，转换为百分比
    catch
    end
end
% ================= 尝试加载 Root-Ours 结果 =================
root_ours_file = fullfile(data_dir, 'C1_RootOurs.h5');
RootOurs_rmse = nan(size(delta_theta_vec));
RootOurs_por  = nan(size(delta_theta_vec));
if exist(root_ours_file, 'file')
    try
        RootOurs_rmse = h5read(root_ours_file, '/RootOurs_RMSE');
        RootOurs_rmse = RootOurs_rmse(:);
    catch
    end
    try
        RootOurs_por = h5read(root_ours_file, '/RootOurs_PoR');
        RootOurs_por = RootOurs_por(:) * 100;
    catch
    end
end

% ================= 尝试加载 SubspaceNet 结果 =================
subspace_file = fullfile(data_dir, 'C1_SubspaceNet.h5');
SubspaceNet_rmse = nan(size(delta_theta_vec));
SubspaceNet_por  = nan(size(delta_theta_vec));
if exist(subspace_file, 'file')
    try
        SubspaceNet_rmse = h5read(subspace_file, '/SubspaceNet_RMSE');
        SubspaceNet_rmse = SubspaceNet_rmse(:);
    catch
    end
    try
        SubspaceNet_por = h5read(subspace_file, '/SubspaceNet_PoR');
        SubspaceNet_por = SubspaceNet_por(:) * 100;
    catch
    end
end

% ================= 构建动态方法列表 =================
allMethods = {'MUSIC','Root-MUSIC','ESPRIT','l1-SVD','UnESPRIT','MLC','SubspaceNet','Ours','Root-Ours'};
rmse_data = {MUSIC_rmse, RMUSIC_rmse, ESPRIT_rmse, l1SVD_rmse, UnESPRIT_rmse, CNN_rmse, SubspaceNet_rmse, Ours_rmse, RootOurs_rmse};
por_data  = {MUSIC_por, RMUSIC_por, ESPRIT_por, l1SVD_por, UnESPRIT_por, CNN_por, SubspaceNet_por, Ours_por, RootOurs_por};
show_data = [show_MUSIC, show_RootMUSIC, show_ESPRIT, show_l1SVD, show_UnESPRIT, ...
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
        methodPOR(:,end+1) = p;
    end
end

% 输出表格
Tab_RMSE = array2table([delta_theta_vec(:), methodRMSE], ...
    'VariableNames', ['DeltaTheta', methodNames]);
fprintf('=== C1 RMSE 表格 ===\n');
disp(Tab_RMSE);
if ~isempty(methodPOR) && any(~all(isnan(methodPOR)))
    Tab_PoR = array2table([delta_theta_vec(:), methodPOR], ...
        'VariableNames', ['DeltaTheta', methodNames]);
    fprintf('=== C1 PoR (%% ) 表格 ===\n');
    disp(Tab_PoR);
end

% ================= 绘图：双面板 =================
fig = figure('Position', [100 100 900 560], 'Color', 'w');
x_plot = 1:length(delta_theta_vec);
x_tick_labels = arrayfun(@(x) sprintf('%g', x), delta_theta_vec, 'UniformOutput', false);

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

% 上：RMSE
subplot(2,1,1);
hold on; grid on;
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
set(gca, 'XTick', x_plot, 'XTickLabel', x_tick_labels);
set(gca, 'YScale', 'log');
valid_rmse = methodRMSE(isfinite(methodRMSE) & methodRMSE > 0);
if ~isempty(valid_rmse)
    ymin = 10 ^ floor(log10(min(valid_rmse)));
    ymax = 10 ^ ceil(log10(max(valid_rmse)));
    ylim([ymin, ymax]);
end
xlabel('Angular separation $\Delta\theta$ (deg)', 'Interpreter', 'latex');
ylabel('RMSE (deg)', 'Interpreter', 'latex');
title('(a) RMSE', 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
lgd = legend('Location', 'northoutside', 'Orientation', 'horizontal', ...
    'NumColumns', 3, 'Interpreter', 'latex');
lgd.FontSize = 8;
format_ieee_axes(gca);
hold off;

% 下：PoR
subplot(2,1,2);
hold on; grid on;
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
set(gca, 'XTick', x_plot, 'XTickLabel', x_tick_labels);
xlabel('Angular separation $\Delta\theta$ (deg)', 'Interpreter', 'latex');
ylabel('Probability of Resolution (\%)', 'Interpreter', 'latex');
title('(b) PoR', 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
ylim([0 105]);
format_ieee_axes(gca);
hold off;

export_ieee_figure(fig, 'plotC1_close_source');

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
