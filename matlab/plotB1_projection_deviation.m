% plotB1_projection_deviation.m
% Read B1 projection-deviation results and export paper figures.
% 图1：单变量对子空间偏差的影响（填充带表示标准差，子图中央加标题）
% 图2：子空间偏差 vs MUSIC RMSE（只连线，图例在左上角）
% 不保存文件，仅显示图形

clear; close all; clc;

%% 路径设置
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(fileparts(script_dir), 'data', 'B1');
res_file = fullfile(data_dir, 'B1_ProjectionDeviation_results.mat');
if ~exist(res_file, 'file')
    error('请先运行 genB1_projection_deviation.m 生成结果文件');
end
load(res_file);

%% 绘图风格
set_trans_style();
colors = trans_color_order();

%% ================= 图1：单变量影响（填充带代替误差棒） =================
fig1 = figure('Position', [100 100 900 360], 'Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) SNR
nexttile;
x = SNR_vec; y = mean_dn_snr; y_std = std_dn_snr;
x_fill = [x, fliplr(x)];
y_upper = y + y_std;
y_lower = y - y_std;
fill(x_fill, [y_upper, fliplr(y_lower)], colors(1,:), ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none');
hold on;
plot(x, y, '-', 'Color', colors(1,:), 'LineWidth', 1.2);
format_ieee_axes(gca); xlabel('SNR (dB)'); ylabel('Projection deviation');
title('(a) SNR scan', 'FontWeight', 'normal', 'FontSize', 10);

% (b) Snapshots
nexttile;
x = T_vec; y = mean_dn_T; y_std = std_dn_T;
x_fill = [x, fliplr(x)];
y_upper = y + y_std;
y_lower = y - y_std;
fill(x_fill, [y_upper, fliplr(y_lower)], colors(2,:), ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none');
hold on;
semilogx(x, y, '--', 'Color', colors(2,:), 'LineWidth', 1.2);
format_ieee_axes(gca); xlabel('Number of snapshots'); ylabel('Projection deviation');
title('(b) Snapshot scan', 'FontWeight', 'normal', 'FontSize', 10);

% (c) Array mismatch
nexttile;
x = rho_err_vec; y = mean_dn_err; y_std = std_dn_err;
x_fill = [x, fliplr(x)];
y_upper = y + y_std;
y_lower = y - y_std;
fill(x_fill, [y_upper, fliplr(y_lower)], colors(3,:), ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none');
hold on;
plot(x, y, '-.', 'Color', colors(3,:), 'LineWidth', 1.2);
format_ieee_axes(gca); xlabel('Array mismatch factor'); ylabel('Projection deviation');
title('(c) Array mismatch', 'FontWeight', 'normal', 'FontSize', 10);

export_ieee_figure(fig1, 'plotB1_projection_deviation');

%% ================= 图2：子空间偏差 vs RMSE（只连线，图例左上角） =================
fig2 = figure('Position', [100 100 900 560], 'Color', 'w');
hold on;

plot(mean_dn_snr, median_rmse_snr, '-', ...
    'Color', colors(1,:), 'LineWidth', 1.2, 'DisplayName', 'SNR');

plot(mean_dn_T, median_rmse_T, '--', ...
    'Color', colors(2,:), 'LineWidth', 1.2, 'DisplayName', 'Snapshots');

plot(mean_dn_err, median_rmse_err, '-.', ...
    'Color', colors(3,:), 'LineWidth', 1.2, 'DisplayName', 'Array mismatch');

xlabel('Projection deviation');
ylabel('MUSIC RMSE (deg)');
lgd = legend('Location', 'northwest', 'Box', 'off');
lgd.FontSize = 8;
grid on; box on;
format_ieee_axes(gca);
% 自动调整坐标范围（留5%余量）
xlim([0, max([mean_dn_snr, mean_dn_T, mean_dn_err]) * 1.05]);
ylim([0, max([median_rmse_snr, median_rmse_T, median_rmse_err]) * 1.05]);

drawnow;
export_ieee_figure(fig2, 'plotB1_projection_deviation_relation');
fprintf('图形已显示（图1、图2），未保存文件。\n');

%% ======================= 辅助函数 =======================
function set_trans_style()
    set(0, 'DefaultFigureColor', 'w');
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultAxesColorOrder', trans_color_order());
    set(0, 'DefaultAxesFontSize', 11);
    set(0, 'DefaultTextFontSize', 11);
    set(0, 'DefaultLineLineWidth', 1.5);
    set(0, 'DefaultAxesLineWidth', 0.8);
    set(0, 'DefaultAxesBox', 'on');
    set(0, 'DefaultAxesTickDir', 'in');
    set(0, 'DefaultLegendFontSize', 11);
end

function colors = trans_color_order()
    colors = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.4660 0.6740 0.1880; ...
              0.4940 0.1840 0.5560; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840];
end

function format_trans_axes(ax)
    grid(ax, 'on'); box(ax, 'on');
    ax.GridColor = [0.85 0.85 0.85]; ax.GridAlpha = 0.45;
    ax.LineWidth = 0.8; ax.FontSize = 11; ax.TickDir = 'in';
end
