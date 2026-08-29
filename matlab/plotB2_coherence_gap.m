% plotB2_coherence_gap.m
% Read B2 coherence-gap results and export the paper figure.


clear; close all; clc;

%% 路径
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(fileparts(script_dir), 'data', 'B2');
res_file = fullfile(data_dir, 'B2_CoherenceGap_results.mat');
if ~exist(res_file, 'file'); error('请先运行 genB2_coherence_gap.m'); end
load(res_file);

%% 绘图风格
set_trans_style();
colors = trans_color_order();

%% 图形
fig = figure('Position', [100 100 900 360], 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 子图 (a) 理论谱间隙
nexttile;
plot(rho_s_vec, gamma_norm_theory, '-o', 'Color', colors(1,:), ...
    'LineWidth', 1.2, 'MarkerSize', 3.6);
format_ieee_axes(gca);
xlabel('Source correlation coefficient $\rho$', 'Interpreter', 'latex');
ylabel('Normalized eigengap');
title('(a) Eigengap', 'FontWeight', 'normal', 'FontSize', 10);

% 子图 (b) 子空间偏差
nexttile;
plot(rho_s_vec, mean_dn, '-o', 'Color', colors(4,:), ...
    'LineWidth', 1.2, 'MarkerSize', 3.6);
format_ieee_axes(gca);
xlabel('Source correlation coefficient $\rho$', 'Interpreter', 'latex');
ylabel('Projection deviation');
title('(b) Projection deviation', 'FontWeight', 'normal', 'FontSize', 10);

drawnow;
export_ieee_figure(fig, 'plotB2_coherence_gap');
fprintf('图形已显示，未保存文件。\n');

%% 辅助函数
function set_trans_style()
    set(0, 'DefaultFigureColor', 'w');
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', 11);
    set(0, 'DefaultLineLineWidth', 1.5);
    set(0, 'DefaultAxesLineWidth', 0.8);
    set(0, 'DefaultAxesBox', 'on');
    set(0, 'DefaultAxesTickDir', 'in');
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
