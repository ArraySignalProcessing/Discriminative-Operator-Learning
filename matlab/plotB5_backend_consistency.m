% plotB5_backend_consistency.m
% Plot B5: grid-to-root convergence of the learned Q.

clear; close all;

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B5');
fname = fullfile(data_dir, 'B5_BackendConvergence.h5');
if ~exist(fname, 'file')
    error('Missing convergence result: %s', fname);
end

grid_steps = h5read(fname, '/grid_steps');
g2r_rmse = h5read(fname, '/G2R_RMSE');
root_est = normalize_estimates(h5read(fname, '/Q_root_DOA'));

K = size(root_est, 2);
root_mean = mean(root_est, 1);
grid_mean = zeros(length(grid_steps), K);

for ii = 1:length(grid_steps)
    ds = ['/Q_grid_DOA/' make_step_tag(grid_steps(ii))];
    grid_est = normalize_estimates(h5read(fname, ds));
    grid_mean(ii, :) = mean(grid_est, 1);
end

fprintf('=== B5 backend convergence ===\n');
disp(table(grid_steps(:), g2r_rmse(:), ...
    'VariableNames', {'GridStep_deg', 'GridToRoot_RMSE_deg'}));

fig = figure('Position', [100 100 760 560], 'Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
plot(grid_steps, grid_mean(:, 1), '-o', 'LineWidth', 1.2, 'MarkerSize', 3.6);
plot(grid_steps, grid_mean(:, 2), '-s', 'LineWidth', 1.2, 'MarkerSize', 3.6);
yline(root_mean(1), '--', 'LineWidth', 1.0);
yline(root_mean(2), '--', 'LineWidth', 1.0);
hold off;
ylim([-30, 30]);
set(gca, 'XScale', 'log', 'XDir', 'reverse');
set(gca, 'XTick', sort(grid_steps));
xtickformat('%.3g');
grid on;
xlabel('Grid step $\Delta_g$ (deg)', 'Interpreter', 'latex');
ylabel('Estimated DOA (deg)', 'Interpreter', 'latex');
title('(a) DOA estimates', 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
legend({'$Q$-grid $\hat{\theta}_1$', '$Q$-grid $\hat{\theta}_2$', ...
    '$Q$-root $\hat{\theta}_1$', '$Q$-root $\hat{\theta}_2$'}, ...
    'Interpreter', 'latex', 'Location', 'best', 'FontSize', 8);
format_ieee_axes(gca);

nexttile;
plot(grid_steps, g2r_rmse, '-o', 'LineWidth', 1.2, 'MarkerSize', 3.6);
ylim([0, 1.4]);
set(gca, 'XScale', 'log', 'XDir', 'reverse');
set(gca, 'XTick', sort(grid_steps));
xtickformat('%.3g');
grid on;
xlabel('Grid step $\Delta_g$ (deg)', 'Interpreter', 'latex');
ylabel('Grid-root RMSE discrepancy (deg)', 'Interpreter', 'latex');
title('(b) Grid-root discrepancy', 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
format_ieee_axes(gca);

export_ieee_figure(fig, 'plotB5_backend_consistency');

function est = normalize_estimates(est)
    if size(est, 2) ~= 2 && size(est, 1) == 2
        est = est.';
    end
    est = sort(est, 2);
end

function tag = make_step_tag(step)
    label = regexprep(sprintf('%.6f', step), '0+$', '');
    label = regexprep(label, '\.$', '');
    tag = ['step_' label];
    tag = strrep(tag, '.', 'p');
end
