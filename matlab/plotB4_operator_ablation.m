% plot_B4.m
% Plot B4: construction-criterion ablation for K=1 and K=2.

clear; close all;

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B4');
K_vec = 1:2;

% Existing Ours grid tests use a sign flip during DOA readout, so the default
% applies the same convention for learned-Q spectra.
flip_Q_angle_axis = true;

% ================= Slice selection for lower panels =================
slice_theta_vec = [-30, 0, 30];   % K=1: true DOA slices
slice_delta_vec = [5, 15, 30];     % K=2: angular-separation slices

methods = {'projector', 'null_only', 'full'};
method_titles = { ...
    'Projector fitting', ...
    'Null-space only', ...
    'Full objective'};
n_method = numel(methods);

for kk = 1:numel(K_vec)
    K = K_vec(kk);
    data_file = fullfile(data_dir, sprintf('B4_OperatorAblation_QPn_K%d.h5', K));
    q_file = fullfile(data_dir, sprintf('B4_Ablation_Q_K%d.h5', K));

    if ~exist(data_file, 'file')
        error('Missing B4 data file: %s', data_file);
    end
    if ~exist(q_file, 'file')
        error('Missing B4 Q file: %s', q_file);
    end

    angle_scan = h5read(data_file, '/angle_scan');
    angle_scan = angle_scan(:).';
    M = h5readatt(data_file, '/', 'M');
    theta_center = h5readatt(data_file, '/', 'theta_center');

    if K == 1
        scan_vec = h5read(data_file, '/theta_vec');
        scan_vec = scan_vec(:).';
        y_label = 'True angle $\theta_0$ (deg)';
        slice_vec = slice_theta_vec;
        slice_label_prefix = '\theta_0';
        group_fun = @make_theta_group;
    else
        scan_vec = h5read(data_file, '/delta_theta_vec');
        scan_vec = scan_vec(:).';
        y_label = 'Angular separation $\Delta\theta$ (deg)';
        slice_vec = slice_delta_vec;
        slice_label_prefix = '\Delta\theta';
        group_fun = @make_delta_group;
    end

    if any(slice_vec < min(scan_vec)) || any(slice_vec > max(scan_vec))
        error('Requested B4 K=%d slice(s) %s exceed generated scan range [%.1f, %.1f]. Regenerate the test data with a wider scan range.', ...
            K, mat2str(slice_vec), min(scan_vec), max(scan_vec));
    end

    A_scan_Q = steering_matrix(angle_scan * ternary(flip_Q_angle_axis, -1, 1), M);
    n_scan = numel(scan_vec);
    n_angle = numel(angle_scan);
    spec = zeros(n_method, n_scan, n_angle);

    for ii = 1:n_scan
        grp = group_fun(scan_vec(ii));

        for mm = 1:n_method
            method = methods{mm};
            q_real_path = sprintf('%s/%s/Q_real', grp, method);
            q_imag_path = sprintf('%s/%s/Q_imag', grp, method);
            Q = read_complex_stack(q_file, q_real_path, q_imag_path, true);
            spec(mm, ii, :) = mean_spectrum(Q, A_scan_Q);
        end
    end

    spec_norm = zeros(size(spec));
    for mm = 1:n_method
        spec_norm(mm, :, :) = row_normalize(squeeze(spec(mm, :, :)));
    end

    slice_idx = zeros(1, numel(slice_vec));
    slice_actual = zeros(1, numel(slice_vec));
    for ss = 1:numel(slice_vec)
        [~, slice_idx(ss)] = min(abs(scan_vec - slice_vec(ss)));
        slice_actual(ss) = scan_vec(slice_idx(ss));
    end

    fprintf('=== B4 selected spectrum slices: K=%d ===\n', K);
    if K == 1
        fprintf('Slices: theta0 = %s deg\n', mat2str(slice_actual, 3));
    else
        fprintf('Slices: Delta theta = %s deg\n', mat2str(slice_actual, 3));
    end

    fig = figure('Position', [80 80 1320 720], 'Color', 'w');
    tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for mm = 1:n_method
        nexttile;
        panel_title = sprintf('(%s) %s', char('a' + mm - 1), method_titles{mm});
        plot_contour_panel(angle_scan, scan_vec, squeeze(spec_norm(mm, :, :)), ...
            theta_center, K, y_label, panel_title);
    end

    legend_handles = [];
    legend_labels = {};
    for mm = 1:n_method
        nexttile;
        panel_title = sprintf('(%s) %s', char('d' + mm - 1), method_titles{mm});
        S_slice = squeeze(spec_norm(mm, slice_idx, :));
        if numel(slice_idx) == 1
            S_slice = S_slice.';
        end
        [line_handles, line_labels] = plot_slice_panel(angle_scan, S_slice, theta_center, K, slice_actual, ...
            slice_label_prefix, panel_title);
        if mm == 1
            legend_handles = line_handles;
            legend_labels = line_labels;
        end
    end

    lgd = legend(legend_handles, legend_labels, ...
        'Interpreter', 'latex', 'Orientation', 'horizontal');
    lgd.FontSize = 8;
    lgd.Layout.Tile = 'south';
    title(tl, sprintf('Operator-ablation response, $K=%d$', K), ...
        'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);

    export_ieee_figure(fig, sprintf('plotB4_operator_ablation_K%d', K));
end

function A = steering_matrix(angle_grid, M)
    theta = deg2rad(angle_grid(:));
    pos = (0:M-1) * 0.5;
    A = exp(1j * 2 * pi * sin(theta) * pos);
end

function y = ternary(cond, a, b)
    if cond
        y = a;
    else
        y = b;
    end
end

function grp = make_theta_group(theta)
    if theta < 0
        prefix = 'min';
    else
        prefix = '';
    end
    tag = strrep(sprintf('%.1f', abs(theta)), '.', 'p');
    grp = sprintf('/Theta_%s%sdeg', prefix, tag);
end

function grp = make_delta_group(delta)
    tag = strrep(sprintf('%.1f', delta), '.', 'p');
    grp = sprintf('/Delta_%sdeg', tag);
end

function X = read_complex_stack(fname, real_path, imag_path, transpose_slices)
    Xr = h5read(fname, real_path);
    Xi = h5read(fname, imag_path);
    X = double(Xr) + 1j * double(Xi);
    if ndims(X) ~= 3
        error('Expected complex stack with shape M x M x N at %s', real_path);
    end
    if transpose_slices
        for mc = 1:size(X, 3)
            X(:, :, mc) = X(:, :, mc).';
        end
    end
end

function spec = mean_spectrum(Mstack, A)
    n_mc = size(Mstack, 3);
    n_angle = size(A, 1);
    spec_all = zeros(n_mc, n_angle);
    Mdim = size(Mstack, 1);
    for mc = 1:n_mc
        B = Mstack(:, :, mc);
        q = real(sum(conj(A) .* (A * B.'), 2)) / Mdim;
        q = max(q, 0);
        spec_all(mc, :) = 1 ./ (q.' + 1e-6);
    end
    spec = mean(spec_all, 1);
end

function out = row_normalize(S)
    out = zeros(size(S));
    for ii = 1:size(S, 1)
        row = S(ii, :);
        row = row - min(row);
        denom = max(row);
        if denom > 0
            out(ii, :) = row / denom;
        end
    end
end

function theta_vec = true_angles(theta_center, K, scan_value)
    if K == 1
        theta_vec = scan_value;
    else
        offsets = ((1:K) - (K + 1) / 2) * scan_value;
        theta_vec = theta_center + offsets;
    end
end

function plot_contour_panel(angle_scan, scan_vec, S, theta_center, K, y_label, panel_title)
    imagesc(angle_scan, scan_vec, S);
    set(gca, 'YDir', 'normal');
    colormap(gca, parula);
    cb = colorbar;
    cb.FontSize = 8;
    hold on;
    if K == 1
        plot(scan_vec, scan_vec, 'k--', 'LineWidth', 1.2);
    else
        for src = 1:K
            offsets = (src - (K + 1) / 2) * scan_vec;
            plot(theta_center + offsets, scan_vec, 'k--', 'LineWidth', 1.2);
        end
    end
    hold off;
    xlim([min(angle_scan), max(angle_scan)]);
    xlabel('Angle $\theta$ (deg)', 'Interpreter', 'latex');
    ylabel(y_label, 'Interpreter', 'latex');
    title(panel_title, 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
    format_ieee_axes(gca);
end

function [line_handles, line_labels] = plot_slice_panel(angle_scan, S_slice, theta_center, K, slice_actual, slice_label_prefix, panel_title)
    colors = lines(numel(slice_actual));
    line_handles = gobjects(1, numel(slice_actual));
    line_labels = cell(1, numel(slice_actual));
    hold on; grid on;
    for ss = 1:numel(slice_actual)
        line_labels{ss} = sprintf('$%s=%.1f^\\circ$', slice_label_prefix, slice_actual(ss));
        line_handles(ss) = plot(angle_scan, S_slice(ss, :), '-', 'LineWidth', 1.2, ...
            'Color', colors(ss, :), ...
            'DisplayName', line_labels{ss});
        theta_vec = true_angles(theta_center, K, slice_actual(ss));
        for ii = 1:numel(theta_vec)
            xline(theta_vec(ii), ':', 'Color', colors(ss, :), ...
                'LineWidth', 0.9, 'HandleVisibility', 'off');
        end
    end
    hold off;
    xlim([min(angle_scan), max(angle_scan)]);
    ylim([0, 1.05]);
    xlabel('Angle $\theta$ (deg)', 'Interpreter', 'latex');
    ylabel('Normalized spectrum', 'Interpreter', 'latex');
    title(panel_title, 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
    format_ieee_axes(gca);
end
