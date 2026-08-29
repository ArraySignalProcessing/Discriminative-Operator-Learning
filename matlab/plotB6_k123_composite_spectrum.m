% plotB6_k123_composite_spectrum.m
% Plot B6: K=3 learned-Q spectrum generalization on K=1/2/3 composite data.

clear; close all;

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B6');
K_vec = 1:3;

% Existing Ours grid tests use a sign flip during DOA readout, so the default
% applies the same convention for learned-Q spectra.
flip_Q_angle_axis = true;

slice_theta_vec = [-30, 0, 30];  % K=1 true-DOA slices
slice_delta_k2 = [5, 15, 30];    % K=2 separation slices
slice_delta_k3 = [8, 15, 20];    % K=3 adjacent-spacing slices

for kk = 1:numel(K_vec)
    K = K_vec(kk);
    data_file = fullfile(data_dir, sprintf('B6_K123_CompositeSpectrum_QPn_K%d.h5', K));
    q_file = fullfile(data_dir, sprintf('B6_K123_CompositeSpectrum_Q_K%d.h5', K));

    if ~exist(data_file, 'file')
        error('Missing B6 data file: %s', data_file);
    end
    if ~exist(q_file, 'file')
        error('Missing B6 Q file: %s', q_file);
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
    elseif K == 2
        scan_vec = h5read(data_file, '/delta_theta_vec');
        scan_vec = scan_vec(:).';
        y_label = 'Angular separation $\Delta\theta$ (deg)';
        slice_vec = slice_delta_k2;
        slice_label_prefix = '\Delta\theta';
        group_fun = @make_delta_group;
    else
        scan_vec = h5read(data_file, '/delta_theta_vec');
        scan_vec = scan_vec(:).';
        y_label = 'Adjacent spacing $\Delta\theta$ (deg)';
        slice_vec = slice_delta_k3;
        slice_label_prefix = '\Delta\theta';
        group_fun = @make_delta_group;
    end

    if any(slice_vec < min(scan_vec)) || any(slice_vec > max(scan_vec))
        error('Requested B6 K=%d slice(s) %s exceed generated scan range [%.1f, %.1f].', ...
            K, mat2str(slice_vec), min(scan_vec), max(scan_vec));
    end

    A_scan_nominal = steering_matrix(angle_scan, M);
    A_scan_Q = steering_matrix(angle_scan * ternary(flip_Q_angle_axis, -1, 1), M);
    n_scan = numel(scan_vec);
    n_angle = numel(angle_scan);
    spec_pn = zeros(n_scan, n_angle);
    spec_q = zeros(n_scan, n_angle);

    for ii = 1:n_scan
        grp = group_fun(scan_vec(ii));
        Pn = read_complex_stack(data_file, [grp '/Pn_real'], [grp '/Pn_imag'], false);
        Q = read_complex_stack(q_file, [grp '/Q_real'], [grp '/Q_imag'], true);
        spec_pn(ii, :) = mean_spectrum(Pn, A_scan_nominal);
        spec_q(ii, :) = mean_spectrum(Q, A_scan_Q);
    end

    spec_pn_norm = row_normalize(spec_pn);
    spec_q_norm = row_normalize(spec_q);

    slice_idx = zeros(1, numel(slice_vec));
    slice_actual = zeros(1, numel(slice_vec));
    for ss = 1:numel(slice_vec)
        [~, slice_idx(ss)] = min(abs(scan_vec - slice_vec(ss)));
        slice_actual(ss) = scan_vec(slice_idx(ss));
    end

    fprintf('=== B6 selected spectrum slices: K=%d ===\n', K);
    fprintf('Slices: %s = %s deg\n', slice_label_prefix, mat2str(slice_actual, 3));

    fig = figure('Position', [80 80 1120 760], 'Color', 'w');
    tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plot_contour_panel(angle_scan, scan_vec, spec_pn_norm, theta_center, K, y_label, '(a) EVD-$P_n$');

    nexttile;
    plot_contour_panel(angle_scan, scan_vec, spec_q_norm, theta_center, K, y_label, '(b) K=3 model $Q$');

    nexttile;
    [pn_handles, pn_labels] = plot_slice_panel(angle_scan, spec_pn_norm(slice_idx, :), ...
        theta_center, K, slice_actual, slice_label_prefix, '(c) EVD-$P_n$ slices');

    nexttile;
    plot_slice_panel(angle_scan, spec_q_norm(slice_idx, :), ...
        theta_center, K, slice_actual, slice_label_prefix, '(d) K=3 model $Q$ slices');

    lgd = legend(pn_handles, pn_labels, 'Interpreter', 'latex', 'Orientation', 'horizontal');
    lgd.FontSize = 8;
    lgd.Layout.Tile = 'south';
    title(tl, sprintf('B6 composite robustness spectrum, tested on $K=%d$', K), ...
        'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);

    export_ieee_figure(fig, sprintf('plotB6_k123_composite_spectrum_K%d', K));
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
    elseif K == 2
        theta_vec = [theta_center - scan_value / 2, theta_center + scan_value / 2];
    else
        theta_vec = [theta_center - scan_value, theta_center, theta_center + scan_value];
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
            theta_track = true_angle_track(theta_center, K, scan_vec, src);
            plot(theta_track, scan_vec, 'k--', 'LineWidth', 1.2);
        end
    end
    hold off;
    xlim([min(angle_scan), max(angle_scan)]);
    xlabel('Angle $\theta$ (deg)', 'Interpreter', 'latex');
    ylabel(y_label, 'Interpreter', 'latex');
    title(panel_title, 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
    format_ieee_axes(gca);
end

function theta_track = true_angle_track(theta_center, K, scan_vec, src)
    if K == 2
        coeff = [-0.5, 0.5];
        theta_track = theta_center + coeff(src) * scan_vec;
    else
        coeff = [-1, 0, 1];
        theta_track = theta_center + coeff(src) * scan_vec;
    end
end

function [line_handles, line_labels] = plot_slice_panel(angle_scan, S_slice, theta_center, K, slice_actual, slice_label_prefix, panel_title)
    if isvector(S_slice)
        S_slice = S_slice(:).';
    end
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
