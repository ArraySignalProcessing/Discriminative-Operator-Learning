% plotB3_operator_response.m
% Plot B3: spectrum-structure comparison between EVD Pn and learned Q.

clear; close all;

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B3');
data_file = fullfile(data_dir, 'B3_OperatorResponse_QPn.h5');
q_file = fullfile(data_dir, 'B3_OperatorResponse_Q.h5');

if ~exist(data_file, 'file')
    error('Missing B3 data file: %s', data_file);
end
if ~exist(q_file, 'file')
    error('Missing B3 Q file: %s', q_file);
end

delta_theta_vec = h5read(data_file, '/delta_theta_vec');
angle_scan = h5read(data_file, '/angle_scan');
delta_theta_vec = delta_theta_vec(:).';
angle_scan = angle_scan(:).';

M = h5readatt(data_file, '/', 'M');
theta_center = h5readatt(data_file, '/', 'theta_center');

% If the learned-Q spectrum appears mirrored relative to the true DOA
% trajectories, set this to true. Existing Ours grid tests use a sign flip
% during DOA readout, so the default applies the same convention for spectra.
flip_Q_angle_axis = true;

% ================= Slice selection for lower panels =================
% Subplots (c) and (d) show 1-D spectrum slices at two selected angular
% separations. The nearest available separations in delta_theta_vec are used.
% Choose one difficult near-angle slice and one milder slice.
slice_delta_small = 5;
slice_delta_mid = 13;

A_scan_nominal = steering_matrix(angle_scan, M);
A_scan_Q = steering_matrix(angle_scan * ternary(flip_Q_angle_axis, -1, 1), M);

n_delta = numel(delta_theta_vec);
n_angle = numel(angle_scan);
spec_pn = zeros(n_delta, n_angle);
spec_q = zeros(n_delta, n_angle);

for ii = 1:n_delta
    Delta = delta_theta_vec(ii);
    grp = make_delta_group(Delta);

    Pn = read_complex_stack(data_file, [grp '/Pn_real'], [grp '/Pn_imag'], false);
    Q = read_complex_stack(q_file, [grp '/Q_real'], [grp '/Q_imag'], true);

    spec_pn(ii, :) = mean_spectrum(Pn, A_scan_nominal);
    spec_q(ii, :) = mean_spectrum(Q, A_scan_Q);
end

spec_pn_norm = row_normalize(spec_pn);
spec_q_norm = row_normalize(spec_q);

[~, slice_idx_small] = min(abs(delta_theta_vec - slice_delta_small));
[~, slice_idx_mid] = min(abs(delta_theta_vec - slice_delta_mid));
slice_delta_small_actual = delta_theta_vec(slice_idx_small);
slice_delta_mid_actual = delta_theta_vec(slice_idx_mid);

fprintf('=== B3 selected spectrum slices ===\n');
fprintf('Small-separation slice: Delta theta = %.1f deg\n', slice_delta_small_actual);
fprintf('Mid-separation slice:   Delta theta = %.1f deg\n', slice_delta_mid_actual);

fig = figure('Position', [80 80 1060 720], 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot_contour_panel(angle_scan, delta_theta_vec, spec_pn_norm, theta_center, ...
    '(a) EVD-$P_n$');

nexttile;
plot_contour_panel(angle_scan, delta_theta_vec, spec_q_norm, theta_center, ...
    '(b) Ours-$Q$');

nexttile;
plot_slice_panel(angle_scan, spec_pn_norm(slice_idx_small, :), spec_q_norm(slice_idx_small, :), ...
    theta_center, slice_delta_small_actual, ...
    sprintf('(c) $\\Delta\\theta=%.1f^\\circ$', slice_delta_small_actual));

nexttile;
plot_slice_panel(angle_scan, spec_pn_norm(slice_idx_mid, :), spec_q_norm(slice_idx_mid, :), ...
    theta_center, slice_delta_mid_actual, ...
    sprintf('(d) $\\Delta\\theta=%.1f^\\circ$', slice_delta_mid_actual));

export_ieee_figure(fig, 'plotB3_operator_response');

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

function grp = make_delta_group(delta)
    tag = strrep(sprintf('%.1f', delta), '.', 'p');
    grp = ['/Delta_' tag 'deg'];
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

function theta_pair = true_angles(theta_center, delta)
    theta_pair = [theta_center - delta / 2, theta_center + delta / 2];
end

function plot_contour_panel(angle_scan, delta_vec, S, theta_center, panel_title)
    imagesc(angle_scan, delta_vec, S);
    set(gca, 'YDir', 'normal');
    colormap(gca, parula);
    cb = colorbar;
    cb.FontSize = 8;
    hold on;
    plot(theta_center - delta_vec / 2, delta_vec, 'k--', 'LineWidth', 1.2);
    plot(theta_center + delta_vec / 2, delta_vec, 'k--', 'LineWidth', 1.2);
    hold off;
    xlim([min(angle_scan), max(angle_scan)]);
    xlabel('Angle $\theta$ (deg)', 'Interpreter', 'latex');
    ylabel('Angular separation $\Delta\theta$ (deg)', 'Interpreter', 'latex');
    title(panel_title, 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
    format_ieee_axes(gca);
end

function plot_slice_panel(angle_scan, spec_pn_slice, spec_q_slice, theta_center, delta, panel_title)
    theta_pair = true_angles(theta_center, delta);
    hold on; grid on;
    plot(angle_scan, spec_pn_slice, '-', 'LineWidth', 1.2, ...
        'Color', [0.00 0.45 0.74], 'DisplayName', 'EVD-$P_n$');
    plot(angle_scan, spec_q_slice, '--', 'LineWidth', 1.2, ...
        'Color', [0.90 0.10 0.10], 'DisplayName', 'Ours-$Q$');
    xline(theta_pair(1), ':k', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    xline(theta_pair(2), ':k', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    hold off;
    xlim([min(angle_scan), max(angle_scan)]);
    ylim([0, 1.05]);
    xlabel('Angle $\theta$ (deg)', 'Interpreter', 'latex');
    ylabel('Normalized spectrum', 'Interpreter', 'latex');
    title(panel_title, 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', 10);
    legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 8);
    format_ieee_axes(gca);
end
