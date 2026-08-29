function plotB6_k123_interactive_slices()
% plotB6_k123_interactive_slices.m
% Interactive B6 slice viewer. It overlays EVD-Pn and learned-Q spectra in
% one panel and uses a slider to select the K=1 true angle or K=2/3 Delta.

clearvars -except ans; close all;

repo_dir = fileparts(fileparts(mfilename('fullpath')));
data_dir = fullfile(repo_dir, 'data', 'B6');

% Keep the same learned-Q sign convention as plotB6_k123_composite_spectrum.
flip_Q_angle_axis = true;

cache = cell(1, 3);
current_K = 3;

fig = figure('Name', 'B6 interactive spectrum slices', ...
    'Position', [120 120 980 620], 'Color', 'w', 'NumberTitle', 'off');
ax = axes('Parent', fig, 'Position', [0.08 0.20 0.88 0.70]);

uicontrol(fig, 'Style', 'text', 'String', 'Source count', ...
    'Units', 'normalized', 'Position', [0.08 0.08 0.12 0.04], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left');
popup = uicontrol(fig, 'Style', 'popupmenu', ...
    'String', {'K=1: true angle', 'K=2: separation', 'K=3: adjacent spacing'}, ...
    'Value', current_K, 'Units', 'normalized', ...
    'Position', [0.19 0.08 0.22 0.05], 'Callback', @on_k_changed);

slider = uicontrol(fig, 'Style', 'slider', ...
    'Units', 'normalized', 'Position', [0.47 0.085 0.36 0.035], ...
    'Callback', @on_slider_changed);
value_label = uicontrol(fig, 'Style', 'text', 'String', '', ...
    'Units', 'normalized', 'Position', [0.84 0.075 0.12 0.055], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left');
status_label = uicontrol(fig, 'Style', 'text', ...
    'String', 'Joint normalization: Pn and Q share the same min/max per slice.', ...
    'Units', 'normalized', 'Position', [0.08 0.02 0.88 0.04], ...
    'BackgroundColor', 'w', 'HorizontalAlignment', 'left');

configure_slider();
update_plot();

    function on_k_changed(~, ~)
        current_K = get(popup, 'Value');
        configure_slider();
        update_plot();
    end

    function on_slider_changed(~, ~)
        idx = round(get(slider, 'Value'));
        set(slider, 'Value', idx);
        update_plot();
    end

    function configure_slider()
        data = get_data(current_K);
        n_scan = numel(data.scan_vec);
        if n_scan <= 1
            slider_step = [1 1];
        else
            slider_step = [1 / (n_scan - 1), min(10 / (n_scan - 1), 1)];
        end
        set(slider, 'Min', 1, 'Max', n_scan, ...
            'Value', max(1, round(n_scan / 2)), ...
            'SliderStep', slider_step);
    end

    function update_plot()
        data = get_data(current_K);
        idx = round(get(slider, 'Value'));
        idx = min(max(idx, 1), numel(data.scan_vec));
        set(slider, 'Value', idx);

        if ~data.computed(idx)
            set(status_label, 'String', sprintf('Computing K=%d slice %d/%d ...', ...
                current_K, idx, numel(data.scan_vec)));
            drawnow;
            data = compute_slice(data, idx);
            cache{current_K} = data;
        end

        scan_value = data.scan_vec(idx);
        [pn_plot, q_plot] = normalize_pair(data.spec_pn(idx, :), data.spec_q(idx, :));
        theta_vec = true_angles(data.theta_center, current_K, scan_value);

        delete(allchild(ax));
        legend(ax, 'off');
        hold(ax, 'on');
        plot(ax, data.angle_scan, pn_plot, '-', 'LineWidth', 1.4, ...
            'Color', [0.00 0.45 0.74], 'DisplayName', 'EVD-$P_n$');
        plot(ax, data.angle_scan, q_plot, '--', 'LineWidth', 1.4, ...
            'Color', [0.85 0.10 0.10], 'DisplayName', 'K=3 model $Q$');
        for ii = 1:numel(theta_vec)
            xline(ax, theta_vec(ii), ':k', 'LineWidth', 1.0, ...
                'HandleVisibility', 'off');
        end
        hold(ax, 'off');
        grid(ax, 'on');
        xlim(ax, [min(data.angle_scan), max(data.angle_scan)]);
        ylim(ax, [0, 1.05]);
        xlabel(ax, 'Angle $\theta$ (deg)', 'Interpreter', 'latex');
        ylabel(ax, 'Joint-normalized spectrum', 'Interpreter', 'latex');
        title(ax, make_title(current_K, scan_value, theta_vec), ...
            'Interpreter', 'latex', 'FontWeight', 'normal');
        legend(ax, 'Interpreter', 'latex', 'Location', 'best');
        format_ieee_axes(ax);

        if current_K == 1
            set(value_label, 'String', sprintf('\\theta_0 = %.1f deg', scan_value));
        else
            set(value_label, 'String', sprintf('\\Delta = %.1f deg', scan_value));
        end
        set(status_label, 'String', ...
            'Joint normalization: Pn and Q share the same min/max per slice.');
    end

    function data = get_data(K)
        if ~isempty(cache{K})
            data = cache{K};
            return;
        end

        data_file = fullfile(data_dir, sprintf('B6_K123_CompositeSpectrum_QPn_K%d.h5', K));
        q_file = fullfile(data_dir, sprintf('B6_K123_CompositeSpectrum_Q_K%d.h5', K));
        if ~exist(data_file, 'file')
            error('Missing B6 data file: %s', data_file);
        end
        if ~exist(q_file, 'file')
            error('Missing B6 Q file: %s', q_file);
        end

        data.K = K;
        data.data_file = data_file;
        data.q_file = q_file;
        data.angle_scan = h5read(data_file, '/angle_scan');
        data.angle_scan = data.angle_scan(:).';
        data.M = h5readatt(data_file, '/', 'M');
        data.theta_center = h5readatt(data_file, '/', 'theta_center');
        if K == 1
            data.scan_vec = h5read(data_file, '/theta_vec');
            data.group_fun = @make_theta_group;
        else
            data.scan_vec = h5read(data_file, '/delta_theta_vec');
            data.group_fun = @make_delta_group;
        end
        data.scan_vec = data.scan_vec(:).';
        data.A_scan_nominal = steering_matrix(data.angle_scan, data.M);
        q_angle_scan = data.angle_scan;
        if flip_Q_angle_axis
            q_angle_scan = -q_angle_scan;
        end
        data.A_scan_Q = steering_matrix(q_angle_scan, data.M);
        data.spec_pn = zeros(numel(data.scan_vec), numel(data.angle_scan));
        data.spec_q = zeros(numel(data.scan_vec), numel(data.angle_scan));
        data.computed = false(1, numel(data.scan_vec));
        cache{K} = data;
    end

    function data = compute_slice(data, idx)
        grp = data.group_fun(data.scan_vec(idx));
        Pn = read_complex_stack(data.data_file, [grp '/Pn_real'], [grp '/Pn_imag'], false);
        Q = read_complex_stack(data.q_file, [grp '/Q_real'], [grp '/Q_imag'], true);
        data.spec_pn(idx, :) = mean_spectrum(Pn, data.A_scan_nominal);
        data.spec_q(idx, :) = mean_spectrum(Q, data.A_scan_Q);
        data.computed(idx) = true;
    end
end

function A = steering_matrix(angle_grid, M)
    theta = deg2rad(angle_grid(:));
    pos = (0:M-1) * 0.5;
    A = exp(1j * 2 * pi * sin(theta) * pos);
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

function [a_norm, b_norm] = normalize_pair(a, b)
    lo = min([a(:); b(:)]);
    hi = max([a(:); b(:)]);
    denom = hi - lo;
    if denom > 0
        a_norm = (a - lo) / denom;
        b_norm = (b - lo) / denom;
    else
        a_norm = zeros(size(a));
        b_norm = zeros(size(b));
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

function title_text = make_title(K, scan_value, theta_vec)
    if K == 1
        title_text = sprintf('B6 slice, $K=1$, $\\theta_0=%.1f^\\circ$', scan_value);
    else
        title_text = sprintf('B6 slice, $K=%d$, $\\Delta\\theta=%.1f^\\circ$, true DOAs = %s', ...
            K, scan_value, mat2str(theta_vec, 3));
    end
end
