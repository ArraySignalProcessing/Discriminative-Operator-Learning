function export_ieee_figure(fig, base_name)
% export_ieee_figure Export a MATLAB figure for IEEE-style paper review.
%
% Outputs are written to <repo>/figures/<base_name>.pdf and .png.

repo_dir = fileparts(fileparts(mfilename('fullpath')));
out_dir = fullfile(repo_dir, 'figures');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

set(fig, 'Color', 'w');
pdf_path = fullfile(out_dir, [base_name '.pdf']);
png_path = fullfile(out_dir, [base_name '.png']);

try
    exportgraphics(fig, pdf_path, 'ContentType', 'vector');
    exportgraphics(fig, png_path, 'Resolution', 600);
catch
    print(fig, pdf_path, '-dpdf', '-painters');
    print(fig, png_path, '-dpng', '-r600');
end

fprintf('Saved figure: %s\n', pdf_path);
fprintf('Saved figure: %s\n', png_path);
end
