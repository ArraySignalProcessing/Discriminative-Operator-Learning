function [sq_err, max_abs_err] = match_doa_error(est_angles, true_angles)
%MATCH_DOA_ERROR Minimum-permutation angular error for one Monte Carlo trial.

est = est_angles(:);
gt = true_angles(:);
K = numel(gt);

if numel(est) ~= K || any(~isfinite(est)) || any(~isfinite(gt))
    sq_err = inf;
    max_abs_err = inf;
    return;
end

perm_list = perms(1:K);
best_sq = inf;
best_abs = inf;

for ii = 1:size(perm_list, 1)
    diff = est(perm_list(ii, :)) - gt;
    sq = sum(diff.^2);
    if sq < best_sq
        best_sq = sq;
        best_abs = max(abs(diff));
    end
end

sq_err = best_sq;
max_abs_err = best_abs;
end
