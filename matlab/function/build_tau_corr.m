function tau_corr = build_tau_corr(Y, tau_max)
%BUILD_TAU_CORR Build multi-lag autocorrelation features for SubspaceNet.
%   tau_corr = BUILD_TAU_CORR(Y, tau_max) returns a tensor of size
%   (tau_max+1, 2*M, M), where Y is an M-by-T complex snapshot matrix.

if nargin < 2
    tau_max = 4;
end

[m, t] = size(Y);
tau_corr = zeros(tau_max + 1, 2 * m, m, 'single');

for lag = 0:tau_max
    if lag >= t
        error('tau_max must be smaller than the number of snapshots.');
    end
    y1 = Y(:, 1:t-lag);
    y2 = Y(:, 1+lag:t);
    corr_mat = (y1 * y2') / (t - lag);
    tau_corr(lag + 1, 1:m, :) = single(real(corr_mat));
    tau_corr(lag + 1, m + 1:2*m, :) = single(imag(corr_mat));
end
end
