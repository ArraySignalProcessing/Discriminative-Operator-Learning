function R_shrink = shrinkage_covariance(R, alpha, beta)
%SHRINKAGE_COVARIANCE Shrink sample covariance toward Toeplitz/white targets.
%   alpha controls Toeplitz-structured shrinkage and beta controls diagonal
%   loading toward trace(R)/M * I.

if nargin < 3
    beta = 0;
end

M = size(R, 1);
alpha = min(max(real(alpha), 0), 1);
beta = min(max(real(beta), 0), 1 - alpha);

T = toeplitz_covariance_reconstruct(R);
mu = real(trace(R)) / M;
R_shrink = (1 - alpha - beta) * R + alpha * T + beta * mu * eye(M);
R_shrink = (R_shrink + R_shrink') / 2;

% Toeplitz projection is structural, not a PSD projection. MATLAB musicdoa
% rejects covariance matrices with small negative eigenvalues, so project the
% shrinkage result back to the Hermitian PSD cone before use.
[V, D] = eig(R_shrink);
eig_vals = real(diag(D));
floor_val = max(eps(max(abs(eig_vals))), eps);
eig_vals = max(eig_vals, floor_val);
R_shrink = V * diag(eig_vals) * V';
R_shrink = (R_shrink + R_shrink') / 2;
end
