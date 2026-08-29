function R_toeplitz = toeplitz_covariance_reconstruct(R)
%TOEPLITZ_COVARIANCE_RECONSTRUCT Hermitian Toeplitz projection of covariance.
%   The lag values are estimated by averaging each sample-covariance diagonal.

M = size(R, 1);
lag_vals = zeros(M, 1);
for lag = 0:M-1
    lag_vals(lag + 1) = mean(diag(R, lag));
end

R_toeplitz = toeplitz(conj(lag_vals), lag_vals);
R_toeplitz = (R_toeplitz + R_toeplitz') / 2;
end
