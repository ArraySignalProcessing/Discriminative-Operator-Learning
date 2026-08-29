function [ang_est, sp_val] = l1_SVD_DoA_est(Y, ULA_N, threshold, SOURCE_K, THETA_angles)
% INPUTS:
%   Y            : 阵列快拍矩阵 (ULA_N × T)
%   ULA_N        : 阵元数 (M)
%   threshold    : 残差约束上限 (标量)
%   SOURCE_K     : 信源数 (K)
%   THETA_angles : 扫描角度网格 (行向量)
%
% OUTPUT:
%   ang_est : 估计的 DOA (升序排列，列向量)
%   sp_val  : 对应角度的功率谱值

    % ---------- 导向矢量与字典 ----------
    ULA_steer_vec = @(x, N) exp(1j * pi * sin(deg2rad(x)) * (0:1:N-1)).';
    NGrids = length(THETA_angles);
    A_dic = zeros(ULA_N, NGrids);
    for n = 1:NGrids
        A_dic(:, n) = ULA_steer_vec(THETA_angles(n), ULA_N);
    end

    % ---------- ℓ₂,₁‑SVD：正确降维到信号子空间 (K 维) ----------
    [~, ~, V] = svd(Y, 'econ');                     % V : T × min(M,T)
    Ydr = Y * V(:, 1:SOURCE_K);                    % M × K  (仅保留信号子空间)
    % Ydr 现在是 M×K 的实值化等效矩阵，保留了 K 个主分量

    % ---------- 求解 SOCP ----------
    cvx_begin quiet
        variable S_est_dr(NGrids, SOURCE_K) complex;   % 稀疏系数矩阵 NGrids × K
        minimize( sum(norms(S_est_dr.')) );             % ℓ₂,₁ 混合范数 (行稀疏)
        subject to
            norm(Ydr - A_dic * S_est_dr, 'fro') <= threshold;
    cvx_end

    % ---------- 重构全快拍空间的表示并计算功率 ----------
    S_est = S_est_dr * V(:, 1:SOURCE_K)';             % NGrids × T
    Ps = sum(abs(S_est).^2, 2);                       % 每个网格点的功率

    % ---------- 提取 K 个最大峰值 ----------
    [sp_val, spa_ind] = maxk(Ps, SOURCE_K);
    ang_est = sort(THETA_angles(spa_ind))';            % 升序，列向量
end