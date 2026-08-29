function ang = ESPRIT_doa(R, ds, D)
% Standard TLS-ESPRIT for DoA estimation with a ULA.

N = size(R,1);
Ns = N - ds;

% ---- Standard shift-selection matrices ----
Js1 = [eye(Ns) zeros(Ns, ds)];
Js2 = [zeros(Ns, ds) eye(Ns)];

% ---- EVD ----
[eigenvects, sED] = eig((R+R')/2);
sED = diag(sED);
[~, indx] = sort(sED, 'descend');
eigenvects = eigenvects(:, indx);

% ---- Subarray signal subspaces ----
Us1 = Js1 * eigenvects(:, 1:D);
Us2 = Js2 * eigenvects(:, 1:D);

% ---- TLS-ESPRIT ----
C = [Us1'; Us2'] * [Us1 Us2];
[U, ~, ~] = svd(C);
V12 = U(1:D, D+1:2*D);
V22 = U(D+1:2*D, D+1:2*D);
psi = -V12 * pinv(V22);

% ---- DOA mapping ----
phases = angle(eig(psi));
u = phases / (pi * ds);          % half-wavelength spacing, ds subarray shift
u = min(max(real(u), -1), 1);     % numerical guard for asin
ang = sort(asind(u));

% Force row vector
ang = ang(:).';
end
