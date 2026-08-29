% Unitary ESPRIT for DoA estimation (TLS implementation)
function ang = unit_ESPRIT_fast(Rx_est, ds, K, w)
% ds: missing elements of each sub-array for overlapping arrays
% K: number of sources
N = size(Rx_est,1); % number of sensors
Ns = N-ds;

weights = diag(sqrt([1:w-1 w*ones(1,Ns-2*(w-1)) w-1:-1:1])); % Eq 9.132 in [1]
O = zeros(Ns,ds);

% Js1 = [weights O]; % don't really need that
J2 = [O weights];

% Calculate the Q matrices
QNs = Q_mat(Ns);
QN = Q_mat(N);

% Perform the eigendecomposition of Rx_est
% Check for positive semi definite
[eigenvects,sED] = eig((Rx_est+Rx_est')/2);  % ensure Hermitian
sED = diag(sED);
[~, ind_max_eig] = maxk(sED,K);
ES = eigenvects(:,ind_max_eig);

% Calculate the K1, K2 matrices
K_mat = QNs'*J2*QN;
K1 = 2*real(K_mat);
K2 = 2*imag(K_mat);

TLS1 = K1*ES; % size Ns*K
TLS2 = K2*ES; % size Ns*K

C = [TLS1';TLS2']*[TLS1 TLS2]; 
[U,~,~] = svd(C);             % C is 2*D x 2*D
V12 = U(1:K,K+1:2*K);         % D x D
V22 = U(K+1:2*K,K+1:2*K);     % D x D
psi = -V12/V22;               % Eq. (9.122) in [1]
psieig = real(eig(psi));
%   Extract angle information estimated from two subarrays based on the
%   distance of the phase center between subarrays.
psi = 2*atan(psieig)/pi/ds;
ang = sort(asind(psi));

end
