
function wts = getOptWt_Quadprog(mu, V, lambda, lb, ub)

% optimize the following form
%       -w' * m + lambda * w' * v * w
% with the lower bound 'lb' and upper bound 'ub'
% m must be p x 1 vector and v must be p x p covariance

m = length(mu);
f = -0.5 * mu;
H = lambda * V;
A = zeros(1, m);    b = [0];
Aeq = ones(1, m);   beq = [1];
wts0 = zeros(m, 1) / m;
%wts0 = ones(m, 1) / m;
wts = quadprog(H, f, A, b, Aeq, beq, lb, ub, wts0, ...
    optimset('LargeScale', 'off', 'Display', 'off'));

