
% PART I: single-factor time-series forecasting
% ------


% Retrieves monthly returns from data
smlo_ret = FF(:,2);
smme_ret = FF(:,5);
smhi_ret = FF(:,8);
bilo_ret = FF(:,11);
bime_ret = FF(:,14);
bihi_ret = FF(:,17);
X_rets = [smlo_ret smme_ret smhi_ret bilo_ret bime_ret bihi_ret]; % matrix of returns
FFDates = FF(:,1); % saves Fama-French dates for comparison with S&P


m = length(smlo_ret); % number of observations
n = 6; % number of stocks

% Part (a): single-factor model (S&P)
SPDates = SandPDates;
SPRets = SandPRets;
SP = [SPDates(1:end-1) SPRets];

% regression matrix: S&P returns and ones
regress_mat = [ones(1:end-1,1) SP(:,2)];

% creates empty matrices to save regression coefficients, fitted returns, and
% errors
LS_coeffs = zeros(2,n);
single_factor_fit = zeros(m-1,n);
regress_err = zeros(m-1,n);

% Performs regression and saves coefficients, fitted returns, and errors
% for each FF portfolio.
% Note that the returns are regressed on the previous month's S&P returns
% rather than the current month's S&P returns.
for i = 1:n
    LS_coeffs(:,i) = regress(X_rets(2:end,i), regress_mat);
    single_factor_fit(:,i) = regress_mat*LS_coeffs(:,i);
    regress_err(:,i) = X_rets(2:end,i) - single_factor_fit(:,i);
end


% Parts (b) and (c)
[PredictedMean, PredictedCov, Error, GARCHCoeffs, ARCoeffs, Innos] = Fit_AR_GARCH(X_rets, m);


% Part (d): NPEB approach






% PART II
% -------

% Step (1): Derive equilibrium weights and risk prima from historical data


% Sets gamma = market's risk parameter
gamma = 2.5; 

% window = 5 years of monthly returns
window = 12*5;

% calculates market capitalization of each portfolio
smlo_w = diag(FF(1:window,3))*FF(1:window,4);
smme_w = diag(FF(1:window,6))*FF(1:window,7);
smhi_w = diag(FF(1:window,9))*FF(1:window,10);
bilo_w = diag(FF(1:window,12))*FF(1:window,13);
bime_w = diag(FF(1:window,15))*FF(1:window,16);
bihi_w = diag(FF(1:window,18))*FF(1:window,19);

% calculates market-capitalization weights by calculating the mean size of
% each portfolio, then normalizing
weights = [mean(smlo_w); mean(smme_w); mean(smhi_w); mean(bilo_w); mean(bime_w); mean(bihi_w)];
weights = weights/sum(weights);

% asset return for each portfolio
%returns = [ FF(1:window,2) FF(1:window,5) FF(1:window,8) FF(1:window,11) FF(1:window,14) FF(1:window,17)];
returns = X_rets(1:window,:);

% covariance matrix for returns (taken as known by BL)
sigma = cov(returns);

% equilibrium risk prima = prior mean
pi = gamma*sigma*weights;

% historical risk prima
hist_mean = mean(returns, 1);




% Step (2): Specify views based on a "momentum-based factor model" 

P = eye(n); % projection matrix

tau = 1/window; % scalar indicating uncertainty of prior

t = window; % current end index

sigma_pi = tau*sigma; % prior covariance matrix


q = zeros(n,1); % REPLACE WITH OUTPUT (mean) FROM FORECAST MODEL
omega = eye(n); % REPLACE WITH OUTPUT (covariance of errors from model)


S = ( sigma_pi\eye(n) + P'*(omega\eye(n))*P )\eye(n); % defines posterior covariance matrix
M = S*( (sigma_pi\eye(n))*pi + P'*(omega\eye(n))*q ); % defines posterior mean

% Values of mu and sigma used in Black-Litterman portfolio optimization
mu_BL = M;
sigma_BL = S + sigma;

% Replace this with the actual benchmarks from S&P data.
bench_mu = 0.006;
bench_t = 0.01;

% NOTE: REPLACE RETURNS ABOVE WITH A MORE GENERALIZED MATRIX AND DEFINE
% THIS AS THE WINDOW+1ST ROW OF THAT MATRIX.
%realized_returns = [FF(window+1,2) FF(window+1,5) FF(window+1,8) FF(window+1,11) FF(window+1,14) FF(window+1,17)];
realized_returns = X_rets(t+1, :);

% now, perform M-V portfolio optimization
[w_p, return_p, excess_p, stdev_p] = OptimizePortfolio(mu_BL, sigma_BL, 0, bench_mu, bench_t, realized_returns);




% Step (3): Black-Litterman Master Formula: plot cumulative returns over
% time


% NOTE: incorporate this into step (2) so as to not duplicate code.

excess_returns = zeros(m - t, 1); % keeps track of CUMULATIVE excess returns over time
total = 0; % keeps track of total excess returns
iteration = 1; % keeps track of iterations

% while the end of the FF data has not yet been reached, re-applies
% Black-Litterman and keeps track of cumulative returns.
while ( t < m )
     
    q = zeros(n,1); % REPLACE WITH ACTUAL OUTPUT FROM FORECAST MODEL
    omega = eye(n); % REPLACE WITH ACTUAL OUTPUT FROM FORECAST MODEL
    
    % NOTE: I THINK WE WILL WANT TO RE-COMPUTE PRIOR EQUILIBRIUM WEIGHTS ON
    % EVERY ITERATION, WITH AN EXPANDING WINDOW.
    % pi = ...

    S = ( sigma_pi\eye(n) + P'*(omega\eye(n))*P )\eye(n); % defines posterior covariance matrix
    M = S*( (sigma_pi\eye(n))*pi + P'*(omega\eye(n))*q ); % defines posterior mean

    % Values of mu and sigma used in Black-Litterman portfolio optimization
    mu_BL = M;
    sigma_BL = S + sigma;

    
    % CHECK S&P DATA: maybe use t-1, t instead of t, t+1.
    bench_mu = mean(SPRets(1:t));
    bench_t = SPRets(t+1);

    % Realized return
    realized_returns = X_rets(t+1, :);

    % now, perform M-V portfolio optimization
    [w_p, return_p, excess_p, stdev_p] = OptimizePortfolio(mu_BL, sigma_BL, 0, bench_mu, bench_t, realized_returns);
    
    total = total + excess_p; % updates total
    excess_returns(iteration) = total; % saves cumulative excess return at time t
    
    % updates time and iteration #
    t = t + 1;
    iteration = iteration + 1;
end

% Plots cumulative returns
figure
plot(excess_returns);



% Step (4): Plot and compare turnover rates



