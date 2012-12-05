
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
SPDates = flipud(SandP(:,1));
SPPrices = flipud(SandP(:,5));

% Converts S&P prices to returns
SPRets = PricesToReturns(SPPrices);

% Note: the S&P dates are on average 28 days lagged rather than a month
% lagged: look-ahead bias?
% To completely remove look-ahead bias, use 1:(end-1) instead of 2:end and
% regress on what is essentially 
SP = [SPDates(2:end) SPRets];

regress_mat = [ones(m,1) SP(:,2)]; % regression matrix: S&P returns and ones

% creates empty matrices to save regression coefficients, fitted returns, and
% errors
LS_coeffs = zeros(2,n);
single_factor_fit = zeros(m,n);
regress_err = zeros(m,n);

% Performs regression and saves coefficients, fitted returns, and errors
% for each FF portfolio
for i = 1:n
    LS_coeffs(:,i) = regress(X_rets(:,i), regress_mat);
    single_factor_fit(:,i) = regress_mat*LS_coeffs(:,i);
    regress_err(:,i) = X_rets(:,i) - single_factor_pred(:,i);
end


% Parts (b) and (c)
[PredictedMean, PredictedCov, Error] = Fit_AR_GARCH(X_rets, m);


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
returns = [ FF(1:window,2) FF(1:window,5) FF(1:window,9) FF(1:window,12) FF(1:window,15) FF(1:window,18)];

% covariance matrix for returns (taken as known by BL)
sigma = cov(returns);

% equilibrium risk prima = prior mean
pi = gamma*sigma*weights;

% historical risk prima
hist_mean = mean(returns, 1);




% Step (2): Specify views based a "momentum-based factor model" 

P = eye(n); % projection matrix

q = zeros(n,1); % REPLACE WITH OUTPUT (mean) FROM FORECAST MODEL
omega = eye(n); % REPLACE WITH OUTPUT (covariance of errors from model)

tau = 1/window; % scalar indicating uncertainty of prior

sigma_pi = tau*sigma; % prior covariance matrix

S = ( sigma_pi\eye(n) + P'*(omega\eye(n))*P )\eye(n); % defines posterior covariance matrix
m = S*( (sigma_pi\eye(n))*pi + P'*(omega\eye(n))*q ); % defines posterior mean

% Values of mu and sigma used in Black-Litterman portfolio optimization
mu_BL = m;
sigma_BL = S + sigma;

% now, perform M-V portfolio optimization






% Step (3): Black-Litterman Master Formula: plot cumulative returns over
% time



% Step (4): Plot and compare turnover rates



