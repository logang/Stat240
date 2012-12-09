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
regress_mat = [ones(m-1,1) SP(:,2)];

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
    LS_coeffs(:,i) = regress(X_rets(1:end-1,i), regress_mat);
    single_factor_fit(:,i) = regress_mat*LS_coeffs(:,i);
    regress_err(:,i) = X_rets(1:end-1,i) - single_factor_fit(:,i);
end


% Parts (b) and (c)
[PredictedMean, PredictedCov, Error, GARCHCoeffs, ARCoeffs, Innos] = Fit_AR_GARCH(X_rets, m);


% Part (d): NPEB approach: run Lai/Xing/Chen code on Fama-French portfolios

% calculates market capitalization of each portfolio and stores in a matrix
% (used by both (d) and B-L optimization)
smlo_w = diag(FF(:,3))*FF(:,4);
smme_w = diag(FF(:,6))*FF(:,7);
smhi_w = diag(FF(:,9))*FF(:,10);
bilo_w = diag(FF(:,12))*FF(:,13);
bime_w = diag(FF(:,15))*FF(:,16);
bihi_w = diag(FF(:,18))*FF(:,19);
Mkt_cap = [smlo_w smme_w smhi_w bilo_w bime_w bihi_w];

% window size and weights for NPEB code
NPEB_window = 120;
NPEB_wts = Mkt_cap(NPEB_window,:)./sum(Mkt_cap(NPEB_window,:));



% PART II
% -------

% Step (1): Derive equilibrium weights and risk prima from historical data


% Sets gamma = market's risk parameter
gamma = 2.5; 

% window = 5 years of monthly returns
window = 12*5;

% calculates market-capitalization weights by normalizing mkt cap at
% current month (end of window)
weights = Mkt_cap(window,:)./sum(Mkt_cap(window,:));


excess_rets = X_rets - diag(Libor)*ones(size(X_rets));


% covariance matrix for excess returns (taken as known by BL)
sigma = cov(excess_rets);

% mean returns implied by market equilibrium
pi = gamma*sigma*weights';

% historical mean returns
%hist_mean = mean(returns, 1);
hist_mean = mean(excess_rets, 1);



% Step (2): Specify views based on a "momentum-based factor model" 

P = eye(n); % projection matrix

t = window; % current end index

returns = excess_rets(1:t,:);

tau = 1/t; % scalar indicating uncertainty of prior

sigma_pi = tau*sigma; % prior covariance matrix

% forecasted returns for next month
q = ForecastReturns(returns, SPRets - Libor(1:end-1));

delta = 10;

% initializes omega. Since it is defined as a function of the forecast
% errors from the previous forecast, it is initialized
omega = delta*eye(n);


S = ( sigma_pi\eye(n) + P'*(omega\eye(n))*P )\eye(n); % defines posterior covariance matrix
M = S*( (sigma_pi\eye(n))*pi + P'*(omega\eye(n))*q ); % defines posterior mean

% Values of mu and sigma used in Black-Litterman portfolio optimization
mu_BL = M;
sigma_BL = S + sigma;

% Replace this with the actual benchmarks from S&P data.
bench_t = SPRets(t+1);

% Realized returns
%realized_returns = X_rets(t+1, :);
realized_returns = excess_rets(t+1, :);

% now, perform M-V portfolio optimization (B-L)
[w_p, return_p, excess_p, stdev_p, sharpe_p] = OptimizePortfolio(mu_BL, sigma_BL, -0.3, bench_t - Libor(t), realized_returns);

% Markowitz for comparison
[w_m, return_m, excess_m, stdev_m, sharpe_m] = OptimizePortfolio(hist_mean', sigma, -0.3, bench_t - Libor(t), realized_returns);

% Saves old weights for tracking turnover rate in the next step
w_p_old = w_p;
w_m_old = w_m;



% Step (3): Black-Litterman Master Formula: plot cumulative returns over
% time
% Step (4): Track turnover rate


excess_returns_BL = zeros(m - t - 2, 1);  % keeps track of CUMULATIVE excess returns over time
excess_returns_M = zeros(m - t - 2, 1); 
turnover_BL = zeros(m - t - 2, 1);  % tracks turnover rate
turnover_M = zeros(m - t - 2, 1); 
sharpe_BL = zeros(m - t - 2, 1);  % tracks Sharpe ratios
sharpe_M = zeros(m - t - 2, 1); 


total_BL = excess_p; % keeps track of total excess returns for B-L
total_M = excess_m; % keeps track of total excess returns for Markowitz

Pred_error = zeros(m-t-2, n);


iteration = 1; % keeps track of iterations
t = t + 1; % the following loop tracks optimization results for t > 1.

% while the end of the FF data has not yet been reached, re-applies
% Black-Litterman and keeps track of cumulative returns and turnover rate.
% Also tracks Markowitz for comparison.
while ( t < m-1 )
     
    
    % Redefine prior based on expanding window and current
    % market-capitalization weights
    weights = Mkt_cap(t,:)./sum(Mkt_cap(t,:));
    %returns = X_rets(1:t,:);
    returns = excess_rets(iteration:t,:);
    
    sigma = cov(returns); % historical covariance matrix (plug-in estimate)
    
    hist_mean = mean(returns, 1); % historical mean returns (plug-in estimate)
    pi = gamma*sigma*weights';
    tau = 1/t;
    sigma_pi = tau*sigma;
    
    
    % Redefine posterior.
    
    % Note that Omega is a function of the errors from the previous
    % prediction, and is therefore defined at the end of this loop to form
    % Omega(t+1).
    q = ForecastReturns(returns, SPRets(iteration:end) - Libor(iteration:end-1));

    % Apply Black-Litterman master formula
    S = ( sigma_pi\eye(n) + P'*(omega\eye(n))*P )\eye(n); % defines posterior covariance matrix
    M = S*( (sigma_pi\eye(n))*pi + P'*(omega\eye(n))*q ); % defines posterior mean

    % Values of mu and sigma used in Black-Litterman portfolio optimization
    mu_BL = M;
    sigma_BL = S + sigma; 

    
    bench_t = SPRets(t+1); % used as benchmark in computing excess returns


    %realized_returns = X_rets(t+1,:); % actual returns observed in subsequent period
    realized_returns = excess_rets(t+1,:);
    
    Pred_error(iteration,:) = abs((realized_returns' - q)');
    
    omega = delta*diag(Pred_error(iteration,:)); % defines uncertainty of (next) mean based on performance
    
    
    % BLACK-LITTERMAN PORTFOLIO OPTIMIZATION
    [w_p, return_p, excess_p, stdev_p, sharpe_p] = OptimizePortfolio(mu_BL, sigma_BL, -0.3, bench_t - Libor(t), realized_returns);
    
    total_BL = total_BL + excess_p; % updates total for B-L
    excess_returns_BL(iteration) = total_BL; % saves cumulative excess return at time t for B-L
    sharpe_BL(iteration) = sharpe_p;
    turnover_BL(iteration) = GetTurnoverRate(w_p, w_p_old); % tracks B-L turnover rate
    w_p_old = w_p; % saves previous result for next iteration
    
   
    
    % MARKOWITZ PLUG-IN PORTFOLIO OPTIMIZATION
    [w_m, return_m, excess_m, stdev_m, sharpe_m] = OptimizePortfolio(hist_mean', sigma, -0.3, bench_t - Libor(t), realized_returns);

    
    total_M = total_M + excess_m; % updates total for Markowitz
    excess_returns_M(iteration) = total_M; % saves cumulative excess return at time t for B-L
    sharpe_M(iteration) = sharpe_m;
    turnover_M(iteration) = GetTurnoverRate(w_m, w_m_old); % tracks Markowitz turnover rate
    w_m_old = w_m; % saves previous result for next iteration
     
    
    
    % updates time and iteration #
    t = t + 1;
    iteration = iteration + 1;
    
end


% Plots cumulative returns
figure
plot(excess_returns_M, ':');
hold on
plot(excess_returns_BL);
legend('Markowitz', 'Black-Litterman');
title ('Cumulative Excess Returns, Two-Factor w/ 5-year Window', 'FontSize', 14);
xlabel('Time in months', 'FontSize', 12);
ylabel('Excess returns over S&P 500 index', 'FontSize', 12);


% Plots sharpe ratio
figure
plot(sharpe_M, ':');
hold on
plot(sharpe_BL);
legend('Markowitz', 'Black-Litterman');
title ('Sharpe Ratio Over Time, Two-Factor w/ 5-year Window', 'FontSize', 14);
xlabel('Time in months', 'FontSize', 12);
ylabel('Sharpe ratio', 'FontSize', 12);


% Step (4): Plot and compare turnover rates
figure
plot(turnover_M, ':');
hold on
plot(turnover_BL);
legend('Markowitz', 'Black-Litterman');
title ('Portfolio Turnover Rate, Two-Factor w/ 5-year Window', 'FontSize', 14);
xlabel('Time in months', 'FontSize', 12);
ylabel('Portfolio turnover rate', 'FontSize', 12);