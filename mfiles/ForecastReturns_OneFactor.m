
% FUNCTION: ForecastReturns_OneFactor
% ----------------------
% This function returns a one-step-ahead forecast for the given time series
% using a simple one-factor regression on the (lagged) time series fact.
% It is assumed that fact is a common regressor for each column in r, which
% is a matrix of returns.

function PredMean = ForecastReturns_OneFactor(r, fact)

    m = size(r, 1); % number of observations
    n = size(r, 2); % number of assets
    
    % lagged factor
    fact_lag = fact(1:m-1);

    % returns
    X = r(2:m, :);

    %Coeff = zeros(3,n); % initializes matrix of regression parameters
    Coeff = zeros(n, 2);
    
    X_fit = zeros(m-1,n); % initializes fit
    X_resid = zeros(m-1,n); % initializes residuals
    
    reg_mat = [ones(m-1,1) fact_lag]; % regression matrix is common to all time series in matrix
    
    for i = 1:n
        Coeff(i,:) = reg_mat\X(:,i); % saves least-squares coefficients
        X_fit(:,i) = reg_mat*Coeff(i,:)'; % fitted returns
        X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
    end

    PredMean = Coeff(:,1) + fact(m)*Coeff(:,2); % nx1 vector of one-step-ahead forecasts
end