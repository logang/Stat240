
% FUNCTION: ForecastReturns
% ----------------------
% This function returns a one-step-ahead forecast for the given time series
% using 2 factors and an intercept coefficient. The first factor is assumed
% to be an order-1 autoregression; the latter is a lagged time series
% presumed to be a vector (e.g., S&P returns or LIBOR rate)

function PredMean = ForecastReturns(r, fact)

    m = size(r, 1); % number of time points
    n = size(r, 2); % number of assets
    
    % lagged returns
    X_lag = r(1:m-1, :);
    fact_lag = fact(1:m-1); % lagged factor

    % returns
    X = r(2:m, :);

    % initializes matrix of regression parameters
    Coeff = zeros(n, 3);
    
    X_fit = zeros(m-1,n); % initializes fit
    X_resid = zeros(m-1,n); % initializes residuals
    
    for i = 1:n
        reg_mat = [ones(m-1,1) fact_lag X_lag(:,i)]; % regression matrix for series i
        Coeff(i,:) = reg_mat\X(:,i); % saves least-squares coefficient
        X_fit(:,i) = reg_mat*Coeff(i,:)'; % fitted returns
        X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
    end
 
    PredMean = Coeff(:,1) + Coeff(:,2)*fact_lag(m-1) +  diag(Coeff(:,3))*X_lag(m-1,:)'; % one-step-ahead forecast based on fit
    
end