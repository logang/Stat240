


% FUNCTION: Forecast_AR1
% ----------------------
% This function returns a one-step-ahead forecast for the given time series
% using a simple AR(1) regression model with an intercept coefficient.
% Returns a vector of predictions for each column in the matrix r.

function PredMean = Forecast_AR1(r)

    m = size(r, 1); % number of observations
    n = size(r, 2); % number of assets
    
    % lagged returns
    X_lag = r(1:m-1, :);

    % returns
    X = r(2:m, :);

    Coeff = zeros(n, 2); % initializes coefficients
    X_fit = zeros(m-1,n); % initializes fit
    X_resid = zeros(m-1,n); % initializes residuals
    
    for i = 1:n
        reg_mat = [ones(m-1,1) X_lag(:,i)];
        Coeff(i,:) = (reg_mat\X(:,i))'; % saves coefficients of least-squares
        X_fit(:,i) = reg_mat*Coeff(i,:)'; % fitted returns
        X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
    end
 
    PredMean = Coeff(:,1) + diag(Coeff(:,2))*(r(m,:)'); % predicted mean = return value
end