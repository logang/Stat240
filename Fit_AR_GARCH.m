% FUNCTION: Fit_AR_GARCH
% ----------------------
% Inputs:   returns   --  a pxn matrix of asset returns
%           m         --  the index of the one-step-ahead returns we wish to predict
%                         (thus window size is m - 1)
% Outputs:  PredMean  -- a vector of predicted mean asset returns for t = m,
%                        based on an AR(1) model
%           PredCov   -- the predicted covariance matrix for t = m, 
%                        based on a GARCH(1,1) model
%           Error     -- a vector of differences (r_m - PredMean) between
%                        the observed and predicted mean @ t = m

function [PredMean, PredCov, Error] = Fit_AR_GARCH(returns, m)
    
    n = size(returns, 2); % number of assets
    % lagged returns (training)
    X_lag = returns(1:m-2, :);

    % returns (training)
    X = returns(2:m-1, :);

    % returns (testing)
    X_test = returns(m,:);

    % AR(1) model
    % ---------------------
    phi = zeros(2,n); % initializes matrix of regression parameters
    X_fit = zeros(m-2,n); % initializes fit
    X_resid = zeros(m-2,n); % initializes residuals

    % regresses each return on its lags to yield phi (AR(1) model)
    for i = 1:n
        reg_mat = [ones(m-2,1) X_lag(:,i)]; % regression matrix (includes y-intercept term)
        phi(:,i) = reg_mat\X(:,i); % solves for coeffs mu and phi
        X_fit(:,i) = reg_mat*phi(:,i); % fitted returns
        X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
    end

    
    % Fits GARCH(1,1) model to residuals of AR(1) fit
    
    % sets specs for GARCH(1,1) model of residuals from AR(1) fit
    spec = garchset('R', 0, 'M', 0, 'P', 1, 'Q', 1, 'display', 'off');

    Coeff = zeros(n,4); % initializes coefficient matrix (one row per asset)
    Innos = zeros(m-2,n); % initializes matrix of innovations u
    Sigmas = zeros(m-2,n); % initializes matrix of sigmas
    PredCov = zeros(n,n); % initializes one-step-ahead predicted covariance matrix

    % for each asset, fits a GARCH(1,1) model and stores coefficients, sigmas,
    % and innovations in their respective pre-initialized matrices
    for i = 1:n
        [co, err, LLF, Innos(:,i), Sigmas(:,i), summary] = garchfit(spec, X_resid(:,i));
        Coeff(i,:) = [co.C, co.K, co.GARCH, co.ARCH];
        PredCov(i,i) = co.K + (Sigmas(m-2,i)^2)*co.GARCH + (Innos(m-2, i)^2)*co.ARCH;
    end
    
    StdInnos = Innos./Sigmas; % standardized innovations

    % Fills out covariance matrix for i != j
    for i = 1:n
        for j = 2:n
            if (j > i) 
                covariance = sqrt(PredCov(i,i)*PredCov(j,j))*corr(StdInnos(:,i), StdInnos(:,j));
                PredCov(i,j) = covariance;
                PredCov(j,i) = covariance;
            end
        end
    end

    % Predicts mean using  AR(1) model's one-step-ahead forecast
    PredMean = (phi(1,:))' +  diag(returns(m-1,:))*(phi(2,:)');
    Error = X_test' - PredMean; % difference between predicted mean and realized returns

end
