% FUNCTION: fitStoRegGARCH
% ------------------------
% A commented version of Lai, Xing, Chen's code as linked to their paper,
% Mean-Variance Portfolio Optimization.

function [coeff, stdinno, sigmas, fitted, meanPred, ...
    secPred] = fitStoRegGARCH(Xtrain, u, winsize)

nStocks = size(Xtrain, 2); % number of stocks
coeff = zeros(nStocks, 6); % initializes matrix of coefficients (6 per stock)
inno = zeros( winsize-1, nStocks); % initializes innovations
stdinno = zeros( winsize-1, nStocks); % initializes standardized innovations
sigmas = inno;      fitted = inno; % initializes sigma and fit
covPred = zeros(nStocks,nStocks); % initializes covariance matrix

spec=garchset('R', 0, 'M', 0, 'P', 1, 'Q', 1, 'display', 'off'); % GARCH(1,1) model

for ind = 1:nStocks
    
    % regression matrix: 2 variables, including autoregressive component
    regMat = [u(1:(winsize-1))'; Xtrain(1:(winsize-1), ind)']';
    
    % time series
    regY = Xtrain(2:winsize, ind);
    
    % fits garch model
    [tmp,err,llf,inno(:,ind),sigmas(:,ind),...
            summary]=garchfit(spec,regY,regMat);
        
    % saves coefficients
    coeff(ind,:)=[tmp.C,tmp.Regress,tmp.K,tmp.GARCH,tmp.ARCH];
    
    % saves fit of model
    fitted(:,ind) = regMat*coeff(ind,2:3)'+coeff(ind,1);
    
    % saves variance of stock ind in covariance matrix
    % sigma_sq(t) = omega + beta(GARCH)*sigma(t-1) +
    % alpha(ARCH)*u(innovation(t-1))^2
    covPred(ind,ind) = tmp.K + tmp.GARCH*sigmas(end,ind)^2 ...
            + tmp.ARCH*inno(end,ind)^2;
end

% standardizes innovations by dividing by sigmas (time-varying standard
% deviation)
stdinno = inno./sigmas;

% uses model to predict mean
% mean = C + regression coefficient * u(??) + K (omega) * X_train
meanPred = coeff(:,1) + sum( coeff(:,2:3).*[u(winsize)* ...
    ones(nStocks,1), Xtrain(winsize,:)'],2);

% computes predicted covariances for each pair of stocks
% predicted correlation = correlation between standardized innovations of 2
% stocks
for ind1 = 2:nStocks
    for ind2 = 1:(ind1-1)
        covPred(ind1,ind2)=sqrt(covPred(ind1,ind1)*covPred(ind2,ind2))...
            *corr(stdinno(:,ind1),stdinno(:,ind2));
        covPred(ind2,ind1)=covPred(ind1, ind2);
    end
end
secPred = covPred + meanPred'*meanPred;


    
