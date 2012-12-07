% Retrieves monthly returns from data
smlo_ret = FF(:,2);
smme_ret = FF(:,5);
smhi_ret = FF(:,8);
bilo_ret = FF(:,11);
bime_ret = FF(:,14);
bihi_ret = FF(:,17);
X_rets = [smlo_ret smme_ret smhi_ret bilo_ret bime_ret bihi_ret]; % matrix of returns

m = length(smlo_ret); % number of observations
n = 6; % number of stocks

% lagged returns (training)
X_lag = X_rets(1:m-2, :);

% returns (training)
X = X_rets(2:m-1, :);

% returns (testing)
X_test = X_rets(m,:);

phi = zeros(2,n);
X_fit = zeros(m-2,n);
X_resid = zeros(m-2,n);

% regresses each return on its lags to yield phi (AR(1) model)
for i = 1:n
    reg_mat = [ones(m-2,1) X_lag(:,i)]; % regression matrix (includes y-intercept term)
    phi(:,i) = reg_mat\X(:,i); % solves for coeffs mu and phi
    X_fit(:,i) = reg_mat*phi(:,i); % fitted returns
    X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
end

% sets specs for GARCH(1,1) model of residuals from AR(1) fit
spec = garchset('R', 0, 'M', 0, 'P', 1, 'Q', 1, 'display', 'off');

Coeff = zeros(n,4); % initializes coefficient matrix (one row per asset)
Innos = zeros(m-2,n); % initializes matrix of innovations u
Sigmas = zeros(m-2,n); % initializes matrix of sigmas
PredCov = zeros(n,n); % initializes one-step-ahead predicted covariance matrix
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
mean_predict = (phi(1,:))' +  diag(X_rets(m-1,:))*(phi(2,:)');


