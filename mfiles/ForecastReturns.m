



function PredMean = ForecastReturns(r, fact)

    m = size(r, 1);
    n = size(r, 2); % number of assets
    
    % lagged returns
    X_lag = r(1:m-1, :);
    fact_lag = fact(1:m-1);

    % returns
    X = r(2:m, :);

    %Coeff = zeros(3,n); % initializes matrix of regression parameters
    Coeff = zeros(n, 3);
    
    X_fit = zeros(m-1,n); % initializes fit
    X_resid = zeros(m-1,n); % initializes residuals
    
    for i = 1:n
        %reg_mat = [ones(m-1,1) X_lag(:,i) SP_lag]; % regression matrix (includes y-intercept term)
        reg_mat = [ones(m-1,1) fact_lag X_lag(:,i)];
        
        %Coeff(:,i) = reg_mat\X(:,i); % solves for coeffs mu and phi
        Coeff(i,:) = reg_mat\X(:,i);
        
        
        X_fit(:,i) = reg_mat*Coeff(i,:)'; % fitted returns
        X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
    end
 
    %PredMean = (Coeff(1,:))' +  diag(r(m,:))*(Coeff(2,:)') + SP(m)*ones(n)*(Coeff(3,:)');
    %PredMean = Coeff* [1; SP(m); r(m,:)'];
    PredMean = Coeff(:,1) + Coeff(:,2)*fact_lag(m-1) +  diag(Coeff(:,3))*X_lag(m-1,:)';
    
end