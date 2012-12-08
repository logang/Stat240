


function PredMean = Forecast_AR1(r)

    m = size(r, 1);
    n = size(r, 2); % number of assets
    
    % lagged returns
    X_lag = r(1:m-1, :);

    % returns
    X = r(2:m, :);

    Coeff = zeros(n, 2);
    
    X_fit = zeros(m-1,n); % initializes fit
    X_resid = zeros(m-1,n); % initializes residuals
    
    for i = 1:n
        %reg_mat = [ones(m-1,1) X_lag(:,i) SP_lag]; % regression matrix (includes y-intercept term)
        reg_mat = [ones(m-1,1) X_lag(:,i)];
        
        %Coeff(:,i) = reg_mat\X(:,i); % solves for coeffs mu and phi
        Coeff(i,:) = (reg_mat\X(:,i))';
        
        
        X_fit(:,i) = reg_mat*Coeff(i,:)'; % fitted returns
        X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
    end
 
    %PredMean = (Coeff(1,:))' +  diag(r(m,:))*(Coeff(2,:)') + SP(m)*ones(n)*(Coeff(3,:)');
    PredMean = [Coeff(:,1) eye(n)] * [1 r(m,:)'];
end