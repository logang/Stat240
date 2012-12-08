

function PredMean = Forecast_OneFactor(r, f)

    m = size(r, 1);
    n = size(r, 2); % number of assets


    % returns
    X = r(2:m, :);
    f_lag = f(1:m-1);

    Coeff = zeros(n, 2);
    
    X_fit = zeros(m-1,n); % initializes fit
    X_resid = zeros(m-1,n); % initializes residuals
    
    reg_mat = [ones(m-1,1) f_lag];
    
    for i = 1:n
        
        %Coeff(:,i) = reg_mat\X(:,i); % solves for coeffs mu and phi
        Coeff(i,:) = (reg_mat\X(:,i))';
        
        
        X_fit(:,i) = reg_mat*Coeff(i,:)'; % fitted returns
        X_resid(:,i) = X(:,i) - X_fit(:,i); % residuals
    end
 
    %PredMean = (Coeff(1,:))' +  diag(r(m,:))*(Coeff(2,:)') + SP(m)*ones(n)*(Coeff(3,:)');
    PredMean = [Coeff(:,1) eye(n)] * [1 r(m,:)'];
end