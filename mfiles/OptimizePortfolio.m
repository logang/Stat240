
% FUNCTION: OptimizePortfolio
% ---------------------------
% Computes an optimal portfolio and returns its characteristics.
% INPUTS:
%        mu_hat       -- nx1 vector of estimated excess returns
%        sigma_hat    -- nxn matrix of estimated covariances of excess
%                        returns
%        lb           -- lower bound on portfolio weights (e.g., 0 or -0.2)
%        bench_t      -- rate of return realized by benchmark fund @ next
%                        time point (S&P - LIBOR)
%        ret_t        -- rate of return realized assets at next time point
% 
% OUTPUTS:
%        w      -- nx1 vector of optimal weights
%        mu     -- mean expected return for weights w
%        ex_mu  -- excess return of optimal portfolio (over realized
%                  benchmark return)
%        sd     -- estimated standard deviation of portfolio with weights w
%        sharpe -- sharpe ratio of optimal portfolio for fixed lambda.
%                  Note that this is the *expected* Sharpe ratio (e.g.,
%                  mean returns are estimated and not known.)

function [w, mu, ex_mu, sd, sharpe] = OptimizePortfolio(mu_hat, sigma_hat, lb, bench_t, ret_t)
    
   n = length(mu_hat); % number of stocks

   lambda = 4; % fixes lambda rather than using a time-consuming grid search (although this probably would have improved results)
   
 
   % using cvx, a software package for convex programming.
   % solves standard M-V portfolio optimization problem with a risk
   % parameter of lambda.
   % Sets a bound on short selling and requires weights to sum to one.
      
   cvx_begin
      variable w(n)
      maximize ( mu_hat' * w - lambda * quad_form(w, sigma_hat) )
      subject to
          sum(w) == 1;
          w >= lb; % sets limit on short selling
   cvx_end
      
      
   % defines expected mean, standard deviation, and information ratio of
   % this M-V optimizing portfolio
   mu = mu_hat' * w;
   sd = sqrt( w' * sigma_hat * w);
   sharpe =  mu/sd;
      

   ex_mu = ret_t*w - bench_t; % excess *realized* returns over S&P at time t??

end
