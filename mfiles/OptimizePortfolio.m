
% FUNCTION: OptimizePortfolio
% ---------------------------
% Computes an optimal portfolio and returns its characteristics.
% INPUTS:
%        mu_hat       -- nx1 vector of estimated returns
%        sigma_hat    -- nxn matrix of estimated covariances
%        lb           -- lower bound on portfolio weights (e.g., 0 or -0.2)
%        bench_mu     -- benchmark rate of return (mean S&P or bond rate)
%        bench_t      -- rate of return realized by benchmark fund @ next
%                        time point
%        ret_t        -- rate of return realized by optimal portfolio @
%                        next time point
% OUTPUTS:
%        w      -- nx1 vector of optimal weights
%        mu     -- mean expected return for weights w
%        ex_mu  -- excess return of optimal portfolio (over realized
%                  benchmark return)
%        sd     -- estimated standard deviation of portfolio with weights w

function [w, mu, ex_mu, sd] = OptimizePortfolio(mu_hat, sigma_hat, lb, bench_mu, bench_t, ret_t)
    
   n = length(mu_hat); % number of stocks

   lambda = 0:1:5; % initializes lambda for grid search, which has
   m = length(lambda); % m values
   
   % initializes mean return, standard deviation, optimal weights, and
   % information ratio. We define the optimal portfolio as the one that
   % maximizes the information ratio using the average S&P 500 returns
   % (bench_mu) as a benchmark.
   mu = 0;
   sd = 0;
   w_max = 0;
   info_max = -1000;
   
   % grid search over values of lambda to maximize expected information
   % ratio
   for i = 1:m
   
      % using cvx, a software package for convex programming.
      % solves standard M-V portfolio optimization problem with a risk
      % parameter of lambda(i).
      % Sets a bound on short selling and requires weights to sum to one.
      cvx_begin
        variable w(n)
        maximize ( mu_hat' * w - lambda(i) * quad_form(w, sigma_hat) )
        subject to
            sum(w) == 1;
            w >= lb; % sets limit on short selling
      cvx_end
      
      
      % defines expected mean, standard deviation, and information ratio of
      % this M-V optimizing portfolio
      mu_temp = mu_hat' * w;
      sd_temp = sqrt( w' * sigma_hat * w);
      info_ratio =  (mu_temp - bench_mu)/sd_temp;
      
      % if information ratio exceeds previous max, redefines max and its
      % associated mean, sd, and weights.
      if (info_ratio > info_max) 
          info_max = info_ratio;
          mu = mu_temp;
          sd = sd_temp;
          w_max = w;
      end
      
   end

   w = w_max; % weights that maximize information ratio
   ex_mu = ret_t*w - bench_t; % excess *realized* returns over S&P at time t

end
