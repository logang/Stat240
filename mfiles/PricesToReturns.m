
% FUNCTION: PricesToReturns
% -------------------------
% This function takes a vector of m prices as its argument and returns a
% vector of m-1 returns (not log returns).
function r = PricesToReturns(p) 

    m = length(p);
    r = zeros(m-1,1);
    
    for i = 1:m-1
        r(i) = (p(i+1) - p(i))/p(i);
    end

end