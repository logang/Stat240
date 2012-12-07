% FUNCTION: GetTurnoverRate
% -------------------------
% This function defines the turnover rate for a given set of weights (past
% and current).
% Since the amount invested is assumed to be the same on every iteration,
% we calculate in terms of weights and do not need to normalize by total
% amount invested (sum of weights presumed to be 1).

function T = GetTurnoverRate(weights_cur, weights_past)

   T = sum( abs(weights_cur - weights_past));

end