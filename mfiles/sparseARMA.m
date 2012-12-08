function [pred] = sparseARMA(target, input_matrix, lag_vector)

% sparseARMA takes a target matrix and a matrix of possible predictors,
% lags the target and predictors by the number of lags specified in lag_vector
% and makes a one-step-ahead prediction.
%
% Both target and input_matrix should have the most recent dates first. 
% The lag vector should be 1 element longer than the number of columns
% of input_matrix, with the first element corresponding to the lagging
% of the target variable.

% Add paths to relavant libraries 
addpath('./glmnet_matlab');

n_inputs = size(input_matrix,2);
lagged_mat = lagmatrix(target,lag_vector[1]);
for i in 1:n_inputs
    lagged_mat = [lagged_mat lagmatrix(input_matrix(:,i),lag_vector[i+1])];

% trim matrix to accommodate maximum lag
max_lag = max(lag_vector);

y = target((mat_lag+1):end);
X = lagged_mat((mat_lag+1):end,:));
fit = glmnet(x,y);
pred = glmnetPredict(fit,'response',x(1:10,:),[0.01,0.005]') 