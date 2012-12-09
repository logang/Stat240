classdef OneStepPrediction
   % Class for one-step-ahead prediction

   properties

     path_to_data = '../data/matfiles/All_variables_matched.mat'; % path to data
     predictor_matrix			% raw predictors
     target_matrix			% raw target variables
     target				% lagged univariate target
     design_matrix			% lagged design matrix
     lag = 10;				% lag used to construct data
     active_set = [1:60];		% window of indices on which to fit
     current_y				% current windowed regression target
     current_y_idx = 1;			% target index of current targed
     current_X	     			% current windowed design matrix
     center = 1;			% should predictors be standardized
     window_type = 'expanding';		% type of window
     fit	   			% container for current fit
     pred    				% container for current prediction
     predictions			% predictions for all targets over all windows
     test_X				% current held out predictors
     test_y				% current held out
     model_selection = 1;		% criterion for model selection (1) CV (2) AIC (3) BIC
     AICs	       			% current AICs
     BICs				% current BICs
     best_AIC				% current best AIC
     best_BIC				% current best BIC
     best_lambda			% current best lambda
     coefs				% current coefs
     num_vars_selected			% number of variables selected across fits
     num_coefs				% current number of coefs
     options				% options for fitting glmnet

   end

   methods

     % initialize class
     function obj = setup(obj)
       addpath('./glmnet_matlab/');
       obj = obj.load;
       obj = obj.get_lagged_design;
       obj.options = glmnetSet;
       obj.options.alpha = 0.8;
       obj.options.nlambda = 100;
       obj.options.lambda_min = 0.001;
       if obj.center == 0
          obj.options.standardize = 0;
       end
     end

     % load the data
     function obj = load(obj)
     	data_matrix = load(obj.path_to_data);
	obj.predictor_matrix = data_matrix.predictors;
	obj.target_matrix = data_matrix.targets;
     end

     % generate the design matrix of lagged predictors and targets
     function obj = get_lagged_design(obj)
        % get number of predictors
	n_targets = size(obj.target_matrix,2);
        n_inputs = size(obj.predictor_matrix,2);

	% construct lagged design matrix of lag obj.lag
	design_mat = [];
	for i = 1:n_targets
	   x = obj.target_matrix(:,i);
	   tmpmat = x;
	   for l = 1:obj.lag
	     tmpmat = [tmpmat, lagmatrix(x,l)];
	   end
	   disp(tmpmat);
    	   design_mat = [design_mat tmpmat];
	end
	for i = 1:n_inputs
	   x = obj.predictor_matrix(:,i);
	   tmpmat = x;
	   for l = 1:obj.lag
	     tmpmat = [tmpmat, lagmatrix(x,l)];
	   end
    	   design_mat = [design_mat tmpmat];
	end
	obj.design_matrix = design_mat( (obj.lag+1):end,: );
      end

      % Get active windowed variables, specifying which target variable using 'y_idx'
      function obj = get_current(obj,y_idx)
	% get active windowed regression design matrix and target
	idx = (y_idx-1)*(obj.lag+1)+1;
	obj.current_X = obj.design_matrix(obj.active_set,:);
	obj.current_X(:,idx) = [];
	obj.current_y = obj.design_matrix(obj.active_set,idx);

	% get test points for prediction
	last = obj.active_set(end)
	obj.test_X = obj.design_matrix(last+1,:);
	obj.test_X(:,idx) = [];
	obj.test_y = obj.design_matrix(last+1,idx);
      end
      
      % fit an elasic net regression 
      function obj = get_enet_fit(obj)
	% number of current observations
	n = size(obj.current_X,1);

	% fit full model to current
        obj.fit = glmnet(obj.current_X, obj.current_y, 'gaussian',obj.options);

	% model selection with 10-fold cross validation
	if obj.model_selection == 1
	  indices = crossvalind('Kfold',n,10);
	  best_lam = zeros(10,1);
	  for i = 1:10
	    test = (indices == i); train = ~test;
            train_fit = glmnet(obj.current_X(train,:), obj.current_y(train), 'gaussian', obj.options);
	    test_err = sum((glmnetPredict(train_fit,'response',obj.current_X(test,:)) - repmat(obj.current_y(test),1,length(train_fit.lambda))).^2)'
	    [best_err, best_ind] = min(test_err);
	    best_lam(i) = train_fit.lambda(best_ind);
	  end
	  obj.best_lambda = median(best_lam);
	end

	% model selection with information criteria
        err =  sum((glmnetPredict(obj.fit,'response',obj.current_X) - repmat(obj.current_y,1,length(obj.fit.lambda))).^2)'
	df = obj.fit.df;
	AICs = -err + (2*df)/n; obj.AICs = AICs;
	BICs = -err + (log(size(obj.current_X,1))*df)/n; obj.BICs = BICs;
	if obj.model_selection==2 % AIC
	   [best_AIC, best_idx] = min(AICs(2:end));
	   obj.best_AIC = best_AIC;
	   obj.best_lambda = obj.fit.lambdas(best_idx+1); 
	elseif obj.model_selection==3 % BIC
	   [best_BIC, best_idx] = min(BICs(2:end));
	   obj.best_BIC = best_BIC;
	   obj.best_lambda = obj.fit.lambdas(best_idx+1); 
	end
        obj.coefs = glmnetPredict(obj.fit,'coefficients','s', obj.best_lambda);
	obj.num_coefs = sum(obj.coefs ~= 0.0);
	disp(obj.num_coefs);
        obj.pred =  glmnetPredict(obj.fit, 'response', obj.test_X, obj.best_lambda);
      end

      function obj = fit_full_data(obj,window_size)
         % initialize
         n = size(obj.design_matrix,1);
	 n_periods = n - window_size - 1;
	 n_targets = size(obj.target_matrix,2);

	 % containers for predictions and number of variables kept
	 obj.predictions = zeros(n_periods,n_targets);
	 obj.num_vars_selected = zeros(n_periods,n_targets);

	 % run predictions for all periods and all targets
	 for i = 1:n_periods
	    if obj.window_type == 'expanding'
	       obj.active_set = [1:(window_size+i-1)];
	    else
	       obj.active_set = [i:(window_size+i-1)];
	    end
	    for j = 1:n_targets
	        obj = obj.get_current(j);
	    	obj = obj.get_enet_fit;
		obj.predictions(i,j) = obj.pred;
		obj.num_vars_selected(i,j) = obj.num_coefs;
	    end
	 end
      end

   end % methods
end % class     




