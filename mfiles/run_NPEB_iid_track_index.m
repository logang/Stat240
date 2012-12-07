%%% The input data are extracted from the CRSP database via the WRDS.
%%% The return data used here are raw returns.

warning off;     
clear all;
MAXFUNEVAL = 100;

start_winsize=120; % Number of time points considered per fit
portsize=6; % Size of portfolio

% load dates, portfolio data, and market cap weights
FF_dates=csvread('../data/dates.csv');
FF_data=importdata('../data/FF6Portfolios.txt', ' ', 3);
FF_data=FF_data.data;
indexweight=load('../data/NPEB_wts.mat');
indexweight = indexweight.NPEB_wts;

% Grid over which to search for best eta
%eta=1.0:0.5:10;
eta = 1:10;

% Number of bootstrap replicates
B = 2;

% Define length of period
nPeriod=length(FF_dates)-start_winsize;

% initialize output containers
sharpe_train=zeros(nPeriod, 1);
ret_Value_npeb_iid = zeros(nPeriod, 1);

format long;
lambda = 3.4;

for j = 1 :(length(FF_dates)-start_winsize)
    % Setup training data and held out test data point
    winsize = start_winsize + j - 1;
    Xtrain = FF_data(1:(start_winsize+j), :);
    Xtest = FF_data(1+(start_winsize+j), :);

    % Set lower and upper bounds on weights
    %lb = - indexweight(j,:)';
    lb = - indexweight';
    %ub = ones(portsize, 1)*0.10 - indexweight(j,:)';
    ub = ones(portsize,1)*0.10 - indexweight';

    bi = randint(B, winsize-1, [1, winsize-1]);
    %%% Use iid model for each series
    for b=1:B
        tmpret=Xtrain(bi(b,:), :);    
        tmpMu=mean(tmpret);
        tmpV=covcorr(tmpret)+tmpMu*tmpMu';
        for k=1:length(eta)           
            tmpwt = getOptWt_Quadprog(tmpMu',tmpV,lambda/eta(k),lb,ub);
            C1(k,b) = tmpMu*tmpwt;
            C2(k,b) = tmpwt'*tmpV*tmpwt;
        end
    end
    Cfun=mean(C1,2)-lambda*mean(C2,2)+lambda*mean(C1,2).^2;
    opteta = eta(Cfun==max(Cfun));
    meanPred = mean(Xtrain);
    secPred = meanPred'*meanPred+covcorr(Xtrain);
    if (length(opteta)>1)  opteta=opteta(1);   end
    wts = getOptWt_Quadprog(meanPred,secPred,lambda/opteta, lb, ub);
    ret_train = Xtrain*wts;
    sharpe_train = mean(ret_train)/std(ret_train);
    
    ret_Value_npeb_iid(j) = Xtest*wts;
end


disp('NPEB IID returns were:')
disp(ret_Value_npeb_iid)

% Clumsily ham-fist the data into a file, MATLAB style!
lam = num2str(lambda);
lam_split = regexp(lam,'\.','split');
savefile = [strcat('NPEB_iid_returns_lambda_',lam_split(1),'_',lam_split(2))];
save(savefile{1}, 'ret_Value_npeb_iid'); % I hate you MATLAB
