%%% The input data are extracted from the CRSP database via the WRDS.
%%% The return data used here are raw returns.

warning off;        MAXFUNEVAL = 100;

%%% Compute the market value weight portfolio (i.e., index)
%winsize=120;        portsize=50;
%allmonth=load('../0_Rawdata_done/list_month.txt');
%prefix2='../0_RawData_done/Stocks_Use_Value/stock_use_Value_';
%indexweight=load('stock_Value_based_index_weight.txt');

winsize=120; % Number of time points considered per fit
portsize=6; % Size of portfolio

% load dates, portfolio data, and market cap weights
dates=load('../data/dates.csv');
FF_data=load('../data/FF6Portfolios.txt');
indexweight=load('../data/NPEB_wts.mat');

% starting time point
startmonth=252;

% Grid over which to search for best eta
eta=1.0:0.5:10;

% Number of bootstrap replicates
B = 100;

nPeriod=length(FF_data)-startmonth;
sharpe_train=zeros(nPeriod, 1);
ret_Value_npeb_iid = zeros(nPeriod, 1);

format long;
lambda = 1.4;

for j = 1 : (length(FF_data)-startmonth)
    filename = strcat(prefix2, int2str(FF_data(j+startmonth)), '.txt');
    data = load(filename);
    Xtrain = data(1:winsize, :);
    Xtest = data(1+winsize, :);
    lb = - indexweight(j,:)';
    ub = ones(portsize, 1)*0.10 - indexweight(j,:)';

    bi = randint(B, winsize-1, [1, winsize-1]);
    %%% Use iid model for each series
    for b=1:B
        tmpret=Xtrain(bi(b,:), :);    tmpMu=mean(tmpret);
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
