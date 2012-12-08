
warning off;
MAXFUNEVAL = 100;

start_winsize=120; % Number of time points considered per fit
nStocks=6;  % Size of portfolio

% load dates, portfolio data, and market cap weights
% load dates, portfolio data, and market cap weights
FF_dates=csvread('../data/dates.csv');
FF_data=importdata('../data/FF6Portfolios.txt', ' ', 3);
FF=FF_data.data;
smlo_ret = FF(:,2);
smme_ret = FF(:,5);
smhi_ret = FF(:,8);
bilo_ret = FF(:,11);
bime_ret = FF(:,14);
bihi_ret = FF(:,17);
FF_data = [smlo_ret smme_ret smhi_ret bilo_ret bime_ret bihi_ret]; % matrix of returns
indexweight=load('../data/NPEB_wts.mat');
indexweight = indexweight.NPEB_wts;

% Grid over which to search for best eta
eta=1.0:0.5:10;
%eta = 1:10;

% Number of bootstrap replicates
B = 100;

% Define length of period
nPeriod=length(FF_dates)-start_winsize;

% initialize output containers
sharpe_train=zeros(nPeriod, 1);
ret_Value_npeb_iid = zeros(nPeriod, 1);

% set grid of lambdas
lambdas=2.^( (-3):1:11 );

%% Constraints
lb= ones(nStocks, 1)*(-0.05);
ub=ones(nStocks, 1);

% initialize output containers
sharpe_train=zeros(nPeriod, 1);
res_sharpe=zeros(nPeriod, length(lambdas));
res_rets=zeros(nPeriod, length(lambdas));
ret_Value_npeb_ar = zeros(nPeriod, 2);

for j = 1 : (length(FF_dates)-start_winsize)
    % Setup training data and held out test data point
    winsize = start_winsize + j - 1;
    Xtrain = FF_data(1:(start_winsize+j), :);
    Xtest = FF_data(1+(start_winsize+j), :);

    for lam=1:length(lambdas)
        lambda = lambdas(lam);
        bi = randint(B, winsize-1, [1, winsize-1]);
    
        %%% Use Sto-Reg model for each series
        [coeff, stdinno, sigmas, fitted, meanPred, ...
            secPred]=fitAR(Xtrain, winsize);
        for b=1:B
            tmpinno = stdinno(bi(b,:), :);
            bootsample = [Xtrain(1,:); fitted + ...
                tmpinno.*(ones(winsize-1,1)*sigmas) ];
            [tmpcoeff, tmpinno, tmpsigmas, tmpfitted, tmpmeanPred, ...
                tmpsecPred] = fitAR(bootsample, winsize);
            for k=1:length(eta)
                tmpwt = getOptWt_Quadprog(tmpmeanPred',tmpsecPred,...
                    lambda/eta(k),lb,ub);
                C1(k,b) = tmpmeanPred'*tmpwt;
                C2(k,b) = tmpwt'*tmpsecPred*tmpwt;
            end
        end
        indb = find( ~(mean(C1)>-1000) == 1);
        if (length(indb)>0)
            C1(:,indb)=0;   C2(:,indb)=0;
        end
        Cfun=mean(C1,2)-lambda*mean(C2,2)+lambda*mean(C1,2).^2;    
        opteta = eta(Cfun==max(Cfun));
        if (length(opteta)>1)  opteta=opteta(1);   end
        wts = getOptWt_Quadprog(meanPred,secPred,lambda/opteta, lb, ub);
        ret_train = Xtrain*wts;
        sharpe_train = mean(ret_train)/std(ret_train);
        rets =Xtest*wts;
    
        if lam==1
            maxsharpe=sharpe_train;  rets_maxsharpe=rets;
        end
        if maxsharpe<= sharpe_train
            maxsharpe = sharpe_train; rets_maxsharpe = rets;
        else
            break;
        end
        [j, lam, maxsharpe, rets_maxsharpe]
    end    
    ret_Value_npeb_ar(j,:) = [maxsharpe, rets_maxsharpe];
end

disp('NPEB AR returns were:')
disp(ret_Value_npeb_ar)

% Save results
<<<<<<< Updated upstream
save('../results/NPEB_ar_returns_grid', 'ret_Value_npeb_ar');
=======
save('NPEB_ar_returns_grid', 'ret_Value_npeb_ar');
>>>>>>> Stashed changes

