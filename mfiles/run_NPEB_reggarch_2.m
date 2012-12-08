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
sp = SandPRets;

% Grid over which to search for best eta
%eta=1.0:0.5:10;
%eta = 1:10;
eta = 3:7;

% Number of bootstrap replicates
%B = 100;
B = 3;

% Define length of period
nPeriod=length(FF_dates)-start_winsize;

% initialize output containers
sharpe_train=zeros(nPeriod, 1);
ret_Value_npeb_iid = zeros(nPeriod, 1);

% Set grid of lambdas
%lambdas=2.^( (-3):1:11 );
lambdas = 1:5;

%% Constraints
lb= ones(nStocks, 1)*(-0.05);
ub=ones(nStocks, 1);

% initialize output containers
sharpe_train=zeros(nPeriod, 1);
ret_Value_npeb_srgar = zeros(nPeriod, 3);

for j = 1 : (length(FF_dates)-start_winsize)
    % Setup training data and held out test data point
    winsize = start_winsize + j -1;
    Xtrain = FF_data(1:(start_winsize), :);
    Xtest = FF_data(1+start_winsize, :);
    cursp = sp((j: (j + start_winsize )));
    
    for lam=1:length(lambdas)
        lambda = lambdas(lam);
        bi = randint(B, start_winsize-1, [1, start_winsize-1]);

    %%% Use Sto-Reg-GARCH model for each series
        [coeff, stdinno, sigmas, fitted, meanPred, ...
            secPred]=fitStoRegGARCH(Xtrain, cursp, start_winsize);   
        for b=1:B
            tmpinno = stdinno(bi(b,:), :);    
            bootsample = [Xtrain(1,:); fitted+(tmpinno.*sigmas)];
            [tmpcoeff, tmpinno, tmpsigmas, tmpfitted, tmpmeanPred, ...
                tmpsecPred] = fitStoRegGARCH(bootsample, cursp, start_winsize);
            for k=1:length(eta)
                tmpwt = getOptWt_Quadprog(tmpmeanPred',tmpsecPred,...
                    lambda/eta(k),lb,ub);
                C1(k,b) = tmpmeanPred'*tmpwt;
                C2(k,b) = tmpwt'*tmpsecPred*tmpwt;
            end
        end    
        indb = find( (~(mean(C1)>-1000) == 1) & (~(mean(C2)>-1000) == 1));
        if (length(indb)>0) 
            C1(:,indb)=0;   C2(:,indb)=0;
        end
        ind=find(mean(C1,1)<10^20 & mean(C2,1)<10^20);
        C1=C1(:,ind);   C2=C2(:,ind);
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
    ret_Value_npeb_srgar(j,:) = [lam, maxsharpe, rets_maxsharpe];
end

disp('NPEB GARCH returns were:')
disp(ret_Value_npeb_srgar)

% Save results
save('../results/NPEB_garch_returns_grid', 'ret_Value_npeb_srgar');
