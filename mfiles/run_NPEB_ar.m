
warning off;
MAXFUNEVAL = 100;
winsize=120;

allmonth=load('../0_Rawdata_done/list_month.txt');
prefix2='../0_RawData_done/Stocks_Use_Value/stock_use_Value_';
%%% allmonth(241)=19940131, allmonth(253)=19950131, 
startmonth=252;

lambdas=2.^( (-3):1:11 );
eta=1.0:0.5:10;
B = 100;
nStocks=50;
lb= ones(nStocks, 1)*(-0.05);		ub=ones(nStocks, 1);

nPeriod=length(allmonth)-startmonth;
sharpe_train=zeros(nPeriod, 1);
res_sharpe=zeros(nPeriod, length(lambdas));
res_rets=zeros(nPeriod, length(lambdas));
ret_Value_npeb_ar = zeros(nPeriod, 2);

for j = 1 : (length(allmonth)-startmonth)
    filename = strcat(prefix2, int2str(allmonth(j+startmonth)), '.txt');
    data = load(filename);
    Xtrain = data(1:winsize, :);
    Xtest = data(1+winsize, :);
    
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



