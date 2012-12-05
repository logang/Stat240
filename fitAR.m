
function [coeff, stdinno, sigmas, fitted, meanPred, ...
    secPred] = fitAR(Xtrain, winsize)

nStocks = size(Xtrain, 2);      coeff = zeros(nStocks, 2);
inno = zeros(winsize-1, nStocks);
stdinno = inno;         fitted=inno;    sigmas=zeros(1,nStocks);
covPred = zeros(nStocks,nStocks);

for ind = 1:nStocks
    regMat = [ones(1,winsize-1); Xtrain(1:(winsize-1), ind)']';
    regY = Xtrain(2:winsize, ind);
    coeff(ind,:) = regress(regY, regMat)';
    fitted(:,ind) = regMat * coeff(ind,:)';
    inno(:,ind) = regY-fitted(:,ind);
    stdinno(:,ind)=inno(:,ind)/std(inno(:,ind));
    sigmas(ind) = var(inno(:,ind));
    covPred(ind,ind) = sigmas(ind);
end
meanPred = coeff(:,1) + coeff(:,2).*Xtrain(winsize,:)';

for ind1 = 2:nStocks
    for ind2 = 1:(ind1-1)
        covPred(ind1,ind2)=sqrt(sigmas(ind1)*sigmas(ind2))...
            *corr(stdinno(:,ind1),stdinno(:,ind2));
        covPred(ind2,ind1)=covPred(ind1, ind2);
    end
end
secPred = covPred + meanPred'*meanPred;


    
