function  opteta=getOptimalEta_Grid(logret, lambda, B, lb, ub)

N = size(logret, 1);    p=size(logret, 2);
C1=zeros(B, 1);     C2=zeros(B, 1);
eta=3.5:0.2:6.5;     KK=length(eta);

Cfun=zeros(KK, 1);
for k=1:KK
    bi = randint(B, N, [1, N]);
    for b=1:B
        tmpret=logret(bi(b,:), :);
        tmpMu=mean(tmpret);
        tmpV=covcorr(tmpret)+tmpMu*tmpMu';
        tmpwt = getOptWt_Quadprog(tmpMu', tmpV, lambda/eta(k), lb, ub);
        C1(b) = tmpMu*tmpwt;
        C2(b) = tmpwt'*tmpV*tmpwt;
    end
    Cfun(k) = mean(C1)-lambda*mean(C2)+lambda*mean(C1)^2;
end
opteta = eta(Cfun==max(Cfun));
