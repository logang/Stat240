%%% This code is provided by Ledoit and Wolf 
%%% (2003, Journal of Empirical Finance) and downloaded
%%% from the author's website http://ledoit.net at
%%% Sept 15, 2008.

function [sigma, shrinkage, prior]=covcorr(x)

% function sigma=covcorr(x)
% x (t*n): t iid observations on n random variables
% sigma (n*n): invertible covariance matrix estimator
%
% Shrinks towards constant correlation matrix

% de-mean returns
[t,n]=size(x);
meanx=mean(x);
x=x-meanx(ones(t,1),:);

% compute sample covariance matrix
sample=(1/t).*(x'*x);

% compute prior
var=diag(sample);
sqrtvar=sqrt(var);
rho=(sum(sum(sample./(sqrtvar(:,ones(n,1)).*sqrtvar(:,ones(n,1))')))-n)/(n*(n-1));
prior=rho*sqrtvar(:,ones(n,1)).*sqrtvar(:,ones(n,1))';
prior(logical(eye(n)))=var;

% compute shrinkage parameters
d=1/n*norm(sample-prior,'fro')^2;
y=x.^2;
r2=1/n/t^2*sum(sum(y'*y))-1/n/t*sum(sum(sample.^2));
phidiag=1/n/t^2*(sum(sum(y.^2))-1/n/t*sum(diag(sample).^2));
v=((x.^3)'*x)/(t^2)-(var(:,ones(1,n)).*sample)/t;
v(logical(eye(n)))=zeros(n,1);
phioff=sum(sum(v.*(sqrtvar(:,ones(n,1))./sqrtvar(:,ones(n,1))')))/n;
phi=phidiag+rho*phioff;

% compute the estimator
shrinkage=max(0,min(1,(r2-phi)/d));
sigma=shrinkage*prior+(1-shrinkage)*sample;

 
	
