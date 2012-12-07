# Import libraries
library(timeSeries)
library(tseries)
library(fGarch)

# Setup and load data
setwd("/Users/logang/Documents/Code/python/Stat240")
SandP500 = as.matrix(read.csv("data/SandP500_returns.csv",header=F))
FF = as.matrix(read.csv("data/FF_returns.csv",header=F))
dates = read.csv("data/dates.csv",header=F)[,1]
colnames(FF) = c('smlo_ret', 'smme_ret', 'smhi_ret', 'bilo_ret', 'bime_ret', 'bihi_ret')

# get number of observations and stocks
n = dim(FF)[1]
m = dim(FF)[2] 

# Regress Fama-French returns on SandP500 
fit = lm(FF~SandP500)

# -- Fit AR(1)-GARCH(1,1)
# Leave the last point out for prediction
FF_test = FF[n,]
FF = FF[1:(n-1),]
dates = dates[1:(n-1)]
# Fit AR(1) model
ar_fits = FF_garch_fits = FF_garch_std_fits = list()
covmat = matrix(0,m,m)
std_innovations = matrix(0,n-2,m)
coefmat = matrix(0,m,4)
for(i in 1:m)
{
  # generate times
  FF_ts = timeSeries(FF[,i],as.character(dates))
  
  # fit AR(1) model and get residuals
  ar_fits[[i]] = arma(FF[,i],order=c(1,0))
  resids_ts = na.omit(timeSeries(ar_fits[[i]]$resid,as.character(dates)))
  
  # Fit standard and studentized GARCH models to residuals of AR(1) model
  FF_garch_fits[[i]] = garchFit(. ~ garch(1,1), data=resids_ts,trace=F)
  FF_garch_std_fits[[i]] = garchFit(formula = ~ garch(1,1), data=resids_ts, cond.dist="std",trace=F)

  # Construct covariance matrix 
  coefs = coef(FF_garch_fits[[i]])
  coefmat[i,] = coefs
  resids = residuals(FF_garch_fits[[i]])
  vols = volatility(FF_garch_fits[[i]])
  covmat[i,i] = coefs[2] + coefs[4]*vols[n-2]^2 + coefs[3]*resids[n-2]^2
  std_innovations[,i]=resids/vols
}

# Estimate mean 
mu_pred = coefs[,1] + sum(coefs[,2:3].*[u(winsize)*ones(nStocks,1), Xtrain(winsize,:)'],2);

# Generate off-diagonal covariance matrix entries
for(i in 2:m)
{
  for(j in 1:(i-1))
  {
    cov_pred[i,j]=sqrt(cov_pred[i,i]*cov_pred[j,j])*cor(std_innovations[,i],std_innovations[,j])
    cov_pred[j,i]=cov_pred[i,j]
  }
}
