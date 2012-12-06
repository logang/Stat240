# Import libraries
library(timeSeries)
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
# Fit AR(1) model
ar_fits = FF_garch_fits = FF_garch_std_fits = list()
for(i in 1:m)
{
  FF_ts = timeSeries(FF[,i],as.character(dates))
  ar_fits[[i]] = ar(FF[,i])
  resids_ts = na.omit(timeSeries(ar_fits[[i]]$resid,as.character(dates)))
  FF_garch_fits[[i]] = garchFit(. ~ garch(1,1), data=resids_ts)
  FF_garch_std_fits[[i]] = garchFit(formula = ~ garch(1,1), data=resids_ts, cond.dist="std")
}

# Construct covariance matrix
