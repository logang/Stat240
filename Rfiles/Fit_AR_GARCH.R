# Import libraries
library(timeSeries)
library(tseries)
library(fGarch)
library(xtable)

#------------------------------ SETUP ------------------------------------

# Setup top directory
setwd("/Users/logang/Documents/Code/python/Stat240")

# Load data
SandP500 = as.matrix(read.csv("data/SandP500_returns.csv",header=F))
FF = as.matrix(read.csv("data/FF_returns.csv",header=F))
dates = read.csv("data/dates.csv",header=F)[,1]
colnames(FF) = c('smlo_ret', 'smme_ret', 'smhi_ret', 'bilo_ret', 'bime_ret', 'bihi_ret')

# get number of observations and stocks
n = dim(FF)[1]
m = dim(FF)[2] 

#---------------------------- PROBLEM 1A ----------------------------------

# Regress Fama-French returns on SandP500
confidence_intervals = matrix(0,m,2)
for(i in 1:m)
{
  # Fit the linear model
  fit = lm(FF[,i]~SandP500)
  
  # Plot the residuals and QQ-plot as pdf
  pdf(paste("figures/Problem1a_FamaFrench_residal_plots",colnames(FF)[i],".pdf",sep=""))
  par(mfrow=c(2,2))
  plot(fit)
  dev.off()
  
  # Get 95% intervals for the parameters
  confidence_intervals[i,]=confint(fit)[2,]
}

# Print 95% confidence intervals to LaTeX
print(xtable(confidence_intervals), type="latex", file="figures/tables/Problem1a_CIs.tex")

#---------------------------- PROBLEM 1B & C --------------------------------
# -- Fit AR(1)-GARCH(1,1)
# Leave the last point out for prediction
FF_test = FF[n,]
FF = FF[1:(n-1),]
dates = dates[1:(n-1)]

# Plot Fama French time series
pdf("figures/Problem1b_return_time_series.pdf")
FF_ts = timeSeries(FF)
plot(FF_ts,col="black",main="Fama French Portfolio Returns")
dev.off()

# Initialize containers for AR(1) model results
ar_fits = FF_garch_fits = FF_garch_std_fits = list()
ar_coefs = matrix(0,m,2)
cov_pred = matrix(0,m,m)
residmat = matrix(0,n-2,m)
std_innovations = matrix(0,n-2,m)
coefmat = matrix(0,m,4)

# Fit AR(1)-GARCH(1,1) models to the Fama French portfolio returns
for(i in 1:m)
{
  # generate times
  FF_ts = timeSeries(FF[,i],as.character(dates))
  
  # Get ACF and PACF of data
  FF_acf = acf(FF[,i])
  FF_pacf = pacf(FF[,i])
  
  # fit AR(1) model and get coefficients and residuals
  ar_fits[[i]] = arma(FF[,i],order=c(1,0))
  ar_coefs[i,] = ar_fits[[i]]$coef
  resids_ts = na.omit(timeSeries(ar_fits[[i]]$resid,as.character(dates)))
  residmat[,i] = resids_ts
  
  # Get ACF and PACF of residuals
  FF_acf_resids = acf(resids_ts)
  FF_pacf_resids = pacf(resids_ts)
  
  # Plot before and after ACF and PACF
  pdf(paste("figures/Problem1b_ACF_and_PACFs_",colnames(FF)[i],".pdf"))
  par(mfrow=c(2,2))
  plot(FF_acf, main=paste("ACF for",colnames(FF)[i]))
  plot(FF_pacf,main=paste("PACF for",colnames(FF)[i]))
  plot(FF_acf_resids, main=paste("ACF of residuals for",colnames(FF)[i]))
  plot(FF_pacf_resids, main=paste("ACF of residuals for",colnames(FF)[i]))
  dev.off()
    
  # Fit standard and studentized GARCH models to residuals of AR(1) model
  FF_garch_fits[[i]] = garchFit(. ~ garch(1,1), data=resids_ts,trace=F)
  FF_garch_std_fits[[i]] = garchFit(formula = ~ garch(1,1), data=resids_ts, cond.dist="std",trace=F)

  # Gather coefficents, volatilities, and residuals  
  coefs = coef(FF_garch_fits[[i]])
  coefmat[i,] = coefs
  resids = residuals(FF_garch_fits[[i]])
  vols = volatility(FF_garch_fits[[i]])
  
  # Construct one-step-ahead covariance matrix diagonal
  cov_pred[i,i] = coefs[2] + coefs[4]*vols[n-2]^2 + coefs[3]*resids[n-2]^2
  
  # Calculate standardized innovations
  std_innovations[,i]=resids/vols
}

# Plot AR(1) fit residuals
residmat = timeSeries(residmat)
colnames(residmat) = colnames(FF)
pdf("figures/Problem1b_AR1_fit_residuals.pdf")
plot(residmat,col="black",main="Residuals from AR(1) fits to Fama French portfolios")
dev.off()

# Estimate one-step-ahead mean 
mean_pred = ar_coefs[,2] + diag(ar_coefs[,1])%*%FF[n-1,]

# Generate off-diagonal one-step-ahead covariance matrix entries
for(i in 2:m)
{
  for(j in 1:(i-1))
  {
    cov_pred[i,j]=sqrt(cov_pred[i,i]*cov_pred[j,j])*cor(std_innovations[,i],std_innovations[,j])
    cov_pred[j,i]=cov_pred[i,j]
  }
}

# Print predicted mean and covariance to LaTeX
print(xtable(mean_pred), type="latex", file="figures/tables/Problem1c_mean_pred.tex")
print(xtable(cov_pred), type="latex", file="figures/tables/Problem1c_cov_pred.tex")

#---------------------------- PROBLEM 1D ----------------------------------
# MATLAB

