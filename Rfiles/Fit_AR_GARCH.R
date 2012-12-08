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
beta_CIs = alpha_CIs = coefs = matrix(0,m,2)
residuals = estimates = matrix(0,n,m)
for(i in 1:m)
{
  # Fit the linear model
  fit = lm(FF[,i]~SandP500)
  # Get parameters and 95% intervals for the parameters
  coefs[i,]=fit$coefficients
  residuals[,i] = fit$residuals
  estimates[,i] = predict(fit)
  alpha_CIs[i,]=confint(fit)[2,]
  beta_CIs[i,]=confint(fit)[2,]
}

# Plot fits and residuals
pdf("figures/Problem1a_FamaFrench_linear_fit_plots.pdf",height=7, width=9)
par(mfrow=c(2,3))
for(i in 1:m)
{
  FF_ts = timeSeries(FF[,i])
  est_ts = timeSeries(estimates[,i])
  plot(FF_ts,col="black",xlab="Time (months)",ylab=colnames(FF)[i], main=paste("Linear Fit for",colnames(FF)[i]))
  lines(est_ts,col="red")
}
dev.off()

# Plot fit residuals
resids_ts = timeSeries(residuals)
pdf("figures/Problem1a_FamaFrench_residal_plots.pdf")
colnames(resids_ts) = colnames(FF)
plot(resids_ts,col="black",xlab="Time (months)", main="Residuals for Fit to Fama-French returns")
dev.off()

# Print 95% confidence intervals to LaTeX
alpha_results = cbind(coefs[,1], alpha_CIs)
beta_results = cbind(coefs[,2], beta_CIs)
print(xtable(alpha_results,digits=4), type="latex", file="figures/tables/Problem1a_alpha_results.tex")
print(xtable(beta_results),digits=4, type="latex", file="figures/tables/Problem1a_beta_results.tex")

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
ar_fits = FF_garch_fits = FF_garch_st_fits = list()
ar_coefs = matrix(0,m,2)
cov_pred = matrix(0,m,m)
residmat = matrix(0,n-2,m)
std_innovations = std_innovations_st = matrix(0,n-2,m)
coefmat = coefmat_st = matrix(0,m,4)
ar_residmat = matrix(0,n-2,m)

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
  ar_residmat[,i] = resids_ts
  
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
  FF_garch_st_fits[[i]] = garchFit(formula = ~ garch(1,1), data=resids_ts, cond.dist="std",trace=F)

  # Gather coefficents, volatilities, and residuals  
  coefs = coef(FF_garch_fits[[i]])
  coefmat[i,] = coefs
  garch_resids = residuals(FF_garch_fits[[i]])
  vols = volatility(FF_garch_fits[[i]])
  
  # Gather coefficents, volatilities, and residuals for Studentized GARCH fit
  coefs_st = coef(FF_garch_st_fits[[i]])
  coefmat_st[i,] = coefs_st[1:4]
  garch_resids_st = residuals(FF_garch_st_fits[[i]])
  vols_st = volatility(FF_garch_st_fits[[i]])
  
  # Construct one-step-ahead covariance matrix diagonal
  cov_pred[i,i] = coefs[2] + coefs[4]*vols[n-2]^2 + coefs[3]*resids[n-2]^2
  
  # Calculate standardized innovations
  std_innovations[,i]=garch_resids/vols
  std_innovations_st[,i]=garch_resids_st/vols_st
}

# Save coefficents to LaTeX
print(xtable(ar_coefs,digits=4), type="latex", file="figures/tables/Problem1b_ar_coefs_results.tex")
print(xtable(coefmat,digits=4), type="latex", file="figures/tables/Problem1b_coef_results.tex")
print(xtable(coefmat_st,digits=4), type="latex", file="figures/tables/Problem1b_coef_studentized_results.tex")

# Plot AR(1) fit residuals
ar_residmat = timeSeries(ar_residmat)
colnames(ar_residmat) = colnames(FF)
pdf("figures/Problem1b_AR1_fit_residuals.pdf")
plot(ar_residmat,col="black",main="Residuals from AR(1) fits to Fama French portfolios")
dev.off()

# Plot GARCH(1,1) fit standard innovations
garch_inno = timeSeries(std_innovations)
colnames(garch_inno) = colnames(FF)
pdf("figures/Problem1b_GARCH1_1_fit_std_innovations.pdf")
plot(garch_inno,col="black",main="Innovations from GARCH(1,1) fits to F-F portfolios")
dev.off()

# Plot GARCH(1,1) fit standard innovations
garch_inno_st = timeSeries(std_innovations_st)
colnames(garch_inno_st) = colnames(FF)
pdf("figures/Problem1b_GARCH1_1_fit_std_innovations_studentized.pdf")
plot(garch_inno_st,col="black",main="Innovations from GARCH(1,1) studentized fits to F-F portfolios")
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
print(xtable(mean_pred,digits=4), type="latex", file="figures/tables/Problem1c_mean_pred.tex")
print(xtable(cov_pred,digits=4), type="latex", file="figures/tables/Problem1c_cov_pred.tex")

#---------------------------- PROBLEM 1D ----------------------------------
# MATLAB

